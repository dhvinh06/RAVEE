"""One video, all FINCH segments, one Gemini generation; no labels committed.

This cache and prompt are deliberately independent of individual V3 suggestions
and semantic aggregation. Media are sent inline without encoding or Files uploads.
"""
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import cv2
from ai_providers import get_provider, create_adapter

from gemini_runtime import (RequestSession, create_client, generate_with_retry,
                            generation_config, get_model, resolve_credential,
                            redact_text, InvalidResponseError)
from label_suggester import _ACTIVITY_SCHEMA, _clean_label, _response_metadata


_CACHE_DIR = Path(__file__).resolve().parent / 'temp' / 'batch_label_cache'
_VERSION = 1
# Keep the base64-encoded video plus prompt below a conservative 100 MB envelope.
_MAX_VIDEO_BYTES = 64 * 1024 * 1024
_MAX_TEXT_BYTES = 1024 * 1024
_VIDEO_SAMPLE_FPS = 3.0
_MIME_TYPES = {'.mp4': 'video/mp4', '.m4v': 'video/mp4', '.avi': 'video/avi',
               '.mov': 'video/mov', '.mpeg': 'video/mpeg', '.mpg': 'video/mpg',
               '.webm': 'video/webm', '.wmv': 'video/wmv', '.3gp': 'video/3gpp',
               '.flv': 'video/x-flv'}
_PROMPT = """Label EVERY supplied FINCH segment of this video in a single response.
Return exactly one result per segment_id, including uncertain segments. IDs refer
to chronological segments, never cluster IDs. Boundaries are inclusive source
frames; start_seconds and end_seconds_exclusive map them onto the video timeline.
For each segment give the specific observable atomic action, not an overall task
or process. Use adjacent video only as context; evidence must describe an action
inside that exact segment. Do not merge segments or move their boundaries.
Labels must be concise verb + object, 2 to 6 words, at most 100 characters.
Distinguish tying a new bow, tightening an existing bow and adjusting its position.
If visibility or sampling is insufficient, use 'Unclear activity', high uncertainty
and an honest explanation; do not invent actions in unseen frames. Return only
the JSON object specified by the response schema, with no markdown or commentary.
"""


def segment_boundaries(segments):
    """Use stable 1-based GUI IDs; repeated cluster IDs remain separate segments."""
    if not segments:
        raise ValueError('Suggest All requires at least one segment')
    records, seen, previous_end = [], set(), -1
    for index, segment in enumerate(segments):
        values = (segment.get('segment_id', index + 1), segment['start_frame'], segment['end_frame'])
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise ValueError('Segment IDs and frame boundaries must be integers')
        identifier, start, end = map(int, values)
        if identifier < 1 or identifier in seen or start <= previous_end or end < start:
            raise ValueError('Segment IDs must be unique and positive; boundaries must be ordered and nonoverlapping')
        records.append(dict(segment_id=identifier, start_frame=start, end_frame=end))
        seen.add(identifier)
        previous_end = end
    return records


def _schema(records):
    item = deepcopy(_ACTIVITY_SCHEMA)
    item['properties']['segment_id'] = {'type': 'integer', 'enum': [r['segment_id'] for r in records]}
    item['required'].append('segment_id')
    return {'type': 'object', 'properties': {'segments': {'type': 'array', 'items': item,
            'minItems': len(records), 'maxItems': len(records)}},
            'required': ['segments'], 'additionalProperties': False}


def _generation_config(model, types, records):
    return generation_config(model, types, dict(temperature=0.1,
        max_output_tokens=min(65536, max(2048, 256 * len(records) + 1024))), _schema(records))


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate JSON field')
        result[name] = value
    return result


def _validate_response(response, records, model):
    """Validate the complete batch atomically; never accept a partial ID map."""
    metadata = _response_metadata(response)
    if (metadata['blocked'] or metadata['truncated'] or
            any(reason not in (None, 'STOP', 'FINISH_REASON_UNSPECIFIED')
                for reason in metadata['finish_reasons'])):
        raise InvalidResponseError('Batch response blocked, truncated or unfinished')
    try:
        text = response.text
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except (AttributeError, ValueError, TypeError):
        raise InvalidResponseError('Batch response must contain valid JSON') from None
    if redact_text(text, None) != text:
        raise InvalidResponseError('Batch response contains credential-like content')
    if not isinstance(payload, dict) or set(payload) != {'segments'} or not isinstance(payload['segments'], list):
        raise InvalidResponseError('Batch response must contain a segments array')
    expected = {record['segment_id'] for record in records}
    if len(payload['segments']) != len(expected):
        raise InvalidResponseError('Batch response must contain one label for every segment ID')
    labels = {}
    for item in payload['segments']:
        if not isinstance(item, dict) or set(item) != set(_ACTIVITY_SCHEMA['required']) | {'segment_id'}:
            raise InvalidResponseError('Batch segment fields do not match the declared schema')
        identifier = item['segment_id']
        if type(identifier) is not int or identifier not in expected or identifier in labels:
            raise InvalidResponseError('Batch response contains a duplicate, unknown or invalid segment ID')
        activity = {key: value for key, value in item.items() if key != 'segment_id'}
        labels[identifier] = _clean_label(SimpleNamespace(text=json.dumps(activity)), model=model)
    if set(labels) != expected:
        raise InvalidResponseError('Batch response is missing segment IDs')
    return labels, payload


def _video_identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(sha256=digest.hexdigest(), size=path.stat().st_size)


def _video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot read video metadata for Suggest All')
        fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not all(math.isfinite(value) and value > 0 for value in (fps, count)):
            raise ValueError('Suggest All requires valid video FPS and frame count')
        return float(fps), int(count)
    finally:
        cap.release()


def suggest_all_activities(video_path, segments, *, cache_dir=None, on_daily_quota=None,
                           progress=None, stats=None, on_upload=None):
    """Return {segment_id: suggested label}; a warm batch cache makes no API call.

    Run on a worker. A cold batch makes one generation request for all segments;
    only shared transient retries or a user-approved quota recovery may replay it.
    Oversized/unsupported media fail without splitting into per-segment requests.
    """
    if get_provider('label') == 'twelvelabs':
        return _suggest_pegasus_batch(video_path, segments, cache_dir, on_daily_quota,
                                      progress, stats, on_upload)
    from google.genai import types
    notify = progress or (lambda message: None)
    report = stats if stats is not None else {}
    report.update(api_calls=0, cache_hit=False)
    records = segment_boundaries(segments)
    path = Path(video_path)
    model = get_model('label')
    config = _generation_config(model, types, records)
    notify('Checking the batch suggestion cache...')
    identity = _video_identity(path)
    cache_input = dict(version=_VERSION, video=identity, boundaries=records,
        model=model, prompt=_PROMPT, generation=config.model_dump(mode='json', exclude_none=True),
        media=dict(mode='full_video_inline', sample_fps=_VIDEO_SAMPLE_FPS,
                   mime_type=_MIME_TYPES.get(path.suffix.lower())))
    key = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
    cached = directory / f'{key}.json'
    try:
        entry = json.loads(cached.read_text(encoding='utf-8'))
        if entry['key'] == key and entry['complete'] is True:
            raw = json.dumps(entry['response'], sort_keys=True)
            if entry['sha256'] == hashlib.sha256(raw.encode()).hexdigest():
                labels, _ = _validate_response(SimpleNamespace(text=raw), records, model)
                report['cache_hit'] = True
                notify('Batch suggestions loaded from cache; review before committing.')
                return labels
    except (OSError, KeyError, ValueError, TypeError, InvalidResponseError):
        pass  # Corrupt/incomplete entries never become partial suggestions.
    mime_type = _MIME_TYPES.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError('Suggest All does not support this video container; use individual suggestions or a supported video')
    if identity['size'] > _MAX_VIDEO_BYTES:
        raise ValueError('Suggest All inline video limit is 64 MiB. Use individual suggestions or a smaller video; no batch request sent')
    fps, frame_count = _video_info(path)
    if records[-1]['end_frame'] >= frame_count:
        raise ValueError('Segment boundary extends beyond the current video')
    timeline = [dict(record, start_seconds=record['start_frame'] / fps,
                     end_seconds_exclusive=(record['end_frame'] + 1) / fps) for record in records]
    manifest = json.dumps(dict(fps=fps, frame_count=frame_count, segments=timeline))
    if len((_PROMPT + manifest).encode()) > _MAX_TEXT_BYTES:
        raise ValueError('Too many segment boundaries for one safe inline request')
    notify(f'Preparing one request for all {len(records)} segments...')
    media = path.read_bytes()
    if len(media) != identity['size'] or hashlib.sha256(media).hexdigest() != identity['sha256']:
        raise ValueError('Video changed while preparing Suggest All; retry explicitly')
    parts = [types.Part(text=_PROMPT), types.Part(text=manifest),
             types.Part(inline_data=types.Blob(data=media, mime_type=mime_type),
                        video_metadata=types.VideoMetadata(fps=_VIDEO_SAMPLE_FPS))]
    credential = resolve_credential()
    notify(f'Generating {len(records)} segment labels in one batch...')
    try:
        with RequestSession(credential, on_daily_quota, factory=create_client) as client:
            response = generate_with_retry(client, dict(model=model,
                contents=types.Content(parts=parts), config=config), progress=notify, stats=report)
    except Exception as error:
        raise RuntimeError('Suggest All failed: ' + redact_text(error, 300)) from None
    labels, payload = _validate_response(response, records, model)
    temporary = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, sort_keys=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                prefix=f'{key}_', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(dict(key=key, complete=True, response=payload,
                           sha256=hashlib.sha256(raw.encode()).hexdigest()), stream)
        os.replace(temporary, cached)
    except OSError:
        notify('Batch suggestions ready; cache could not be saved. Review remains available.')
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return labels


def _suggest_pegasus_batch(video_path, segments, cache_dir, on_daily_quota, progress, stats, on_upload):
    from twelvelabs_provider import video_metadata, timeline, generation_settings
    records = segment_boundaries(segments)
    metadata = video_metadata(video_path)
    manifest = dict(video=metadata, segments=timeline(records, metadata))
    prompt = _PROMPT + '\n' + json.dumps(manifest)
    report = stats if stats is not None else {}
    report.update(api_calls=0, cache_hit=False)
    root = Path(cache_dir) / 'twelvelabs' if cache_dir is not None else None
    adapter = create_adapter('label', cache_dir=root, on_daily_quota=on_daily_quota,
                             on_upload=on_upload, progress=progress, stats=report)
    try:
        return adapter.cached_analysis('batch', video_path, prompt, _schema(records),
            generation_settings(min(65536, max(2048, 256 * len(records) + 1024))),
            lambda response: _validate_response(response, records, adapter.model)[0], inputs=records)
    finally:
        adapter.close()
