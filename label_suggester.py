"""RAVEE Label Suggester V3: context-aware, evidence-based segment suggestions.

The GUI API and manual confirmation stay unchanged. Separate before/target/after
media parts prevent context actions from being mistaken for the target activity.
Default context: 3 seconds on each side, clipped at video boundaries. Temporary
480px MP4s are sampled at up to 6fps; still frames remain available as fallback.
"""

from __future__ import annotations

import math
import json
import logging
import os
import re
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
from ai_providers import get_provider, create_adapter
from gemini_runtime import (resolve_credential, get_model, create_client,
    generation_config, generate_with_retry, classify_error, InvalidResponseError,
    RequestSession, redact_text as _redact_text)

# Recovery changes do not alter media/prompt semantics. Preserve the existing
# aggregation cache encoder identity; bump this when encoding semantics change.
AGGREGATION_ENCODER_SIGNATURE = 'bc25e92f096f26b8af236943c0652ec432e73a524d3ac8cfac7849ffce927d2c'

# Keep each inline request reasonably small. Files API is intentionally not
# required: short clips avoid the prior upload/processing wait.
_MAX_INLINE_BYTES = 18 * 1024 * 1024
_MAX_DURATION_SECONDS = 20.0
_ENCODE_FPS = 6.0
_GEMINI_SAMPLE_FPS = 3.0
_MAX_IMAGE_EDGE = 480
_CONTEXT_SECONDS = 3.0
_LOGGER = logging.getLogger(__name__)

# This is the existing V3 contract, now declared to generateContent rather than
# requested only in prose. Application validation remains authoritative.
_ACTIVITY_SCHEMA = {
    'type': 'object',
    'properties': {
        'label': {'type': 'string', 'description': 'Specific visible target action, 2 to 6 words, at most 100 characters.'},
        'evidence': {'type': 'string', 'description': 'Directly visible TARGET motion/state change; not a list or object.'},
        'uncertainty': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        'target_visible': {'type': 'boolean'},
        'bow_action': {'type': 'string', 'enum': ['tie_new', 'tighten_existing', 'adjust_existing',
                                                'not_applicable', 'unclear']},
    },
    'required': ['label', 'evidence', 'uncertainty', 'target_visible', 'bow_action'],
    'additionalProperties': False,
}


def _generation_config(model, types):
    return generation_config(model, types,
        dict(temperature=0.1, max_output_tokens=512), _ACTIVITY_SCHEMA)


def _safe_value(value, depth=0):
    """Only JSON-like values; never dump response objects, headers or requests."""
    if depth > 6:
        return '[depth limit]'
    if isinstance(value, dict):
        return {_redact_text(key, 100): ('[REDACTED]' if re.search(
            r'(?i)api.?key|private.?key|authorization|token$|secret|password|credential', str(key)) else
            _safe_value(item, depth + 1)) for key, item in list(value.items())[:50]}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth + 1) for item in value[:50]]
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    return f'<{type(value).__name__}>'


def _enum_name(value):
    return getattr(value, 'value', value) if isinstance(value, str) or hasattr(value, 'value') else None


def _response_metadata(response):
    candidates = getattr(response, 'candidates', None) or []
    feedback = getattr(response, 'prompt_feedback', None)
    block_reason = _enum_name(getattr(feedback, 'block_reason', None))
    reasons = [_enum_name(getattr(candidate, 'finish_reason', None)) for candidate in candidates]
    usage = getattr(response, 'usage_metadata', None)
    usage_fields = ('prompt_token_count', 'candidates_token_count', 'thoughts_token_count',
                    'total_token_count', 'cached_content_token_count')
    token_usage = {name: getattr(usage, name, None) for name in usage_fields}
    blocked_reasons = {'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'IMAGE_SAFETY'}
    return {
        'finish_reasons': reasons, 'candidate_count': len(candidates),
        'finish_messages': [getattr(candidate, 'finish_message', None) for candidate in candidates],
        'block_reason': block_reason,
        'blocked': bool(block_reason and block_reason != 'BLOCK_REASON_UNSPECIFIED') or any(
            reason in blocked_reasons for reason in reasons),
        'truncated': True if 'MAX_TOKENS' in reasons else (False if 'STOP' in reasons else None),
        'usage_metadata': token_usage,
        'http_status': getattr(getattr(response, 'sdk_http_response', None), 'status_code', None),
        'model_version': getattr(response, 'model_version', None),
    }


class ActivityResponseError(InvalidResponseError):
    """A rejected response, retaining redacted diagnostics for local inspection."""

    def __init__(self, model, rule, diagnostics):
        self.rule, self.diagnostics = rule, diagnostics
        super().__init__(f'{_redact_text(model, 80)}: {rule}. See label_suggester ERROR diagnostics.')


def _reject_response(model, rule, response, text=None, parsed=None):
    json_valid = False
    if isinstance(text, str) and text.strip():
        try:
            parsed = json.loads(text)
            json_valid = True
        except json.JSONDecodeError:
            pass
    diagnostics = _safe_value(dict(event='invalid_activity_response', model=model, rule=rule,
        response_text=text, parsed_json=parsed, empty=not bool(text and str(text).strip()),
        json_valid=json_valid, response_text_type=type(text).__name__,
        **_response_metadata(response)))
    _LOGGER.error('Gemini activity response rejected: %s', json.dumps(diagnostics, ensure_ascii=True))
    raise ActivityResponseError(model, _redact_text(rule, 300), diagnostics) from None


def _log_api_error(model, error):
    diagnostics = _safe_value(dict(event='activity_api_error', model=model,
        error_type=type(error).__name__, http_code=getattr(error, 'code', None),
        status=getattr(error, 'status', None), message=str(error),
        details=getattr(error, 'details', None)))
    _LOGGER.error('Gemini activity request failed: %s', json.dumps(diagnostics, ensure_ascii=True))
    return classify_error(error).message

_PROMPT = (
    'Label ONLY the TARGET SEGMENT for video-based process mining. '
    'Describe the specific observable action inside this interval, at action level, '
    'not the overall task, process, scene, finished product, or purpose of the full video. '
    'First identify the directly visible hand/tool motion and the object it acts on. '
    'Choose the narrowest verb + object supported by that TARGET motion/state change. '
    'If a specific manipulation is visible, do not replace it with a broad process label. '
    'BEFORE CONTEXT and AFTER CONTEXT show object state and continuity only; '
    'use them only to resolve ambiguity about what is visible in the target, '
    'never assign an action that occurs only outside the target segment. '
    'Longer context or a completed object is not evidence that the entire process '
    'occurs in this segment. Observe visible hand/tool/object motion and state changes inside the target. '
    'Use an open vocabulary: a concise English verb + object label, 2 to 6 words. '
    'There is NO fixed activity list. Use the same plain wording for repeated '
    'observable actions, without inventing differences between continuing segments. '
    'Evidence must describe directly visible TARGET motion/state change and support '
    'the exact action named in the label, not merely the object or scene. '
    'Before/after appearance alone does not prove what occurred behind hands. '
    'If the specific action is occluded or cannot be distinguished from direct target '
    'evidence, do not guess or substitute an umbrella activity. '
    'Use exactly Unclear activity when evidence is insufficient, the target action '
    'is obscured, or no single action predominates. Never infer hidden intent. '
    'Return ONLY a JSON object with the existing fields: label (the short action label), '
    'evidence (directly visible TARGET motion/state change supporting that label), '
    'uncertainty (low, medium, or high), target_visible (boolean: sufficient direct evidence), '
    'and bow_action (tie_new, tighten_existing, adjust_existing, not_applicable, or unclear). '
    'This last field is compatibility metadata, not a candidate activity list: '
    'tie_new requires visible formation of loops/knot within the target; '
    'tighten_existing and adjust_existing require visible work on an already formed bow. '
    'For other activities use not_applicable. If this distinction is occluded use unclear. '
    'With insufficient evidence set uncertainty high, target_visible false, and label Unclear activity.'
)


def _clip_bounds(start_frame, end_frame, fps):
    max_frames = max(1, int(_MAX_DURATION_SECONDS * fps))
    if end_frame - start_frame + 1 > max_frames:
        start_frame += (end_frame - start_frame + 1 - max_frames) // 2
        end_frame = start_frame + max_frames - 1
    return start_frame, end_frame


def _context_windows(video_path, start_frame, end_frame):
    """Keep roles/bounds explicit, including a disclosed long-target excerpt."""
    context = float(os.getenv('RAVEE_SUGGEST_CONTEXT_SECONDS', str(_CONTEXT_SECONDS)))
    if not math.isfinite(context) or not 0 <= context <= 10:
        raise ValueError('RAVEE_SUGGEST_CONTEXT_SECONDS must be between 0 and 10')
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')
        fps = _get_fps(cap)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError('Cannot determine video boundaries for context sampling')
        if end_frame >= total_frames:
            raise ValueError('Segment ends beyond the end of the video')
    finally:
        cap.release()
    padding = round(context * fps)
    windows = []
    if padding and start_frame > 0:
        windows.append(('BEFORE CONTEXT', max(0, start_frame - padding), start_frame - 1))
    target_start, target_end = _clip_bounds(start_frame, end_frame, fps)
    windows.append(('TARGET SEGMENT', target_start, target_end))
    if padding and end_frame < total_frames - 1:
        windows.append(('AFTER CONTEXT', end_frame + 1, min(total_frames - 1, end_frame + padding)))
    return fps, windows


def _target_context_windows(video_path, start_frame, end_frame):
    """Keep before/target/after roles; bound context dominance for short targets.

    Each context side is at most the target's duration, within the configured
    maximum. Aggregation still uses _context_windows for wider task continuity.
    """
    fps, windows = _context_windows(video_path, start_frame, end_frame)
    target_frames = end_frame - start_frame + 1
    bounded = []
    for role, start, end in windows:
        if role == 'BEFORE CONTEXT':
            start = max(start, start_frame - target_frames)
        elif role == 'AFTER CONTEXT':
            end = min(end, end_frame + target_frames)
        bounded.append((role, start, end))
    return fps, bounded


def _window_description(role, start_frame, end_frame, fps):
    return (f'{role}: source frames {start_frame}-{end_frame} inclusive, '
            f'source time [{start_frame / fps:.3f}, {(end_frame + 1) / fps:.3f}) seconds. '
            'The following media starts at local time 0. '
            + ('Label this interval only.' if role == 'TARGET SEGMENT' else 'Context only; do not label this interval.'))


def _get_fps(cap) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        return 25.0
    return fps


def _resize_even(frame: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    new_width = max(2, int(round(width * scale)) // 2 * 2)
    new_height = max(2, int(round(height * scale)) // 2 * 2)
    if (new_width, new_height) == (width, height):
        return frame
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _make_clip(video_path: str | Path, start_frame: int, end_frame: int) -> bytes:
    """Make an MP4 from consecutive sampled frames in the requested interval.

    A very long segment is cropped to a contiguous 20-second middle excerpt;
    no time-lapse spanning the whole segment is used.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_path = None
    writer = None
    try:
        fps = _get_fps(cap)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and start_frame >= total_frames:
            raise ValueError("Segment starts beyond the end of the video")
        if total_frames > 0 and end_frame >= total_frames:
            raise ValueError("Segment ends beyond the end of the video")

        # All included frames remain chronologically consecutive (apart from
        # uniform downsampling to reduce the size of the MP4).
        cropped_start, cropped_end = _clip_bounds(start_frame, end_frame, fps)
        if (cropped_start, cropped_end) != (start_frame, end_frame):
            print(f"Label Suggester: long segment, analyzing middle frames {cropped_start}-{cropped_end}.")

        stride = max(1, math.ceil(fps / _ENCODE_FPS))
        output_fps = fps / stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, cropped_start)
        temp = tempfile.NamedTemporaryFile(prefix="ravee_label_", suffix=".mp4", delete=False)
        output_path = Path(temp.name)
        temp.close()

        written = 0
        for offset in range(cropped_end - cropped_start + 1):
            if offset % stride == 0:
                success, frame = cap.read()
                if not success:
                    raise RuntimeError('Video decoding stopped before the requested clip ended')
                frame = _resize_even(frame, _MAX_IMAGE_EDGE)
                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV MP4 encoder (mp4v) is unavailable")
                writer.write(frame)
                written += 1
            else:
                if not cap.grab():
                    raise RuntimeError('Video decoding stopped before the requested clip ended')

        if writer is not None:
            writer.release()
            writer = None
        if written < 2:
            raise RuntimeError("Too few frames to encode a usable video clip")
        clip_bytes = output_path.read_bytes()
        if not clip_bytes:
            raise RuntimeError("OpenCV wrote an empty video clip")
        if len(clip_bytes) > _MAX_INLINE_BYTES:
            raise RuntimeError("MP4 clip is too large for a safe inline request")
        print(f"Label Suggester: video clip, {written} frames at {output_fps:.2f} FPS, {len(clip_bytes)/1048576:.2f} MiB.")
        return clip_bytes
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass


def _sample_frames(video_path: str | Path, start_frame: int, end_frame: int, sample_count: int):
    """Old frame-sampling approach, retained for fallback and A/B comparison."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = _get_fps(cap)
        count = min(sample_count, end_frame - start_frame + 1)
        indices = sorted(set(np.linspace(start_frame, end_frame, count, dtype=int)))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _resize_even(frame, 640)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                frames.append((int(idx), idx / fps, jpeg.tobytes()))
        if not frames:
            raise RuntimeError("Could not read any still frames from the segment")
        return frames
    finally:
        cap.release()


def _clean_label(response, *, model=None) -> str:
    model = model or get_model('label')
    try:
        text = response.text
    except (ValueError, TypeError, AttributeError) as error:
        _reject_response(model, f'response.text unavailable ({type(error).__name__})', response)
    result, parse_error = None, None
    if isinstance(text, str) and text.strip():
        try:
            result = json.loads(text)
        except json.JSONDecodeError as error:
            parse_error = f'JSON syntax: {error.msg} at line {error.lineno}, column {error.colno}'
    metadata = _response_metadata(response)
    if metadata['blocked']:
        reason = metadata['block_reason'] or ', '.join(filter(None, metadata['finish_reasons']))
        _reject_response(model, f'blocked response ({reason}); no label accepted', response, text, result)
    if metadata['truncated']:
        _reject_response(model, 'MAX_TOKENS: truncated response; review output/thinking limits', response, text, result)
    failures = [reason for reason in metadata['finish_reasons']
                if reason not in (None, 'STOP', 'FINISH_REASON_UNSPECIFIED')]
    if failures:
        _reject_response(model, f'non-success finish_reason: {failures[0]}', response, text, result)
    if not isinstance(text, str) or not text.strip():
        _reject_response(model, 'empty or non-text response; no JSON activity received', response, text)
    if parse_error:
        _reject_response(model, parse_error, response, text)
    if not isinstance(result, dict):
        _reject_response(model, 'JSON root must be an object', response, text, result)
    missing = [field for field in _ACTIVITY_SCHEMA['required'] if field not in result]
    if missing:
        _reject_response(model, 'missing fields: ' + ', '.join(missing), response, text, result)
    for field in ('label', 'evidence'):
        if not isinstance(result[field], str):
            _reject_response(model, f'{field} must be a string (got {type(result[field]).__name__})', response, text, result)
    if type(result['target_visible']) is not bool:
        _reject_response(model, 'target_visible must be a JSON boolean', response, text, result)
    for field in ('uncertainty', 'bow_action'):
        if result[field] not in _ACTIVITY_SCHEMA['properties'][field]['enum']:
            _reject_response(model, f'{field} must be one of the declared enum values', response, text, result)
    label = ' '.join(result['label'].split()).strip(' .')
    if _redact_text(label) != label:
        _reject_response(model, 'label contains credential-like content; no label accepted', response, text, result)
    if not 2 <= len(label.split()) <= 6 or len(label) > 100 or not any(char.isalpha() for char in label):
        _reject_response(model, 'label must be 2-to-6 words, at most 100 characters, with alphabetic text', response, text, result)
    if (not result['target_visible'] or result['uncertainty'] == 'high'
            or not result['evidence'].strip() or result['bow_action'] == 'unclear'
            or label.lower() == 'unclear activity'):
        return 'Unclear activity'
    canonical = {'tie_new': 'Tie ribbon bow', 'tighten_existing': 'Tighten ribbon bow',
                 'adjust_existing': 'Adjust ribbon bow'}
    if result['bow_action'] in canonical:
        return canonical[result['bow_action']]
    return label[0].upper() + label[1:].lower()


def suggest_activity(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    sample_count: int = 6,
    *, on_daily_quota=None, on_upload=None, cache_dir=None, progress=None,
) -> str:
    """Return an unverified suggested activity label for [start_frame,end_frame].

    The GUI calls this in a worker thread; it should NOT be called directly
    from Tkinter's event loop. 'Use suggestion' and 'Commit & Next' remain
    manual. This function does not change GUI state or LABEL.
    """
    if get_provider('label') == 'twelvelabs':
        return _suggest_pegasus(video_path, start_frame, end_frame, sample_count,
                               on_daily_quota, on_upload, cache_dir, progress)
    credential = resolve_credential()

    start_frame, end_frame = int(start_frame), int(end_frame)
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    # Imported on demand: RAVEE stays usable for manual labeling if Gemini
    # dependencies are missing or the API is unavailable.
    from google import genai
    from google.genai import types

    mode = os.getenv("RAVEE_SUGGEST_MODE", "video").strip().lower()
    if mode not in ("video", "frames"):
        raise ValueError("RAVEE_SUGGEST_MODE must be 'video' or 'frames'")

    fps, windows = _target_context_windows(video_path, start_frame, end_frame)
    prompt = (_PROMPT + f' Original target: source frames {start_frame}-{end_frame} inclusive. '
              'Media windows are supplied chronologically with separate roles.')
    target_bounds = next((start, end) for role, start, end in windows if role == 'TARGET SEGMENT')
    if target_bounds != (start_frame, end_frame):
        prompt += (' Only a middle excerpt of the long target is supplied. Do not infer '
                   'actions in unseen target portions; return Unclear activity if this '
                   'is insufficient to identify the segment activity.')
    parts = [types.Part(text=prompt)]
    total_bytes = 0
    for role, start, end in windows:
        parts.append(types.Part(text=_window_description(role, start, end, fps)))
        video_data = None
        if mode == 'video':
            try:
                video_data = _make_clip(video_path, start, end)
                if total_bytes + len(video_data) > _MAX_INLINE_BYTES:
                    raise RuntimeError('Combined context and target clips exceed inline size limit')
            except (OSError, RuntimeError) as exc:
                video_data = None
                warnings.warn(f'{role} clip encoding failed; using still-frame fallback: {exc}')
        if video_data is not None:
            total_bytes += len(video_data)
            parts.append(types.Part(
                inline_data=types.Blob(data=video_data, mime_type='video/mp4'),
                video_metadata=types.VideoMetadata(fps=_GEMINI_SAMPLE_FPS),
            ))
        else:
            count = sample_count if role == 'TARGET SEGMENT' else min(sample_count, 3)
            for index, seconds, jpeg in _sample_frames(video_path, start, end, count):
                total_bytes += len(jpeg)
                if total_bytes > _MAX_INLINE_BYTES:
                    raise RuntimeError('Combined media exceed inline size limit')
                parts.append(types.Part(text=f'{role} frame {index} at source time {seconds:.3f}s'))
                parts.append(types.Part.from_bytes(data=jpeg, mime_type='image/jpeg'))
    # End the chronological context with an explicit return to the exact target.
    # This does not replay media or create another request.
    parts.append(types.Part(text=(
        f'Now label ONLY TARGET SEGMENT source frames {start_frame}-{end_frame}; '
        f'actual supplied target coverage is {target_bounds[0]}-{target_bounds[1]}. '
        'Use context only to resolve ambiguity. Return the specific visible action, '
        'not the overall task or process; evidence must support that exact action.')))
    contents = types.Content(parts=parts)

    model = get_model('label')
    try:
        with RequestSession(credential, on_daily_quota, factory=create_client) as client:
            response = generate_with_retry(client, dict(model=model, contents=contents,
                config=_generation_config(model, types)))
    except Exception as error:
        safe_error = _log_api_error(model, error)
        # SDK exception strings can echo request credentials. The redacted
        # diagnostics above retain details; do not print an unsafe cause chain.
        raise RuntimeError(f'{_redact_text(model, 80)}: API request failed: {safe_error}') from None
    return _clean_label(response, model=model)


def _suggest_pegasus(video_path, start_frame, end_frame, sample_count,
                     on_daily_quota, on_upload, cache_dir, progress):
    from batch_label_suggester import segment_boundaries
    from twelvelabs_provider import video_metadata, timeline, generation_settings
    records = segment_boundaries([dict(start_frame=int(start_frame), end_frame=int(end_frame))])
    if sample_count < 1:
        raise ValueError('sample_count must be at least 1')
    metadata = video_metadata(video_path)
    target = timeline(records, metadata)[0]
    context = float(os.getenv('RAVEE_SUGGEST_CONTEXT_SECONDS', str(_CONTEXT_SECONDS)))
    if not math.isfinite(context) or context < 0:
        raise ValueError('Suggestion context seconds must be finite and nonnegative')
    # The API window must span >=1s. Padding is context only; target is unchanged.
    start = max(0., target['start_seconds'] - context)
    end = min(metadata['duration'], target['end_seconds_exclusive'] + context)
    end = min(metadata['duration'], max(end, start + 1.))
    start = max(0., min(start, end - 1.))
    prompt = _PROMPT + '\n' + json.dumps(dict(target_segment=target,
        analysis_window_seconds=[start, end], video=metadata,
        context_usage='Frames before/after the exact target are context only. Label ONLY target_segment.'))
    root = Path(cache_dir) / 'twelvelabs' if cache_dir is not None else None
    adapter = create_adapter('label', cache_dir=root, on_daily_quota=on_daily_quota,
                             on_upload=on_upload, progress=progress)
    try:
        origin = metadata.get('start_time')
        window = (start + origin, end + origin) if origin is not None else None
        if window is None and progress is not None:
            progress('Video time origin unverified; analyzing full asset with exact target frame timestamps.')
        return adapter.cached_analysis('individual', video_path, prompt, _ACTIVITY_SCHEMA,
            generation_settings(1024), lambda response: _clean_label(response, model=adapter.model),
            window=window, inputs=target)
    finally:
        adapter.close()
