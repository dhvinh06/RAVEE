"""Optional official SDK adapter for Pegasus Analyze, not video segmentation.

No SDK import, upload or request occurs at module import. Results and remote
asset references are separate from Gemini caches. Call only from AI workers.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Lock
import time
from types import SimpleNamespace

import cv2

from gemini_runtime import (configuration, Credential, ErrorInfo, InvalidResponseError,
                            redact_text, remember_secret, retry_after, status_code)

_CACHE_DIR = Path(__file__).resolve().parent / 'temp' / 'twelvelabs'
_DIRECT_UPLOAD_BYTES = 200_000_000  # Conservative decimal interpretation of 200 MB.
_AUTO_UPLOAD_BYTES = 64 * 1024 * 1024
_ASSET_LOCK = Lock()
_REQUEST_OPTIONS = {'timeout_in_seconds': 120, 'max_retries': 0}


class TwelveLabsError(RuntimeError):
    provider = 'twelvelabs'

    def __init__(self, info):
        self.info, self.kind, self.status_code = info, info.kind, info.code
        super().__init__(info.message)


def error_info(error):
    if isinstance(error, TwelveLabsError):
        return error.info
    code = status_code(error)
    body = getattr(error, 'body', None)
    identifier = body.get('code', '') if isinstance(body, dict) else ''
    if identifier in ('usage_limit_exceeded', 'workload_allowance_exhausted', 'insufficient_credits'):
        return ErrorInfo('provider_quota', False,
                         'TwelveLabs plan allowance/credits exhausted. Continue manually or review your plan; no provider or credential switched.', code)
    if code in (401, 403):
        return ErrorInfo('authentication', False, 'TwelveLabs authentication/permission failure. Check TWELVELABS_API_KEY.', code)
    if code == 429:
        known_rate = identifier in ('rate_limit_exceeded', 'too_many_requests', 'rate_limit') or retry_after(error) > 0
        return ErrorInfo('rate_limit' if known_rate else 'quota_unknown', known_rate,
                         'TwelveLabs temporary rate limit.' if known_rate else
                         'TwelveLabs HTTP 429: quota type unknown; retry explicitly after checking account limits.', code)
    if code in (500, 502, 503, 504):
        return ErrorInfo('service_unavailable', True, f'TwelveLabs HTTP {code}: temporary service failure.', code)
    if isinstance(error, TimeoutError) or 'timeout' in type(error).__name__.lower():
        return ErrorInfo('timeout', False, 'TwelveLabs request timed out; retry explicitly. Uploaded asset references are retained.', code)
    return ErrorInfo('request_error', False, 'TwelveLabs request failed: ' + redact_text(error, 240), code)


def create_client(credential):
    try:
        from twelvelabs import TwelveLabs
    except ImportError:
        raise RuntimeError('Optional TwelveLabs SDK missing. Install requirements-twelvelabs.txt to use Pegasus; Gemini/manual review remain available.') from None
    return TwelveLabs(api_key=credential.key, timeout=120)


def video_identity(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Video changed while computing TwelveLabs input identity; retry explicitly')
    return {'sha256': digest.hexdigest(), 'size': after.st_size}


def video_metadata(path):
    """Header-only inspection; never decode or reprocess a video."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot read video metadata for Pegasus')
        fps, count, width, height = (cap.get(prop) for prop in (
            cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if not all(math.isfinite(value) and value > 0 for value in (fps, count, width, height)):
        raise ValueError('Pegasus requires valid FPS, frame count and resolution')
    duration = count / fps
    if not 1 <= duration <= 3600:
        raise ValueError(f'Pegasus synchronous Analyze requires a 1-second to 1-hour video; actual={duration:g}s')
    if min(width, height) < 360 or max(width, height) > 5184 or min(width, height) > 2160 or max(width, height) / min(width, height) > 2.4:
        raise ValueError(f'Video resolution/aspect ratio is outside Pegasus Analyze limits: {width:g}x{height:g}')
    return {'fps': float(fps), 'frame_count': int(count), 'duration': duration,
            'width': int(width), 'height': int(height), 'start_time': _container_start_time(path)}


def _container_start_time(path):
    # Analyze windows use absolute container timestamps, not playback-relative
    # timestamps. Probe metadata only; an unverified origin uses full-video input.
    probe = shutil.which('ffprobe')
    if probe is not None:
        try:
            result = subprocess.run([probe, '-v', 'error', '-show_entries', 'format=start_time',
                                     '-of', 'json', str(path)], capture_output=True, text=True,
                                    check=True, timeout=10)
            origin = float(json.loads(result.stdout)['format']['start_time'])
            if math.isfinite(origin):
                return origin
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            pass
    return None


def timeline(records, metadata):
    if records[-1]['end_frame'] >= metadata['frame_count']:
        raise ValueError('FINCH segment boundary extends beyond the Pegasus input video')
    return [dict(row, start_seconds=row['start_frame'] / metadata['fps'],
                 end_seconds_exclusive=(row['end_frame'] + 1) / metadata['fps']) for row in records]


def generation_settings(max_tokens=8192):
    return {'temperature': 0.1, 'max_tokens': max_tokens}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResponseError('Duplicate JSON property in Pegasus response')
        result[key] = value
    return result


def _validate_json(value, schema):
    """Validate the schema subset used by RAVEE, including strict JSON types."""
    kind = schema.get('type')
    matches = {'object': isinstance(value, dict), 'array': isinstance(value, list),
               'string': isinstance(value, str), 'boolean': type(value) is bool,
               'integer': type(value) is int, 'number': type(value) in (int, float)}
    if kind not in matches or not matches[kind]:
        raise InvalidResponseError('Pegasus JSON field has an invalid type')
    if kind == 'number' and not math.isfinite(value):
        raise InvalidResponseError('Nonfinite Pegasus JSON number')
    if 'enum' in schema and value not in schema['enum']:
        raise InvalidResponseError('Pegasus JSON field is outside its declared enum')
    if kind == 'object':
        properties = schema.get('properties', {})
        if not set(schema.get('required', ())).issubset(value):
            raise InvalidResponseError('Pegasus JSON is missing required fields')
        if schema.get('additionalProperties') is False and not set(value).issubset(properties):
            raise InvalidResponseError('Unexpected Pegasus JSON fields')
        for key in value.keys() & properties.keys():
            _validate_json(value[key], properties[key])
    if kind == 'array':
        if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', math.inf):
            raise InvalidResponseError('Pegasus JSON array has incomplete or extra results')
        for item in value:
            _validate_json(item, schema['items'])


def _remote_schema(schema):
    # Pegasus accepts minItems only 0/1 and not maxItems. Application validation
    # still enforces the complete original schema, including exact batch count.
    result = deepcopy(schema)
    result.pop('maxItems', None)
    if result.get('minItems', 0) > 1:
        result['minItems'] = 1
    if 'properties' in result:
        result['properties'] = {key: _remote_schema(value) for key, value in result['properties'].items()}
    if 'items' in result:
        result['items'] = _remote_schema(result['items'])
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write(path, value, notify):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pegasus_', dir=path.parent) as directory:
            pending = Path(directory) / 'record.json'
            pending.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
            os.replace(pending, path)
    except OSError:
        notify('TwelveLabs cache could not be saved; current suggestions remain reviewable.')


class TwelveLabsAdapter:
    provider = 'twelvelabs'

    def __init__(self, *, model='pegasus1.6', cache_dir=None, on_daily_quota=None,
                 on_upload=None, progress=None, stats=None):
        key = configuration().get('TWELVELABS_API_KEY')
        if not key:
            raise RuntimeError('Missing TWELVELABS_API_KEY; no provider fallback performed')
        if model not in ('pegasus1.6', 'pegasus1.5'):
            raise ValueError('TWELVELABS_MODEL must name a supported Pegasus Analyze model')
        self.model, self.credential = model, Credential('TWELVELABS_API_KEY', key)
        remember_secret(key)
        # Conservative credential scope + authenticated retrieve prevents a
        # reference crossing accounts. A key change may require a fresh upload.
        self.scope = hashlib.sha256(('ravee-twelvelabs-assets-v1:' + key).encode()).hexdigest()
        self.root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR
        self.on_quota, self.on_upload = on_daily_quota, on_upload
        self.notify = progress or (lambda message: None)
        self.stats = stats if stats is not None else {}
        self.client = None

    def _client(self):
        if self.client is None:
            self.client = create_client(self.credential)
        return self.client

    def close(self):
        client, self.client = self.client, None
        if client is not None:
            try:
                exit_method = getattr(type(client), '__exit__', None)
                if exit_method is not None:
                    exit_method(client, None, None, None)
                elif callable(getattr(client, 'close', None)):
                    client.close()
            except Exception:
                self.notify('TwelveLabs client cleanup failed; completed suggestions retained.')

    def _call(self, operation, *, analysis=False, retry=True):
        for attempt in range(3 if retry else 1):
            try:
                if analysis:
                    self.stats['api_calls'] = self.stats.get('api_calls', 0) + 1
                return operation()
            except Exception as error:
                info = error_info(error)
                if info.kind == 'provider_quota' and self.on_quota is not None:
                    self.on_quota(self.credential)  # Manual/cancel only, no key rotation.
                delay = max(2 ** attempt, retry_after(error))
                if not retry or not info.retryable or attempt == 2 or delay > 30:
                    raise TwelveLabsError(info) from None
                self.notify(info.message + f' Retry {attempt + 1}/2 in {delay:g}s.')
                time.sleep(delay)

    def _wait_ready(self, asset, deadline):
        while getattr(asset, 'status', None) != 'ready':
            if getattr(asset, 'status', None) == 'failed':
                raise TwelveLabsError(ErrorInfo('asset_failed', False, 'TwelveLabs could not process the video asset. No analysis was sent.'))
            if getattr(asset, 'status', None) != 'processing':
                raise TwelveLabsError(ErrorInfo('asset_failed', False, 'Unknown TwelveLabs asset readiness state; no analysis was sent.'))
            if time.monotonic() >= deadline:
                raise TwelveLabsError(ErrorInfo('timeout', False, 'TwelveLabs asset readiness timed out; retry explicitly. Existing asset-cache records are preserved.'))
            self.notify('Waiting for TwelveLabs video asset readiness...')
            time.sleep(min(2, max(0, deadline - time.monotonic())))
            asset = self._call(lambda: self._client().assets.retrieve(asset.id, request_options=_REQUEST_OPTIONS))
        return asset.id

    def asset(self, path, identity):
        with _ASSET_LOCK:
            key = _digest({'scope': self.scope, 'video': identity})
            record_path = self.root / 'assets' / (key + '.json')
            try:
                saved = json.loads(record_path.read_text(encoding='utf-8'))
                payload = {name: saved[name] for name in ('scope', 'video', 'asset_id')}
                if (saved['sha256'] != _digest(payload) or saved['scope'] != self.scope
                        or saved['video'] != identity or not isinstance(saved['asset_id'], str)):
                    raise ValueError('Incompatible asset record')
            except (OSError, ValueError, KeyError, TypeError):
                saved = None
            if saved is not None:
                try:
                    # This authenticates access using the currently selected key.
                    asset = self._call(lambda: self._client().assets.retrieve(saved['asset_id'], request_options=_REQUEST_OPTIONS))
                except TwelveLabsError as error:
                    if error.status_code != 404:
                        raise
                else:
                    if asset.id != saved['asset_id']:
                        raise ValueError('TwelveLabs returned a different video asset identity')
                    metadata = getattr(asset, 'user_metadata', None)
                    if not isinstance(metadata, dict) or metadata.get('ravee_sha256') != identity['sha256']:
                        raise ValueError('TwelveLabs asset metadata does not match the current video; no asset adopted')
                    self.notify('Reusing verified TwelveLabs video asset.')
                    return self._wait_ready(asset, time.monotonic() + 600)
            if identity['size'] <= 0 or identity['size'] > _DIRECT_UPLOAD_BYTES:
                raise ValueError('TwelveLabs local direct upload limit is 200 MB; no upload sent. Multipart upload is not enabled.')
            if identity['size'] > _AUTO_UPLOAD_BYTES and not (
                    self.on_upload is not None and self.on_upload(str(path), identity['size'])):
                raise RuntimeError('TwelveLabs upload above 64 MiB requires explicit approval; no upload sent')
            if video_identity(path) != identity:
                raise ValueError('Video changed before TwelveLabs upload; retry explicitly')
            self.notify('Uploading video to TwelveLabs; FINCH boundaries remain unchanged.')
            with Path(path).open('rb') as stream:
                # Never retry an ambiguous upload automatically (duplicate risk).
                asset = self._call(lambda: self._client().assets.create(method='direct', file=stream,
                    filename=Path(path).name, user_metadata=json.dumps({'ravee_sha256': identity['sha256']}),
                    request_options=_REQUEST_OPTIONS), retry=False)
            if not isinstance(getattr(asset, 'id', None), str) or not asset.id:
                raise InvalidResponseError('TwelveLabs upload returned no usable asset ID')
            # Persist before polling, so timeout/interruption can resume this asset.
            saved = {'scope': self.scope, 'video': identity, 'asset_id': asset.id}
            _write(record_path, dict(saved, sha256=_digest(saved)), self.notify)
            if video_identity(path) != identity:
                raise ValueError('Video changed during TwelveLabs upload; no analysis sent')
            return self._wait_ready(asset, time.monotonic() + 600)

    def analyze(self, video_path, prompt, schema, generation, *, window=None, identity=None):
        identity = identity or video_identity(video_path)
        if len(prompt.encode('utf-8')) > 1024 * 1024:
            raise ValueError('Pegasus prompt exceeds the RAVEE 1 MiB safety budget')
        asset_id = self.asset(video_path, identity)
        from twelvelabs.types import VideoContext_AssetId, SyncResponseFormat
        kwargs = dict(model_name=self.model, video=VideoContext_AssetId(asset_id=asset_id),
                      prompt=prompt, response_format=SyncResponseFormat(type='json_schema', json_schema=_remote_schema(schema)),
                      **generation, request_options=_REQUEST_OPTIONS)
        if window is not None:
            kwargs.update(start_time=window[0], end_time=window[1])
        self.notify(f'Analyzing video with TwelveLabs {self.model}...')
        response = self._call(lambda: self._client().analyze(**kwargs), analysis=True)
        if getattr(response, 'finish_reason', None) != 'stop' or getattr(response, 'error', None):
            raise InvalidResponseError('Pegasus response incomplete, truncated or failed; no partial result accepted')
        try:
            text = response.data
            value = json.loads(text, object_pairs_hook=_unique_object,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            if redact_text(text, None) != text:
                raise ValueError('Credential-like content')
            _validate_json(value, schema)
        except (AttributeError, ValueError, TypeError):
            raise InvalidResponseError('Pegasus response failed structured JSON validation') from None
        return SimpleNamespace(text=json.dumps(value))

    def cached_analysis(self, stage, video_path, prompt, schema, generation, validate, *, window=None, inputs=None):
        identity = video_identity(video_path)
        key = _digest({'version': 1, 'provider': self.provider, 'model': self.model,
                       'video': identity, 'prompt': prompt, 'schema': schema, 'generation': generation,
                       'window': window, 'inputs': inputs, 'media': 'full-video-asset-v1'})
        path = self.root / stage / (key + '.json')
        self.stats.update(cache_hit=False)
        try:
            entry = json.loads(path.read_text(encoding='utf-8'))
            if entry['complete'] is True and entry['key'] == key and entry['sha256'] == _digest(entry['response']):
                _validate_json(entry['response'], schema)
                result = validate(SimpleNamespace(text=json.dumps(entry['response'])))
                self.stats['cache_hit'] = True
                self.notify('TwelveLabs suggestions loaded from cache; review before committing.')
                return result
        except (OSError, ValueError, KeyError, TypeError, InvalidResponseError):
            pass
        response = self.analyze(video_path, prompt, schema, generation, window=window, identity=identity)
        result = validate(response)  # Domain validation BEFORE publishing a result.
        payload = json.loads(response.text)
        _write(path, {'complete': True, 'key': key, 'response': payload, 'sha256': _digest(payload)}, self.notify)
        return result
