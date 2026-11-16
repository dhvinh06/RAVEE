"""Shared Gemini credentials, models, client construction and bounded API retries.

No provider calls on import. Credential changes never enter local AI cache keys.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import os
from pathlib import Path
import random
import re
import time

from dotenv import dotenv_values, load_dotenv


_ENV_PATH = Path(__file__).resolve().parent / '.env'
_PREEXISTING_ENV = set(os.environ)
load_dotenv(_ENV_PATH)
_INJECTED = {key: value for key, value in os.environ.items() if key not in _PREEXISTING_ENV}
_SEEN_SECRETS = set()


def configuration():
    """Reread .env per request; explicit process environment takes precedence."""
    values = dict(dotenv_values(_ENV_PATH)) if _ENV_PATH.is_file() else {}
    values.update({key: value for key, value in os.environ.items()
                   if key not in _INJECTED or value != _INJECTED[key]})
    return values


@dataclass(frozen=True)
class Credential:
    name: str
    key: str = field(repr=False)


def resolve_credential():
    config = configuration()
    active = (config.get('GEMINI_ACTIVE_KEY') or '').strip()
    if active:
        if not re.fullmatch(r'[A-Za-z0-9_]+', active):
            raise RuntimeError('GEMINI_ACTIVE_KEY must be a numeric or named credential selector')
        name = 'GEMINI_API_KEY' if active == '1' else 'GEMINI_API_KEY_' + active.upper()
        value = config.get(name)
        if not value:
            raise RuntimeError('The selected Gemini credential is not configured; check GEMINI_ACTIVE_KEY')
    else:
        # Preserve the original Google-key precedence when selection is absent.
        name = 'GOOGLE_API_KEY' if config.get('GOOGLE_API_KEY') else 'GEMINI_API_KEY'
        value = config.get(name)
        if not value:
            raise RuntimeError('Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable')
    _SEEN_SECRETS.add(value)
    return Credential(name, value)


def get_model(stage):
    config = configuration()
    variable = {'label': 'GEMINI_LABEL_MODEL', 'aggregation': 'GEMINI_AGGREGATION_MODEL'}[stage]
    return config.get(variable) or config.get('GEMINI_MODEL') or 'gemini-3.5-flash-lite'


def create_client(credential=None, *, timeout=60000):
    from google import genai
    from google.genai import types
    credential = credential or resolve_credential()
    return genai.Client(api_key=credential.key, http_options=types.HttpOptions(
        timeout=timeout, retry_options=types.HttpRetryOptions(attempts=1)))


def generation_config(model, types, base, schema=None):
    options = dict(base)
    if schema is not None:
        options.update(response_mime_type='application/json', response_json_schema=schema)
    identifier = model.removeprefix('models/')
    if identifier == 'gemini-4-hydrogen' or identifier.startswith('gemini-4-hydrogen-'):
        for name in ('temperature', 'top_p', 'top_k'):
            options.pop(name, None)
        options.update(max_output_tokens=65536, thinking_config=types.ThinkingConfig(thinking_level='LOW'))
    return types.GenerateContentConfig(**options)


def redact_text(value, limit=8192):
    text = str(value)
    # Include all configured profiles, not just the active one, and credentials
    # used earlier in this session if the configuration has since changed.
    config = configuration()
    secrets = set(_SEEN_SECRETS)
    for key, secret in list(config.items()) + list(os.environ.items()):
        if secret and (key in ('GOOGLE_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS')
                       or re.fullmatch(r'GEMINI_API_KEY(?:_[A-Za-z0-9_]+)?', key)):
            secrets.add(secret)
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----',
                  '[REDACTED PRIVATE KEY]', text, flags=re.DOTALL)
    text = re.sub(r'\bAIza[\w-]+', '[REDACTED API KEY]', text)
    text = re.sub(r'\beyJ[\w-]+\.[\w-]+\.[\w-]+', '[REDACTED TOKEN]', text)
    text = re.sub(r'(?i)\bBearer\s+[^\s"\',;]+', 'Bearer [REDACTED]', text)
    credential = (r'[\w.-]{0,100}(?:api[_-]?key|authorization|(?:access[_-]?|refresh[_-]?)?token|'
                  r'client[_-]?secret|private[_-]?key|password|credentials?)[\w.-]{0,100}')
    text = re.sub(r'(?i)(["\']' + credential + r'["\']\s*:\s*)(["\'])(?:\\.|(?!\2).)*?\2',
                  r'\1"[REDACTED]"', text, flags=re.DOTALL)
    text = re.sub(r'(?i)(?<![\w.-])(' + credential + r'\s*[=:]\s*)[^\s,&}\]"\']+', r'\1[REDACTED]', text)
    text = re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@', r'\1[REDACTED]@', text)
    if limit is None:
        return text
    return text[:limit] + ('... [diagnostic clipped]' if len(text) > limit else '')


def status_code(error):
    value = getattr(error, 'code', None) or getattr(error, 'status_code', None)
    value = value or getattr(getattr(error, 'response', None), 'status_code', None)
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def retry_after(error):
    headers = getattr(getattr(error, 'response', None), 'headers', {}) or {}
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    if value is None:
        return 0.
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0.
    return max(0., delay) if math.isfinite(delay) else 0.


@dataclass(frozen=True)
class ErrorInfo:
    kind: str
    retryable: bool
    message: str
    code: int | None = None


class InvalidResponseError(RuntimeError):
    kind = 'invalid_response'


def classify_error(error):
    code = status_code(error)
    if getattr(error, 'kind', None) == 'invalid_response' or type(error).__name__ == 'ActivityResponseError':
        return ErrorInfo('invalid_response', False, 'Invalid model response: schema or evidence validation failed.', code)
    if code in (401, 403):
        return ErrorInfo('authentication', False, 'Authentication or permission failure; check the selected project credential.', code)
    if code == 429:
        # SDK QuotaFailure details contain quotaId/metric/dimensions. Daily wins
        # over RetryInfo, which may also be present on an exhausted daily quota.
        detail = str(getattr(error, 'details', '')) + str(getattr(error, 'response_json', '')) + str(error)
        normalized = re.sub(r'[^a-z0-9]', '', detail.lower())
        if any(token in normalized for token in ('perday', 'daily', 'rpd', 'quotaexceeded')):
            return ErrorInfo('daily_quota', False, 'Daily quota exhausted (RPD). Wait for reset or manually select an authorized project; no key rotation performed.', code)
        if any(token in normalized for token in ('perminute', 'persecond', 'rpm', 'tpm', 'ratelimitexceeded', 'toomanyrequests')) or retry_after(error) > 0:
            return ErrorInfo('rate_limit', True, 'Temporary request/token rate limit (RPM/TPM).', code)
        return ErrorInfo('quota_unknown', False, 'HTTP 429: quota type unknown. Check project quota details before retrying.', code)
    if code in (500, 502, 503, 504):
        return ErrorInfo('service_unavailable', True, f'HTTP {code}: temporary Gemini service unavailability.', code)
    return ErrorInfo('request_error', False, 'Gemini request failed: ' + redact_text(error, 250), code)


def generate_with_retry(client, kwargs, progress=None, stats=None):
    return call_with_retry(lambda: client.models.generate_content(**kwargs), progress, stats)


def call_with_retry(operation, progress=None, stats=None, *, attempts=4):
    waited = 0.
    for attempt in range(max(1, min(attempts, 4))):
        if stats is not None:
            stats['api_calls'] += 1
        try:
            return operation()
        except Exception as error:
            info = classify_error(error)
            if not info.retryable or attempt == max(1, min(attempts, 4)) - 1:
                raise
            delay = max(retry_after(error), random.uniform(0., min(16., 2. ** attempt)))
            if waited + delay > 60:
                raise
            if progress:
                progress(f'{info.message} Retry {attempt + 1}/3 in {delay:.1f}s')
            time.sleep(delay)
            waited += delay


def get_project_file(client, name):
    """Re-authorize every remote reference with the currently selected client.

    The provider's Files.get checks project ownership and expiry. A reference
    from another project must fail here, before generation; never auto-upload.
    """
    if not re.fullmatch(r'files/[A-Za-z0-9_-]+', name):
        raise RuntimeError('Use a Gemini Files API name in files/<id> format')
    try:
        resource = call_with_retry(lambda: client.files.get(name=name))
        if getattr(resource, 'name', None) != name:
            raise RuntimeError('Provider returned a different remote file reference')
        return resource
    except Exception as error:
        raise RuntimeError('Remote file is unavailable to the selected project or has expired. '
                           + classify_error(error).message + ' No upload performed.') from None


def upload_project_file(client, path, *, authorized=False):
    if not authorized:
        raise RuntimeError('A new Files API upload requires explicit --allow-upload authorization')
    return client.files.upload(file=str(path))
