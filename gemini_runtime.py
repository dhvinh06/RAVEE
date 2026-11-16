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
_SESSION_CREDENTIAL_NAME = None  # Explicit GUI selection; never an automatic fallback.


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
    if _SESSION_CREDENTIAL_NAME is not None:
        value = config.get(_SESSION_CREDENTIAL_NAME)
        if not value:
            raise RuntimeError('The explicitly selected Gemini credential is no longer configured')
        _SEEN_SECRETS.add(value)
        return Credential(_SESSION_CREDENTIAL_NAME, value)
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


def configured_credentials():
    """Safe display labels and private credentials; never display credential values."""
    config = configuration()
    names = sorted(name for name, value in config.items() if value and
                   (name == 'GOOGLE_API_KEY' or re.fullmatch(r'GEMINI_API_KEY(?:_[A-Za-z0-9_]+)?', name)))
    credentials, seen = [], set()
    for name in names:
        value = config[name]
        if value in seen:
            continue
        seen.add(value)
        _SEEN_SECRETS.add(value)
        if name == 'GEMINI_API_KEY':
            label = 'Project 1'
        elif name == 'GOOGLE_API_KEY':
            label = 'Google default'
        else:
            suffix = name.removeprefix('GEMINI_API_KEY_')
            label = 'Project ' + suffix + (' (alternate)' if suffix == '1' else '')
        # Only display safe profile names, even if an unusual env name embeds a key.
        credentials.append((redact_text(label, 80), Credential(name, value)))
    return credentials


def select_credential(name):
    """Called only after the user explicitly selects a configured profile."""
    global _SESSION_CREDENTIAL_NAME
    for _, credential in configured_credentials():
        if credential.name == name:
            _SESSION_CREDENTIAL_NAME = name
            return credential
    raise RuntimeError('Selected credential is no longer configured; refresh the configuration')


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


def _quota_identifiers(value):
    """Read quota fields, excluding Help links and descriptive boilerplate."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _quota_identifiers(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in ('quotaId', 'quotaMetric', 'quota_id', 'quota_metric', 'quota_limit'):
                if isinstance(item, str):
                    yield redact_text(item, None)
            elif key in ('error', 'details', 'violations'):
                yield from _quota_identifiers(item)


def _daily_request_quota(error):
    identifiers = [re.sub(r'[^a-z0-9]', '', item.lower())
                   for payload in (getattr(error, 'details', None),
                                   getattr(error, 'response_json', None))
                   for item in _quota_identifiers(payload)]
    if identifiers:
        # Tokens-per-day and spend limits must not be reported as RPD. When
        # structured details exist they take precedence over generic prose.
        return any('request' in item and 'perday' in item for item in identifiers)
    message = getattr(error, 'message', None)
    message = redact_text(message if isinstance(message, str) else str(error), None)
    # Require an explicit exhaustion statement in the actual message. Merely
    # listing daily/RPD limits in guidance, a URL or Help detail is insufficient.
    quota = r'(?:daily\s+request\s+(?:quota|limit)|requests?\s+per\s+day\s+(?:quota|limit)|RPD\s+(?:quota|limit))'
    exhausted = r'(?:exhausted|exceeded|depleted|reached)'
    statement = r'(?:^|[.!?\n:])\s*'
    return bool(re.search(statement + r'(?:your\s+|the\s+)?' + quota
                          + r'\s+(?:(?:is|has\s+been|was)\s+)?' + exhausted + r'\b', message, re.I)
                or re.search(statement + r'(?:you\s+(?:have\s+)?)?' + exhausted
                             + r'\s+(?:your\s+|the\s+)?' + quota + r'\b', message, re.I))


def classify_error(error):
    code = status_code(error)
    if getattr(error, 'kind', None) == 'invalid_response' or type(error).__name__ == 'ActivityResponseError':
        return ErrorInfo('invalid_response', False, 'Invalid model response: schema or evidence validation failed.', code)
    if code in (401, 403):
        return ErrorInfo('authentication', False, 'Authentication or permission failure; check the selected project credential.', code)
    if code == 429:
        # SDK QuotaFailure details contain quotaId/metric/dimensions. Daily wins
        # over RetryInfo, which may also be present on an exhausted daily quota.
        detail = redact_text(str(getattr(error, 'details', '')) + str(getattr(error, 'response_json', '')) + str(error), None)
        normalized = re.sub(r'[^a-z0-9]', '', detail.lower())
        if _daily_request_quota(error):
            return ErrorInfo('daily_quota', False, 'Daily quota exhausted (RPD). Wait for reset or manually select an authorized project; no key rotation performed.', code)
        if any(token in normalized for token in ('perminute', 'persecond', 'rpm', 'tpm', 'ratelimitexceeded', 'toomanyrequests')) or retry_after(error) > 0:
            return ErrorInfo('rate_limit', True, 'Temporary request/token rate limit (RPM/TPM).', code)
        return ErrorInfo('quota_unknown', False, 'HTTP 429: quota type unknown. Check project quota details before retrying.', code)
    if code in (500, 502, 503, 504):
        return ErrorInfo('service_unavailable', True, f'HTTP {code}: temporary Gemini service unavailability.', code)
    return ErrorInfo('request_error', False, 'Gemini request failed: ' + redact_text(error, 250), code)


def generate_with_retry(client, kwargs, progress=None, stats=None):
    if isinstance(client, RequestSession):
        return client.generate(kwargs, progress, stats)
    return call_with_retry(lambda: client.models.generate_content(**kwargs), progress, stats)


class RequestSession:
    """Pause one pending request on confirmed RPD; resume only by user decision.

    The callback runs on the caller's worker thread. CLI callers omit it and
    retain the existing bounded retry/error behavior. No request/media rebuild
    or cache lookup is repeated when the credential changes.
    """
    def __init__(self, credential, on_daily_quota=None, *, factory=None):
        self.credential = credential
        self.on_daily_quota = on_daily_quota
        self.factory = factory or create_client
        self.owner = self.client = None

    def __enter__(self):
        self.owner = self.factory(self.credential)
        self.client = self.owner.__enter__()
        return self

    def __exit__(self, *exc):
        owner, self.owner, self.client = self.owner, None, None
        if owner is not None:
            return owner.__exit__(*exc)

    def close(self):
        self.__exit__(None, None, None)

    def generate(self, kwargs, progress=None, stats=None):
        while True:
            try:
                return call_with_retry(lambda: self.client.models.generate_content(**kwargs), progress, stats)
            except Exception as error:
                if classify_error(error).kind != 'daily_quota' or self.on_daily_quota is None:
                    raise
                replacement = self.on_daily_quota(self.credential)
                if replacement is None:
                    raise
                if not isinstance(replacement, Credential) or replacement.key == self.credential.key:
                    raise RuntimeError('Select a different configured project credential explicitly') from None
                self.close()
                self.credential = replacement
                self.__enter__()
                # Production V3/aggregation use inline media. Never replay a
                # remote resource under a different project without Files.get.
                self._verify_remote_resources(kwargs.get('contents'))

    def _verify_remote_resources(self, contents):
        if contents is None:
            return
        if isinstance(contents, (list, tuple)):
            for item in contents:
                self._verify_remote_resources(item)
            return
        if isinstance(contents, dict):
            parts = contents.get('parts', [])
            file_data = contents.get('file_data', contents.get('fileData'))
            uri = contents.get('file_uri', contents.get('fileUri'))
        else:
            parts = getattr(contents, 'parts', None) or []
            file_data = getattr(contents, 'file_data', None)
            # The SDK accepts a Files API File object directly as content.
            uri = getattr(contents, 'file_uri', None) or getattr(contents, 'uri', None)
        for part in parts:
            self._verify_remote_resources(part)
        if file_data is not None:
            self._verify_remote_resources(file_data)
        if uri is not None:
            match = re.fullmatch(r'(?:https://generativelanguage\.googleapis\.com/v1(?:beta)?/)?(files/[A-Za-z0-9_-]+)', uri or '')
            if not match:
                raise RuntimeError('Remote media must be explicitly authorized for the selected project; no upload performed')
            get_project_file(self.client, match.group(1))


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
