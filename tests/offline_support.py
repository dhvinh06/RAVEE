"""Test-process isolation only; production configuration/loading is unchanged."""
import atexit
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
_STACK = ExitStack()
_INSTALLED = False
TWELVELABS_CLIENT_FACTORY = None


def _protected_path(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return False
    try:
        relative = Path(os.fsdecode(value)).resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return False
    return relative.as_posix() == '.env' or (relative.parts and relative.parts[0] in
        {'temp', 'EventLog_Output', 'DemoFiles', 'video_input', 'IDT'}) or relative.parts[:2] in (
        ('finch', 'input'), ('finch', 'output'))


def _protect_artifacts(event, args):
    """Fail closed if a regression attempts to publish into research outputs."""
    if event == 'open':
        path, mode, flags = args
        writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writes and _protected_path(path):
            raise AssertionError('Offline test attempted to write a protected research artifact')
        if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() == ROOT / '.env':
            raise AssertionError('Offline test attempted to read the real project .env')
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir'):
        if _protected_path(args[0]):
            raise AssertionError('Offline test attempted to change a protected research directory')
    elif event == 'os.rename':
        if _protected_path(args[0]) or _protected_path(args[1]):
            raise AssertionError('Offline test attempted to move a protected research artifact')


def install_offline_isolation():
    """Run once when the tests package loads, before application/SDK imports."""
    global _INSTALLED, TWELVELABS_CLIENT_FACTORY
    if _INSTALLED:
        return
    _INSTALLED = True
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(('GEMINI_', 'TWELVELABS_', 'TWELVE_LABS_', 'RAVEE_SUGGEST_')) and
                   name not in ('RAVEE_LABEL_PROVIDER', 'RAVEE_AGGREGATION_PROVIDER') and
                   name not in ('GOOGLE_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS')}
    environment['GEMINI_API_KEY'] = 'offline-test-default-never-send'
    _STACK.enter_context(patch.dict(os.environ, environment, clear=True))
    # This also covers SDK imports which may otherwise discover the user's .env.
    _STACK.enter_context(patch('dotenv.load_dotenv', return_value=False))
    from google import genai
    _STACK.enter_context(patch.object(genai, 'Client', side_effect=AssertionError(
        'Real Gemini clients are forbidden in offline tests; mock the client explicitly')))
    import gemini_runtime
    _STACK.enter_context(patch.object(gemini_runtime, '_ENV_PATH', ROOT / 'tests' / 'absent-offline.env'))
    _STACK.enter_context(patch.object(gemini_runtime, '_INJECTED', {}))
    _STACK.enter_context(patch.object(gemini_runtime, '_SESSION_CREDENTIAL_NAME', None))
    import twelvelabs_provider
    TWELVELABS_CLIENT_FACTORY = twelvelabs_provider.create_client
    _STACK.enter_context(patch.object(twelvelabs_provider, 'create_client', side_effect=AssertionError(
        'Real TwelveLabs clients are forbidden in offline tests; mock the client explicitly')))
    sys.addaudithook(_protect_artifacts)
    atexit.register(_STACK.close)


def detailed_events(case_id=1, count=11):
    """Fresh deterministic data including fractional durations and PM4Py aliases."""
    import pandas as pd
    lengths = (3, 5, 2, 1, 7, 4, 2, 3, 5, 2, 1)
    origin = pd.Timestamp('2026-01-01T12:00:00Z')
    rows, start = [], 0
    for index in range(count):
        length = lengths[index % len(lengths)]
        label = 'Align "test", part' if index == 0 else f'Synthetic action {index + 1}'
        begin, finish = origin + pd.Timedelta(seconds=start / 2), origin + pd.Timedelta(seconds=(start + length) / 2)
        rows.append(dict(videoID=case_id, segment_id=index + 1, cluster_id=index % 4,
            activity=label, frame_start=start, frame_end=start + length - 1,
            timestamp_start=begin, timestamp_end=finish, duration_seconds=length / 2,
            **{'case:concept:name': str(case_id), 'concept:name': label,
               'time:timestamp': begin, '@@index': index, '@@case_index': 0}))
        start += length
    return pd.DataFrame(rows)
