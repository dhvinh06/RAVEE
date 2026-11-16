"""Verify isolation against inherited credentials, dotenv discovery and writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from google import genai
import gemini_runtime as runtime
from tests.offline_support import ROOT


class OfflineConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_fake_and_unmocked_clients_are_blocked(self):
        self.assertEqual(runtime.resolve_credential().key, 'offline-test-default-never-send')
        self.assertNotEqual(runtime._ENV_PATH, ROOT / '.env')
        with self.assertRaisesRegex(AssertionError, 'Real Gemini clients are forbidden'):
            genai.Client(api_key='offline-only')

    def test_project_env_reads_and_research_artifact_writes_are_blocked(self):
        with self.assertRaisesRegex(AssertionError, 'real project .env'):
            (ROOT / '.env').read_bytes()
        for relative in ('temp/forbidden-test.json', 'EventLog_Output/forbidden-test.csv',
                         'DemoFiles/forbidden-test.xes', 'finch/output/forbidden-test.csv'):
            with self.subTest(path=relative), self.assertRaisesRegex(AssertionError, 'protected research artifact'):
                (ROOT / relative).write_bytes(b'never written')

    def test_bootstrap_ignores_inherited_profiles_and_a_discoverable_dotenv(self):
        script = textwrap.dedent('''
            import json
            import tests
            import gemini_runtime as runtime
            config = runtime.configuration()
            print(json.dumps(dict(key=runtime.resolve_credential().key,
                model=runtime.get_model('label'),
                leaked=any(name in config for name in ('GEMINI_API_KEY_FOREIGN',
                    'GEMINI_ACTIVE_KEY', 'GEMINI_LABEL_MODEL', 'GOOGLE_API_KEY')))))
        ''')
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, '.env').write_text('GEMINI_API_KEY=dotenv-sentinel\nGEMINI_LABEL_MODEL=dotenv-model\n')
            for suffix in ('alpha', 'beta'):
                environment = dict(os.environ, PYTHONPATH=str(ROOT),
                    GEMINI_API_KEY='inherited-' + suffix,
                    GEMINI_API_KEY_FOREIGN='foreign-' + suffix,
                    GOOGLE_API_KEY='google-' + suffix, GEMINI_ACTIVE_KEY='FOREIGN',
                    GEMINI_LABEL_MODEL='foreign-model-' + suffix)
                result = subprocess.run([sys.executable, '-B', '-c', script],
                    cwd=directory, env=environment, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), dict(key='offline-test-default-never-send',
                    model='gemini-3.5-flash-lite', leaked=False))
