"""Import the application in a clean process without starting external work."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ImportTests(unittest.TestCase):
    def test_application_and_utilities_import_without_starting_work(self):
        script = textwrap.dedent('''
            import importlib
            import socket
            import sys
            from pathlib import Path
            from unittest.mock import patch
            import tests  # Install fake credentials, dotenv isolation and SDK guard first.

            sys.path.insert(0, str(Path.cwd() / 'toolbox'))
            # Load third-party dependencies first: their platform detection may
            # use subprocess. Guard project imports against real runtime work.
            import cv2
            import moviepy
            import moviepy.video.io.VideoFileClip
            import moviepy.video.compositing.CompositeVideoClip
            import sklearn
            import pm4py
            import tkinter
            from google import genai

            def forbidden(*args, **kwargs):
                raise AssertionError('Import attempted runtime work')

            modules = [
                'finch.finch', 'custom_exceptions', 'file_video_stream',
                'preprocessing', 'feature_extraction', 'pipeline_cache',
                'display_video_snippets', 'event_log_creation',
                'toolbox.custom_exceptions', 'toolbox.file_video_stream',
                'toolbox.preprocessing', 'toolbox.feature_extraction',
                'toolbox.pipeline_cache', 'toolbox.display_video_snippets',
                'toolbox.event_log_creation', 'toolbox.semantic_aggregation',
                'label_suggester', 'batch_label_suggester', 'gui.gui_elements', 'gui.semantic_review',
                'gui.theme', 'gui.video_selection', 'gui.video_preview', 'gui.quota_recovery', 'gemini_runtime',
                'toolbox.main', 'integrate_label_suggester', 'make_contact_sheet',
                'ai_providers', 'twelvelabs_provider', 'gui.ai_upload', 'gui.lifecycle',
            ]
            with patch('socket.socket.connect', forbidden), \\
                 patch('subprocess.Popen', forbidden), \\
                 patch.object(genai, 'Client', forbidden), \\
                 patch.object(cv2, 'VideoCapture', forbidden), \\
                 patch.object(cv2, 'imwrite', forbidden), \\
                 patch('tkinter.Tk', forbidden):
                for name in modules:
                    importlib.import_module(name)
            print('Imported', len(modules), 'application/utility module paths')
        ''')
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Imported 32 application/utility module paths', result.stdout)


if __name__ == '__main__':
    unittest.main()
