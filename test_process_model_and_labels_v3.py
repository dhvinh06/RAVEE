"""Offline/mock tests only: no Gemini request, Graphviz run or full video scan."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'toolbox'))

import label_suggester as suggester
from toolbox import event_log_creation as events
from pm4py.visualization.bpmn import visualizer
from google import genai

READ_EXISTING_XES = events.pm4py.read_xes


def response(**updates):
    data = {'label': 'Fold wrapping paper', 'evidence': 'The fingers fold the visible paper edge in the target.',
            'uncertainty': 'low', 'target_visible': True, 'bow_action': 'not_applicable'}
    data.update(updates)
    return SimpleNamespace(text=json.dumps(data))


def capture(frames=1539, fps=15):
    cap = Mock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: fps if prop == cv2.CAP_PROP_FPS else frames
    return cap


class ProcessModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="RAVEE ' & model ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.xes = self.root / 'gift task_eventlog.xes'
        self.xes.write_bytes(b'existing XES')
        self.csv = self.root / 'gift task_eventlog.csv'
        self.csv.write_bytes(b'existing CSV')
        for owner, name in ((events.pm4py, 'read_xes'),
                            (events.pm4py, 'discover_process_tree_inductive'),
                            (events.pm4py, 'convert_to_bpmn'), (events.pm4py, 'view_bpmn'),
                            (visualizer, 'apply')):
            patcher = patch.object(owner, name)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        self.apply.return_value = SimpleNamespace(source='digraph { a [label="Wrap gift"]; }')
        self.stderr = io.StringIO()
        self.stdout = io.StringIO()
        for stream in (contextlib.redirect_stderr(self.stderr), contextlib.redirect_stdout(self.stdout)):
            stream.__enter__()
            self.addCleanup(stream.__exit__, None, None, None)

    def assert_logs_unchanged(self):
        self.assertEqual(self.xes.read_bytes(), b'existing XES')
        self.assertEqual(self.csv.read_bytes(), b'existing CSV')
        self.view_bpmn.assert_not_called()

    @staticmethod
    def render(argv, **kwargs):
        path = Path(argv[-1])
        path.write_bytes(b'PNG rendered' if argv[1] == '-Tpng' else b'<svg>rendered</svg>')
        return subprocess.CompletedProcess(argv, 0)

    def test_saves_both_formats_without_a_viewer_and_preserves_logs(self):
        with patch.object(events.subprocess, 'run', side_effect=self.render) as render:
            result = events.showProcessModel(self.xes)
        self.assertEqual(set(result), {'png', 'svg'})
        self.assertTrue(all(Path(path).is_file() for path in result.values()))
        for entry in render.call_args_list:
            self.assertEqual(entry.kwargs['timeout'], 30)
            self.assertTrue(entry.kwargs['check'])
            self.assertNotIn('shell', entry.kwargs)
            self.assertEqual(entry.kwargs['input'], self.apply.return_value.source)
        self.assert_logs_unchanged()

    def test_timeout_preserves_previous_image_and_allows_other_format(self):
        previous = self.root / 'gift task_eventlog_process_model.png'
        previous.write_bytes(b'previous complete image')

        def render(argv, **kwargs):
            if argv[1] == '-Tpng':
                Path(argv[-1]).write_bytes(b'partial')
                raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
            return self.render(argv, **kwargs)

        with patch.object(events.subprocess, 'run', side_effect=render):
            result = events.showProcessModel(self.xes, timeout=2)
        self.assertEqual(set(result), {'svg'})
        self.assertEqual(previous.read_bytes(), b'previous complete image')
        self.assertIn('TimeoutExpired', self.stderr.getvalue())
        self.assertFalse(list(self.root.glob('ravee_model_*')))
        self.assert_logs_unchanged()

    def test_missing_graphviz_and_empty_output_are_clear_nonfatal_errors(self):
        for effect in (FileNotFoundError('dot is missing'), lambda argv, **kwargs: None):
            with self.subTest(effect=effect):
                with patch.object(events.subprocess, 'run', side_effect=effect):
                    self.assertEqual(events.showProcessModel(self.xes), {})
        self.assertIn('Graphviz dot on PATH', self.stderr.getvalue())
        self.assertIn('CSV/XES files are preserved', self.stderr.getvalue())
        self.assert_logs_unchanged()

    def test_discovery_error_does_not_render_or_stop_caller(self):
        self.discover_process_tree_inductive.side_effect = ValueError('bad event log')
        with patch.object(events.subprocess, 'run') as render:
            self.assertEqual(events.showProcessModel(self.xes), {})
        render.assert_not_called()
        self.assertIn('bad event log', self.stderr.getvalue())
        self.assert_logs_unchanged()

    def test_existing_cached_xes_can_be_rendered_without_pipeline(self):
        # Read one existing event log locally; process discovery/render stay mocked.
        path = ROOT / 'EventLog_Output' / 'all_videos_task_eventlog.xes'
        if not path.exists():
            self.skipTest('User event-log cache is not present')
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        self.read_xes.side_effect = READ_EXISTING_XES
        with patch.object(events.subprocess, 'run', side_effect=self.render):
            result = events.showProcessModel(path, output_dir=self.root)
        self.read_xes.assert_called_once_with(str(path))
        self.assertEqual(set(result), {'png', 'svg'})
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)


class ContextTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'RAVEE_SUGGEST_CONTEXT_SECONDS': '3'})
        env.start()
        self.addCleanup(env.stop)

    def test_context_is_separate_and_boundaries_remain_exact(self):
        cap = capture()
        with patch.object(suggester.cv2, 'VideoCapture', return_value=cap):
            fps, windows = suggester._context_windows('cached.avi', 90, 149)
        self.assertEqual(fps, 15)
        self.assertEqual(windows, [('BEFORE CONTEXT', 45, 89), ('TARGET SEGMENT', 90, 149),
                                   ('AFTER CONTEXT', 150, 194)])
        cap.release.assert_called_once()

    def test_context_clamps_at_both_video_edges(self):
        with patch.object(suggester.cv2, 'VideoCapture', side_effect=lambda _: capture()):
            _, first = suggester._context_windows('cached.avi', 0, 9)
            _, last = suggester._context_windows('cached.avi', 1500, 1538)
        self.assertEqual(first, [('TARGET SEGMENT', 0, 9), ('AFTER CONTEXT', 10, 54)])
        self.assertEqual(last, [('BEFORE CONTEXT', 1455, 1499), ('TARGET SEGMENT', 1500, 1538)])

    def test_invalid_context_and_video_range_fail_before_request(self):
        with patch.dict(os.environ, {'RAVEE_SUGGEST_CONTEXT_SECONDS': 'nan'}):
            with self.assertRaises(ValueError):
                suggester._context_windows('cached.avi', 1, 2)
        with patch.object(suggester.cv2, 'VideoCapture', return_value=capture()):
            with self.assertRaises(ValueError):
                suggester._context_windows('cached.avi', 1, 1539)

    def test_existing_cached_video_uses_only_metadata_and_no_api(self):
        videos = list((ROOT / 'temp' / 'pipeline_cache' / 'preprocessing').glob('*/*.avi'))
        if not videos:
            self.skipTest('User preprocessing cache is not present')
        # Read only the cached video's header. No frame scan or clip encoding.
        with patch.object(genai, 'Client') as client:
            fps, windows = suggester._context_windows(videos[0], 90, 149)
        self.assertGreater(fps, 0)
        self.assertIn(('TARGET SEGMENT', 90, 149), windows)
        self.assertTrue(all(start >= 0 and end >= start for _, start, end in windows))
        client.assert_not_called()


class LabelEvidenceTests(unittest.TestCase):
    def test_new_bow_tightening_and_adjusting_have_distinct_consistent_names(self):
        for action, expected in (('tie_new', 'Tie ribbon bow'),
                                 ('tighten_existing', 'Tighten ribbon bow'),
                                 ('adjust_existing', 'Adjust ribbon bow')):
            with self.subTest(action=action):
                self.assertEqual(suggester._clean_label(response(bow_action=action)), expected)

    def test_occlusion_high_uncertainty_and_missing_evidence_force_unclear(self):
        for updates in ({'target_visible': False}, {'uncertainty': 'high'},
                        {'evidence': ''}, {'bow_action': 'unclear'}):
            with self.subTest(updates=updates):
                self.assertEqual(suggester._clean_label(response(**updates)), 'Unclear activity')

    def test_labels_are_open_vocabulary_and_repeated_wording_is_normalized(self):
        self.assertEqual(suggester._clean_label(response(label='ROTATE copper cylinder')), 'Rotate copper cylinder')
        self.assertEqual(suggester._clean_label(response(label='Rotate  copper cylinder.')), 'Rotate copper cylinder')

    def test_malformed_empty_and_unbounded_labels_are_errors(self):
        for result in (SimpleNamespace(text=''), SimpleNamespace(text='Fold paper'),
                       SimpleNamespace(text='[]'), response(target_visible='false'),
                       response(label='one two three four five six seven')):
            with self.subTest(result=result):
                with self.assertRaises(RuntimeError):
                    suggester._clean_label(result)


class RequestTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-dummy', 'GOOGLE_API_KEY': '',
                                      'RAVEE_SUGGEST_MODE': 'video', 'RAVEE_SUGGEST_CONTEXT_SECONDS': '3'})
        env.start()
        self.addCleanup(env.stop)
        client_patch = patch.object(genai, 'Client')
        self.client_constructor = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.client = self.client_constructor.return_value.__enter__.return_value
        self.client.models.generate_content.return_value = response()
        cap_patch = patch.object(suggester.cv2, 'VideoCapture', side_effect=lambda _: capture())
        cap_patch.start()
        self.addCleanup(cap_patch.stop)

    def test_video_request_marks_each_role_and_labels_only_target(self):
        with patch.object(suggester, '_make_clip', return_value=b'mock clip') as clips:
            label = suggester.suggest_activity('cached.avi', 90, 149)
        self.assertEqual(label, 'Fold wrapping paper')
        self.assertEqual([entry.args[1:] for entry in clips.call_args_list], [(45, 89), (90, 149), (150, 194)])
        request = self.client.models.generate_content.call_args.kwargs
        parts = request['contents'].parts
        texts = '\n'.join(part.text for part in parts if part.text)
        self.assertIn('never assign an action that occurs only outside the target', texts)
        self.assertIn('TARGET SEGMENT: source frames 90-149', texts)
        self.assertIn('Before/after appearance alone does not prove', texts)
        self.assertEqual(sum(part.inline_data is not None for part in parts), 3)
        self.assertEqual(request['config'].response_mime_type, 'application/json')
        self.assertEqual(self.client_constructor.call_args.kwargs['http_options'].timeout, 60000)
        self.client_constructor.return_value.__exit__.assert_called_once()

    def test_frames_mode_and_encoding_fallback_keep_all_roles(self):
        def frames(path, start, end, count):
            return [(start, start / 15, b'mock jpeg')]
        for mode in ('frames', 'video'):
            with self.subTest(mode=mode), patch.dict(os.environ, {'RAVEE_SUGGEST_MODE': mode}):
                with patch.object(suggester, '_make_clip', side_effect=RuntimeError('no codec')), \
                     patch.object(suggester, '_sample_frames', side_effect=frames) as sample, \
                     contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(suggester.suggest_activity('cached.avi', 90, 149), 'Fold wrapping paper')
                self.assertEqual([entry.args[1:] for entry in sample.call_args_list],
                                 [(45, 89, 3), (90, 149, 6), (150, 194, 3)])
                texts = '\n'.join(part.text for part in self.client.models.generate_content.call_args.kwargs['contents'].parts
                                  if part.text)
                self.assertIn('BEFORE CONTEXT frame 45', texts)
                self.assertIn('TARGET SEGMENT frame 90', texts)
                self.assertIn('AFTER CONTEXT frame 150', texts)

    def test_long_target_excerpt_is_disclosed_and_original_bounds_preserved(self):
        with patch.object(suggester, '_make_clip', return_value=b'clip'):
            suggester.suggest_activity('cached.avi', 0, 900)
        text = '\n'.join(part.text for part in self.client.models.generate_content.call_args.kwargs['contents'].parts
                         if part.text)
        self.assertIn('Original target: source frames 0-900 inclusive', text)
        self.assertIn('middle excerpt', text)
        self.assertIn('TARGET SEGMENT: source frames 300-599', text)

    def test_api_failures_propagate_for_existing_gui_manual_fallback(self):
        self.client.models.generate_content.side_effect = RuntimeError('503 service unavailable')
        with patch.object(suggester, '_make_clip', return_value=b'clip'):
            with self.assertRaisesRegex(RuntimeError, '503'):
                suggester.suggest_activity('cached.avi', 90, 149)
        self.client_constructor.return_value.__exit__.assert_called_once()

    def test_no_key_and_invalid_segment_do_not_call_api(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': '', 'GOOGLE_API_KEY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'Missing'):
                suggester.suggest_activity('cached.avi', 1, 2)
        with self.assertRaises(ValueError):
            suggester.suggest_activity('cached.avi', -1, 2)
        self.client_constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
