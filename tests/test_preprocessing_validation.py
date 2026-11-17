"""Reducer/validator parity using synthetic frames; no actual video processing."""
import contextlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from toolbox import main as app  # Same sibling-import setup as the real entry point.
from toolbox import pipeline_cache as cache
from toolbox import preprocessing as reducer
from gui.video_selection import VideoSelectionController


class PreprocessingValidationTests(unittest.TestCase):
    def reduced_metadata(self, source):
        """Execute the unchanged reducer with mock frames, writer and no sleeps."""
        stream, writer = Mock(), Mock()
        stream.fps, stream.width, stream.height = source['fps'], source['width'], source['height']
        stream.start.return_value = stream
        stream.more.side_effect = [True] * source['frames'] + [False]
        stream.read.return_value = object()
        with patch.object(reducer, 'FileVideoStream', return_value=stream), \
             patch.object(reducer.cv2, 'VideoWriter', return_value=writer) as factory, \
             patch.object(reducer.cv2, 'VideoWriter_fourcc', return_value=0), \
             patch.object(reducer.cv2, 'resize'), patch.object(reducer.time, 'sleep'), \
             contextlib.redirect_stdout(io.StringIO()):
            reducer.reduce_video_size("source (online converter)'s video.mp4", 'synthetic output', 2, 'Make boat')
        writer.release.assert_called_once()
        return dict(frames=writer.write.call_count, fps=factory.call_args.args[2],
                    width=factory.call_args.args[3][0], height=factory.call_args.args[3][1])

    def test_converter_boat_fractional_fps_matches_the_actual_reducer(self):
        source = dict(frames=2351, fps=30000 / 1001, width=1280, height=720)
        actual = self.reduced_metadata(source)
        self.assertEqual(actual, dict(frames=2351, fps=15, width=640, height=480))
        with patch.object(cache, 'video_info', return_value=actual) as metadata:
            cache.validate_preprocessed('paper boat (converted)_reduced.avi', source, (15, 640, 480))
        metadata.assert_called_once_with('paper boat (converted)_reduced.avi', decode=True)

    def test_integer_fractional_and_low_fps_inputs_match_existing_sampling(self):
        cases = [(2153, 25., 1920, 1080, 1539), (2351, 30., 1280, 720, 1568),
                 (1709, 30000 / 1001, 640, 360, 1709),
                 (2351, 24000 / 1001, 1280, 720, None),
                 (2351, 60000 / 1001, 1280, 720, None),
                 (100, 10., 640, 360, 100), (100, 15., 1280, 480, 100)]
        for frames, fps, width, height, expected in cases:
            with self.subTest(fps=fps, dimensions=(width, height)):
                source = dict(frames=frames, fps=fps, width=width, height=height)
                actual = self.reduced_metadata(source)
                if expected is not None:
                    self.assertEqual(actual['frames'], expected)
                with patch.object(cache, 'video_info', return_value=actual):
                    cache.validate_preprocessed('synthetic.avi', source, (15, 640, 480))

    def test_wrong_counts_fps_and_resolution_still_fail_with_expected_actual_details(self):
        source = dict(frames=2351, fps=30000 / 1001, width=1280, height=720)
        valid = dict(frames=2351, fps=15, width=640, height=480)
        for changes, detail in [(dict(frames=2000), 'frame_count expected=2351 (+/- 2), actual=2000'),
                                (dict(frames=2348), 'frame_count expected=2351 (+/- 2), actual=2348'),
                                (dict(fps=30), 'fps expected=15 (+/- 0.01), actual=30'),
                                (dict(width=1280, height=720), 'resolution expected=640x480, actual=1280x720')]:
            with self.subTest(changes=changes), patch.object(cache, 'video_info', return_value=dict(valid, **changes)):
                with self.assertRaises(ValueError) as error:
                    cache.validate_preprocessed('named output (2).avi', source, (15, 640, 480))
                self.assertIn(detail, str(error.exception))
                self.assertIn('named output (2).avi', str(error.exception))
        with patch.object(cache, 'video_info', return_value=dict(frames=2000, fps=30, width=1280, height=720)):
            with self.assertRaises(ValueError) as error:
                cache.validate_preprocessed('all-wrong.avi', source, (15, 640, 480))
        self.assertTrue(all(name in str(error.exception) for name in ('frame_count', 'fps', 'resolution')))

    def test_complete_decoding_remains_required_and_has_precise_failure_message(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {
            cache.cv2.CAP_PROP_FRAME_COUNT: 4, cache.cv2.CAP_PROP_FPS: 15,
            cache.cv2.CAP_PROP_FRAME_WIDTH: 640, cache.cv2.CAP_PROP_FRAME_HEIGHT: 480}[prop]
        capture.read.side_effect = [(True, object()), (False, None)]
        with patch.object(cache.cv2, 'VideoCapture', return_value=capture):
            with self.assertRaisesRegex(ValueError, 'decoded frame_count expected=4 from metadata, actual=1'):
                cache.validate_preprocessed('truncated.avi', dict(frames=4, fps=15, width=640, height=480), (15, 640, 480))
        capture.release.assert_called_once()

    def test_invalid_metadata_names_the_invalid_field(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = [10, 0, 640, 480]
        with patch.object(cache.cv2, 'VideoCapture', return_value=capture):
            with self.assertRaisesRegex(ValueError, 'fps: expected finite > 0, actual=0'):
                cache.video_info('invalid.avi')
        capture.release.assert_called_once()

    def test_unreadable_paths_report_the_path_and_length_without_claiming_a_frame_mismatch(self):
        capture = Mock()
        capture.isOpened.return_value = False
        path = 'nested/' + 'long video (converter) ' * 12 + '.mp4'
        with patch.object(cache.cv2, 'VideoCapture', return_value=capture):
            with self.assertRaises(ValueError) as error:
                cache.video_info(path)
        self.assertIn(path, str(error.exception))
        self.assertIn(f'path_length={len(path)}', str(error.exception))
        self.assertIn('readable video expected=True, actual=False', str(error.exception))
        self.assertNotIn('frame_count', str(error.exception))
        capture.release.assert_called_once()

    def test_long_spaced_parenthesized_names_do_not_change_identity_or_cache_reuse(self):
        with tempfile.TemporaryDirectory(prefix="RAVEE's paths (test) ") as directory:
            root = Path(directory)
            for name in ('video_input', 'toolbox'):
                (root / name).mkdir()
            for name in ('preprocessing.py', 'file_video_stream.py'):
                (root / 'toolbox' / name).write_text('stable reducer signature')
            # Exercise a long filename within the host's Windows path limit;
            # this check does not promise support for arbitrary >260-char paths.
            source = root / 'video_input' / ("Paper boat (online converter)'s " + 'long name ' * 5 + '.mp4')
            source.write_bytes(b'synthetic source identity')
            inputs = cache.preprocessing_inputs(root, source, 1, (15, 640, 480))
            before = dict(inputs)
            store = cache.StageCache(root / 'cache')
            producer = Mock(side_effect=lambda path: path.write_bytes(b'synthetic output'))
            info = dict(frames=2351, fps=15, width=640, height=480)
            original = dict(frames=2351, fps=30000 / 1001, width=1280, height=720)
            with patch.object(cache, 'video_info', return_value=info):
                validate = lambda path: cache.validate_preprocessed(path, original, (15, 640, 480))
                path = store.file('preprocessing', inputs, inputs['name'], producer, validate)
                self.assertEqual(store.file('preprocessing', inputs, inputs['name'], producer, validate), path)
            producer.assert_called_once()
            self.assertEqual(inputs['source'], str(source.resolve()))
            self.assertEqual(inputs['name'], f'1_{source.stem}_reduced.avi')
            self.assertEqual(before, cache.preprocessing_inputs(root, source, 1, (15, 640, 480)))
            self.assertIsNone(store.lookup('preprocessing', dict(inputs, sha256='different'), inputs['name']))

    def test_selector_worker_forwards_detailed_validation_error_without_changing_ui_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'Paper boat (converted).mp4'
            source.write_bytes(b'synthetic')
            def process(path, progress):
                with patch.object(cache, 'video_info', return_value=dict(frames=2000, fps=15, width=640, height=480)):
                    cache.validate_preprocessed(path, dict(frames=2351, fps=30000 / 1001, width=1280, height=720), (15, 640, 480))
            controller = VideoSelectionController(lambda path: dict(status='New'), process)
            try:
                controller.select(source)
                self.assertTrue(controller.start())
                deadline, events = time.monotonic() + 3, []
                while time.monotonic() < deadline and not any(kind == 'error' for kind, _ in events):
                    events.extend(controller.poll())
                    time.sleep(.005)
                message = next(payload for kind, payload in events if kind == 'error')
                self.assertIn('frame_count expected=2351 (+/- 2), actual=2000', message)
                self.assertFalse(controller.busy)
            finally:
                controller.close()
