"""Offline startup/selection/cache tests; no Tk desktop, Docker, models or API calls."""

import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests import test_pipeline_cache as fixtures
from toolbox import main as app
from toolbox import pipeline_cache as cache
from gui.video_selection import VideoSelectionController


class SelectedVideoCacheTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    reduce = fixtures.PipelineTests.reduce
    prepare = fixtures.PipelineTests.prepare

    def status(self, path=None, settings=(15, 640, 480)):
        return cache.inspect_video_cache(self.root, self.root / 'cache', path or self.source, settings)

    def cluster(self, store, cases):
        finch = Mock(return_value=(np.zeros((84, 1), dtype=int), [1], None))
        for case in cases.values():
            cache.cached_finch(store, case, None, finch, self.root)
        return finch

    def test_selected_second_video_preserves_old_case_id_and_never_processes_first(self):
        earlier = self.source.parent / 'aaa-uncached.mp4'
        earlier.write_bytes(b'not selected')
        _, cases = self.prepare(videos=[self.source], allow_idt=True)
        self.assertEqual(list(cases), ["2-gift-wrap_user's video"])
        self.assertEqual(next(iter(cases.values()))['case_id'], 2)
        self.reducer.assert_called_once()
        self.assertEqual(Path(self.reducer.call_args.args[0]), self.source)
        self.idt.assert_called_once()
        self.extractor.assert_called_once()

    def test_cached_directory_video_is_reused_when_selected_individually(self):
        (self.source.parent / 'aaa.mp4').write_bytes(b'earlier video')
        store, cases = self.prepare(allow_idt=True)
        self.cluster(store, cases)
        calls = (self.reducer.call_count, self.idt.call_count, self.extractor.call_count)
        _, selected = self.prepare(videos=[self.source])
        self.assertEqual(len(selected), 1)
        self.assertEqual(calls, (self.reducer.call_count, self.idt.call_count, self.extractor.call_count))
        self.assertEqual(self.status()['status'], 'Cached')

    def test_full_warm_cache_and_readonly_status_reuse_all_four_stages(self):
        store, cases = self.prepare(videos=[self.source], allow_idt=True)
        finch = self.cluster(store, cases)
        before = {p: p.read_bytes() for p in (self.root / 'cache').rglob('complete.json')}
        status = self.status()
        self.assertEqual(status['status'], 'Cached')
        self.assertTrue(all(status['stages'].values()))
        again, selected = self.prepare(videos=[self.source])
        cache.cached_finch(again, next(iter(selected.values())), None, finch, self.root)
        self.reducer.assert_called_once()
        self.idt.assert_called_once()
        self.extractor.assert_called_once()
        finch.assert_called_once()
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_new_inspection_does_not_create_cache_or_start_producers(self):
        status = self.status()
        self.assertEqual(status['status'], 'New')
        self.assertFalse(any(status['stages'].values()))
        self.assertFalse((self.root / 'cache').exists())
        self.reducer.assert_not_called()
        self.idt.assert_not_called()
        self.extractor.assert_not_called()
        self.image.assert_not_called()

    def test_partial_fv_failure_resumes_without_preprocessing_or_idt(self):
        self.extractor.side_effect = RuntimeError('GMM interrupted')
        with self.assertRaisesRegex(RuntimeError, 'GMM interrupted'):
            self.prepare(videos=[self.source], allow_idt=True)
        status = self.status()
        self.assertEqual(status['status'], 'Partial Cache')
        self.assertEqual(status['stages'], dict(preprocessing=True, idt=True, fisher_vectors=False, finch=False))
        self.extractor.side_effect = lambda raw, work: np.ones((84, 2))
        store, cases = self.prepare(videos=[self.source])
        self.assertEqual(self.status()['status'], 'Partial Cache')
        self.cluster(store, cases)
        self.assertEqual(self.status()['status'], 'Cached')
        self.reducer.assert_called_once()
        self.idt.assert_called_once()
        self.assertEqual(self.extractor.call_count, 2)

    def test_external_browse_path_and_same_filename_different_content(self):
        external = self.root / 'elsewhere' / self.source.name
        external.parent.mkdir()
        external.write_bytes(b'different content, same filename')
        self.prepare(videos=[self.source], allow_idt=True)
        self.assertEqual(self.status(external)['status'], 'New')
        _, cases = self.prepare(videos=[external], allow_idt=True)
        self.assertEqual(len(cases), 1)
        self.assertEqual(Path(self.reducer.call_args.args[0]), external)
        self.assertEqual(self.idt.call_count, 2)

    def test_changed_source_or_settings_cannot_claim_cached_by_name(self):
        store, cases = self.prepare(allow_idt=True)
        self.cluster(store, cases)
        self.assertEqual(self.status(settings=(10, 640, 480))['status'], 'New')
        self.source.write_bytes(b'changed video at same path')
        self.assertEqual(self.status()['status'], 'New')

    def test_corrupt_fv_is_partial_and_only_fv_and_finch_rebuild(self):
        store, cases = self.prepare(allow_idt=True)
        self.cluster(store, cases)
        manifest = next((self.root / 'cache' / 'fisher_vectors').rglob('complete.json'))
        artifact = manifest.parent / 'arrays.npz'
        artifact.write_bytes(b'corrupt archive')
        before = manifest.read_bytes()
        self.assertEqual(self.status()['status'], 'Partial Cache')
        self.assertEqual(before, manifest.read_bytes())
        self.prepare(videos=[self.source])
        self.reducer.assert_called_once()
        self.idt.assert_called_once()
        self.assertEqual(self.extractor.call_count, 2)

    def test_selfconsistent_but_invalid_fv_rejected_by_real_validator(self):
        self.prepare(allow_idt=True)
        manifest = next((self.root / 'cache' / 'fisher_vectors').rglob('complete.json'))
        artifact = manifest.parent / 'arrays.npz'
        np.savez(artifact, features=np.ones((2, 2)))
        record = json.loads(manifest.read_text())
        record['sha256'] = cache.digest_file(artifact)
        manifest.write_text(json.dumps(record))
        status = self.status()
        self.assertFalse(status['stages']['fisher_vectors'])
        self.assertEqual(status['status'], 'Partial Cache')
        self.prepare(videos=[self.source])
        self.assertEqual(self.extractor.call_count, 2)

    def test_original_v1_manifest_remains_addressable_after_selector_changes(self):
        inputs = {'source': 'old source', 'settings': [15, 640, 480]}
        original = cache.fingerprint({'version': 1, 'inputs': inputs,
            'implementation': '8279078848cad387cc3285d06b49de91eaecd15880a1abf3858f0c1ca094738b'})
        directory = self.root / 'cache' / 'preprocessing' / original
        directory.mkdir(parents=True)
        artifact = directory / 'old.avi'
        artifact.write_bytes(b'original verified artifact')
        (directory / 'complete.json').write_text(json.dumps(dict(
            complete=True, inputs=inputs, sha256=cache.digest_file(artifact))))
        self.assertEqual(cache.StageCache(self.root / 'cache').lookup('preprocessing', inputs, 'old.avi'), artifact)

    def test_older_namespace_requires_exact_provenance_and_checksum(self):
        directory = self.root / 'cache' / 'preprocessing' / ('a' * 64)
        directory.mkdir(parents=True)
        artifact = directory / 'old.avi'
        artifact.write_bytes(b'older completed output')
        inputs = {'source': 'exact historical source'}
        manifest = directory / 'complete.json'
        manifest.write_text(json.dumps(dict(complete=True, inputs=inputs, sha256=cache.digest_file(artifact))))
        store = cache.StageCache(self.root / 'cache')
        self.assertEqual(store.lookup('preprocessing', inputs, 'old.avi'), artifact)
        self.assertIsNone(store.lookup('preprocessing', {'source': 'different'}, 'old.avi'))
        artifact.write_bytes(b'tampered')
        self.assertIsNone(store.lookup('preprocessing', inputs, 'old.avi'))

    def test_discovery_refresh_and_single_selection_validation(self):
        (self.source.parent / 'not-video.txt').touch()
        (self.source.parent / 'directory.mp4').mkdir()
        self.assertEqual(cache.discover_videos(self.source.parent), [self.source])
        new = self.source.parent / 'new.MP4'
        new.touch()
        self.assertIn(new, cache.discover_videos(self.source.parent))
        with self.assertRaisesRegex(ValueError, 'No input videos'):
            self.prepare(videos=[])
        with self.assertRaisesRegex(ValueError, 'Not a supported video'):
            self.prepare(videos=[self.root / 'missing.mp4'])


class SelectionControllerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.first, self.second = [Path(temp.name) / name for name in ('first.mp4', 'second.mp4')]
        self.first.touch()
        self.second.touch()
        self.inspect = Mock(return_value=dict(status='New', stages={}))
        self.process = Mock(return_value='prepared')
        self.controller = VideoSelectionController(self.inspect, self.process)
        self.addCleanup(self.controller.close)

    def wait_for(self, expected):
        deadline = time.monotonic() + 3
        found = []
        while time.monotonic() < deadline:
            found.extend(self.controller.poll())
            if any(kind == expected for kind, _ in found):
                return found
            time.sleep(.005)
        self.fail(f'Missing {expected} event; got {found}')

    def test_select_only_inspects_and_process_requires_one_video(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.controller.start()
        self.controller.select(self.second)
        self.wait_for('cache')
        self.inspect.assert_called_once_with(self.second)
        self.process.assert_not_called()
        self.controller.start()
        events = self.wait_for('success')
        self.assertIn(('success', (self.second, 'prepared')), events)
        self.assertEqual(self.process.call_args.args[0], self.second)

    def test_processing_is_background_reports_progress_and_blocks_double_submit(self):
        gate, started = Event(), Event()
        self.addCleanup(gate.set)
        def work(source, progress):
            progress('Extracting missing IDT')
            started.set()
            gate.wait(3)
            return 'done'
        self.process.side_effect = work
        self.controller.select(self.first)
        self.controller.start()
        self.assertTrue(started.wait(2))
        self.assertTrue(self.controller.busy)
        self.assertFalse(self.controller.start())
        with self.assertRaisesRegex(ValueError, 'Wait'):
            self.controller.select(self.second)
        self.assertIn(('progress', 'Extracting missing IDT'), self.wait_for('progress'))
        gate.set()
        self.wait_for('success')
        self.assertFalse(self.controller.busy)
        self.process.assert_called_once()

    def test_error_is_reported_and_can_retry_same_video(self):
        self.process.side_effect = RuntimeError('Docker is unavailable')
        self.controller.select(self.first)
        self.controller.start()
        self.assertIn(('error', 'Docker is unavailable'), self.wait_for('error'))
        self.assertFalse(self.controller.busy)
        self.process.side_effect = None
        self.controller.start()
        self.wait_for('success')

    def test_stale_cache_result_does_not_replace_current_selection(self):
        gate, started = Event(), Event()
        self.addCleanup(gate.set)
        def inspect(source):
            if source == self.first:
                started.set()
                gate.wait(3)
            return {'video': source}
        self.inspect.side_effect = inspect
        self.controller.select(self.first)
        self.assertTrue(started.wait(2))
        self.controller.select(self.second)
        gate.set()
        self.assertEqual(self.wait_for('cache'), [('cache', {'video': self.second})])
        self.process.assert_not_called()

    def test_close_before_processing_never_starts_pipeline(self):
        self.controller.select(self.first)
        self.controller.close()
        self.assertFalse(self.controller.start())
        self.process.assert_not_called()


class StartupTests(unittest.TestCase):
    def test_no_argument_entrypoint_opens_selector_before_processing(self):
        with patch.object(app, 'launch_interactive') as selector, patch.object(app, 'main') as pipeline:
            app.entrypoint([])
        selector.assert_called_once()
        pipeline.assert_not_called()

    def test_actual_script_noargs_attempts_tk_before_any_pipeline_work(self):
        class SelectorOpened(Exception):
            pass
        with patch.object(sys, 'argv', [str(fixtures.ROOT / 'toolbox/main.py')]), \
             patch('tkinter.Tk', side_effect=SelectorOpened) as root, \
             patch.object(cache, 'prepare_features', side_effect=AssertionError('Pipeline started too early')):
            with self.assertRaises(SelectorOpened):
                runpy.run_path(str(fixtures.ROOT / 'toolbox/main.py'), run_name='__main__')
        root.assert_called_once()

    def test_gui_process_authorizes_only_selected_path_and_hands_off_prepared_run(self):
        source, prepared, report = Path('coffee.mp4'), object(), Mock()
        def choose(directory, inspect, process):
            stages.assert_not_called()
            inspect(source)
            stages.assert_not_called()
            return source, process(source, report)
        with patch('gui.video_selection.select_video_to_process', side_effect=choose), \
             patch.object(app, 'inspect_video_cache', return_value={'status': 'New'}) as inspection, \
             patch.object(app, 'prepare_cases', return_value=prepared) as stages, \
             patch.object(app, 'main') as review:
            app.launch_interactive()
        self.assertEqual(stages.call_args.kwargs['videos'], [source])
        self.assertIs(stages.call_args.kwargs['allow_idt'], True)
        self.assertIs(stages.call_args.kwargs['progress'], report)
        inspection.assert_called_once()
        review.assert_called_once_with(videos=[source], prepared=prepared)

    def test_cancel_selector_does_not_process_or_review(self):
        with patch('gui.video_selection.select_video_to_process', return_value=None), \
             patch.object(app, 'prepare_cases') as stages, patch.object(app, 'main') as review:
            app.launch_interactive()
        stages.assert_not_called()
        review.assert_not_called()

    def test_cli_preserves_authorization_flags_and_single_path(self):
        with patch.object(app, 'main') as pipeline, patch.object(app, 'launch_interactive') as selector:
            app.entrypoint(['--no-gui', '--video', 'coffee.mp4', '--allow-idt',
                            '--adopt-existing-idt', '--adopt-existing-preprocessing', '--rebuild-idt'])
        selector.assert_not_called()
        self.assertEqual(pipeline.call_args.kwargs, dict(videos=['coffee.mp4'], allow_idt=True,
            adopt_preprocessing=True, adopt_idt=True, rebuild=True, prepare_only=False))

    def test_legacy_flags_keep_directory_cli_behavior(self):
        with patch.object(app, 'main') as pipeline, patch.object(app, 'launch_interactive') as selector:
            app.entrypoint(['--allow-idt'])
        selector.assert_not_called()
        self.assertIsNone(pipeline.call_args.kwargs['videos'])
        self.assertTrue(pipeline.call_args.kwargs['allow_idt'])

    def test_prepare_only_never_opens_review_or_exports(self):
        prepared = (object(), {'one': {}}, {'one': ()})
        with patch.object(app, 'prepare_cases', return_value=prepared) as stages, \
             patch.object(app, 'run_snippets') as labels, \
             patch.object(app, 'review_semantic_task_log') as review, \
             patch.object(app, 'export_two_event_logs') as export:
            self.assertEqual(app.main(videos=['coffee.mp4'], allow_idt=True, prepare_only=True), prepared)
        self.assertEqual(stages.call_args.kwargs['videos'], ['coffee.mp4'])
        labels.assert_not_called()
        review.assert_not_called()
        export.assert_not_called()

    def test_prepared_gui_result_reaches_existing_review_without_reprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            case = dict(video_path=str(Path(directory) / 'selected.avi'), case_id=2, video_name='selected')
            prepared = (object(), {'2-selected': case},
                        {'2-selected': (np.zeros((4, 1), dtype=int), [1], None)})
            labels = SimpleNamespace(LABEL={}, counter=1)
            def label_video(*args):
                labels.LABEL[1] = 'Make coffee'
                return 0
            detailed = app.pd.DataFrame({'activity': ['Make coffee']})
            with patch.object(app, 'finchOutput', str(Path(directory) / 'finch')), \
                 patch.object(app, 'eventlogOutput', str(Path(directory) / 'logs')), \
                 patch.object(app, 'gui_elements', labels), \
                 patch.object(app, 'prepare_cases', side_effect=AssertionError('Preparation duplicated')), \
                 patch.object(app, 'run_snippets', side_effect=label_video) as labeling, \
                 patch.object(app, 'extractLog', return_value=detailed), \
                 patch.object(app, 'review_semantic_task_log', return_value=detailed) as review, \
                 patch.object(app, 'export_two_event_logs') as export, \
                 patch.object(app, 'showProcessModel'), patch('builtins.print'):
                app.main(videos=['coffee.mp4'], prepared=prepared)
            labeling.assert_called_once()
            self.assertEqual(labeling.call_args.args[0], case['video_path'])
            review.assert_called_once()
            export.assert_called_once()


if __name__ == '__main__':
    unittest.main()
