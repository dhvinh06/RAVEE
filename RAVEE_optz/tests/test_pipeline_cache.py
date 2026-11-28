"""Targeted tests: no real Docker, GMM/PCA, FINCH or GUI runs."""

import ast
import contextlib
import csv
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from toolbox import pipeline_cache as p
from toolbox.fv_profiling import measure


ROOT = Path(__file__).resolve().parents[1]


def tracks():
    dtype = [('frame_num', 'i4', (1,)), ('trajectory', 'f4', (15, 2)),
             ('hog', 'f4', (96,)), ('hof', 'f4', (108,)),
             ('mbh_x', 'f4', (96,)), ('mbh_y', 'f4', (96,))]
    value = np.zeros(70, dtype=dtype)
    value['frame_num'][:, 0] = np.arange(15, 85)
    return value


def save_tracks(path):
    with Path(path).open('wb') as stream:
        np.save(stream, tracks())


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="RAVEE ' & spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = p.StageCache(self.root / 'cache')

    def test_interrupted_stage_never_becomes_a_hit(self):
        def fail(path):
            path.write_bytes(b'partial')
            raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            self.cache.file('idt', {'input': 1}, 'raw.npy', fail, lambda _: None)
        self.assertIsNone(self.cache.lookup('idt', {'input': 1}, 'raw.npy'))
        self.assertFalse(list(self.cache.root.rglob('complete.json')))

    def test_valid_file_reused_and_tampering_invalidates(self):
        producer = Mock(side_effect=lambda path: path.write_bytes(b'complete'))
        args = ('preprocessing', {'source': 'one'}, 'video.avi', producer, lambda _: None)
        first = self.cache.file(*args)
        self.assertEqual(self.cache.file(*args), first)
        producer.assert_called_once()
        first.write_bytes(b'corrupt')
        self.cache.file(*args)
        self.assertEqual(producer.call_count, 2)
        self.assertIsNone(self.cache.lookup('preprocessing', {'source': 'two'}, 'video.avi'))

    def test_failed_validation_does_not_publish(self):
        def reject(_):
            raise ValueError('partial result')
        with self.assertRaises(ValueError):
            self.cache.file('fv', {}, 'value', lambda path: path.touch(), reject)
        self.assertIsNone(self.cache.lookup('fv', {}, 'value'))

    def test_fv_and_finch_validation(self):
        with self.assertRaises(ValueError):
            p.validate_fv({'features': np.zeros((10, 2))}, rows=84)
        with self.assertRaises(ValueError):
            p.validate_fv({'features': np.full((84, 2), np.nan)})
        with self.assertRaises(ValueError):
            p.validate_finch({'c': np.zeros((4, 1)), 'num_clust': np.array([2]),
                              'req_c': np.array([])}, 4, None)

    def test_raw_validation_rejects_summaries_and_truncation(self):
        path = self.root / 'tracks.npy'
        save_tracks(path)
        p.validate_tracks(path, 100)
        self.assertEqual(p.framewise_rows(path), 84)
        with path.open('ab') as stream:
            stream.write(b'partial')
        with self.assertRaises(ValueError):
            p.validate_tracks(path, 100)
        np.save(path, np.zeros((1, 24832)))
        with self.assertRaises(ValueError):
            p.validate_tracks(path, 100)


class DescriptorTests(unittest.TestCase):
    def test_real_existing_output_is_video_level(self):
        tables = p.descriptor_tables(ROOT / 'IDT' / 'features')
        gift = '1_3197334-hd_1920_1080_25fps_reduced'
        self.assertIn(gift, tables['mbh'])
        self.assertEqual(tables['mbh'][gift].shape, (24832,))
        self.assertIn('person01_boxing_d1_uncomp', tables['hof'])
        self.assertNotIn(gift, tables['hof'])

    def test_named_rows_are_read_and_partial_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hof_features.csv'
            path.write_text(',name,0,1\n0,gift_wrapping,0.1,0.2\n', encoding='utf-8')
            self.assertIn('gift_wrapping', p.descriptor_tables(directory)['hof'])
            for row in ('0,gift_wrapping,0.1\n', '0,gift_wrapping,nan,0.2\n',
                        '0,gift_wrapping,0.1,0.2\n1,gift_wrapping,0.3,0.4\n'):
                with self.subTest(row=row):
                    path.write_text(',name,0,1\n' + row, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        p.descriptor_tables(directory)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="RAVEE ' & spaces ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ('video_input', 'IDT/data', 'IDT/features', 'IDT/src', 'toolbox', 'finch'):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        for relative in ('toolbox/preprocessing.py', 'toolbox/file_video_stream.py',
                         'toolbox/feature_extraction.py', 'finch/finch.py',
                         'IDT/run_idt.py', 'IDT/Dockerfile', 'IDT/src/code.cpp'):
            (self.root / relative).write_text('original code', encoding='utf-8')
        self.source = self.root / 'video_input' / "gift-wrap_user's video.mp4"
        self.source.write_bytes(b'input one')
        (self.root / 'IDT/data/person01_boxing_d1_uncomp.avi').write_bytes(b'sample source')
        self.reducer = Mock(side_effect=self.reduce)
        self.extractor = Mock(side_effect=lambda raw, work: np.ones((84, 2)))
        self.idt = Mock(side_effect=lambda prepared, destination, image, **options: save_tracks(destination))
        self.image = Mock(return_value='mock-image')
        self.info = {'frames': 100, 'fps': 15, 'width': 640, 'height': 480}
        for target, value in (('video_info', Mock(return_value=self.info)),
                              ('validate_preprocessed', Mock()),
                              ('extract_idt', self.idt), ('ensure_image', self.image)):
            patcher = patch.object(p, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def reduce(self, source, directory, case_id, name):
        (Path(directory) / f'{case_id}_{name}_reduced.avi').write_bytes(Path(source).read_bytes())

    def prepare(self, **kwargs):
        return p.prepare_features(self.root, self.root / 'cache', self.reducer,
                                  (15, 640, 480), self.extractor, **kwargs)

    def test_resume_reuses_preprocessing_idt_fv_and_finch(self):
        cache, cases = self.prepare(allow_idt=True)
        _, again = self.prepare()
        self.assertEqual(list(cases), ["1-gift-wrap_user's video"])
        self.reducer.assert_called_once()
        self.idt.assert_called_once()
        self.extractor.assert_called_once()
        self.image.assert_called_once()
        case = next(iter(again.values()))
        self.assertIn("gift-wrap_user's video", case['video_path'])
        self.assertTrue((self.root / 'IDT/data/person01_boxing_d1_uncomp.avi').exists())
        finch = Mock(return_value=(np.zeros((84, 1), dtype=int), [1], None))
        p.cached_finch(cache, case, None, finch, self.root)
        p.cached_finch(cache, case, None, finch, self.root)
        finch.assert_called_once()
        finch.return_value = (np.arange(84).reshape(-1, 1) % 2, [2], np.arange(84) % 2)
        p.cached_finch(cache, case, 2, finch, self.root)
        self.assertEqual(finch.call_count, 2)

    def test_source_change_invalidates_stages(self):
        self.prepare(allow_idt=True)
        self.source.write_bytes(b'changed source')
        self.prepare(allow_idt=True)
        self.assertEqual(self.reducer.call_count, 2)
        self.assertEqual(self.idt.call_count, 2)

    def test_settings_change_invalidates_preprocessing(self):
        self.prepare(allow_idt=True)
        p.prepare_features(self.root, self.root / 'cache', self.reducer,
                           (10, 640, 480), self.extractor, allow_idt=True)
        self.assertEqual(self.reducer.call_count, 2)

    def test_failed_finch_is_retried_without_repeating_previous_stages(self):
        cache, cases = self.prepare(allow_idt=True)
        case = next(iter(cases.values()))
        finch = Mock(side_effect=RuntimeError('interrupted clustering'))
        with self.assertRaises(RuntimeError):
            p.cached_finch(cache, case, None, finch, self.root)
        finch.side_effect = None
        finch.return_value = (np.zeros((84, 1), dtype=int), [1], None)
        p.cached_finch(cache, case, None, finch, self.root)
        self.assertEqual(finch.call_count, 2)
        self.reducer.assert_called_once()
        self.idt.assert_called_once()
        self.extractor.assert_called_once()

    def test_idt_source_change_keeps_preprocessing(self):
        self.prepare(allow_idt=True)
        (self.root / 'IDT/run_idt.py').write_text('new extractor', encoding='utf-8')
        self.prepare(allow_idt=True)
        self.reducer.assert_called_once()
        self.assertEqual(self.idt.call_count, 2)

    def test_failed_fv_retries_only_fv(self):
        self.extractor.side_effect = RuntimeError('interrupted GMM')
        with self.assertRaises(RuntimeError):
            self.prepare(allow_idt=True)
        self.extractor.side_effect = lambda raw, work: np.ones((84, 2))
        self.prepare()
        self.reducer.assert_called_once()
        self.idt.assert_called_once()
        self.assertEqual(self.extractor.call_count, 2)

    def test_partial_idt_is_not_cached(self):
        def partial(prepared, destination, image, **options):
            destination.write_bytes(b'partial npy')
            raise subprocess.CalledProcessError(1, 'mock-docker')
        self.idt.side_effect = partial
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare(allow_idt=True)
        self.idt.side_effect = lambda prepared, destination, image, **options: save_tracks(destination)
        self.prepare(allow_idt=True)
        self.reducer.assert_called_once()
        self.assertEqual(self.idt.call_count, 2)

    def test_default_never_starts_missing_extraction_or_preprocessing(self):
        with self.assertRaises(p.CacheMiss):
            self.prepare()
        self.reducer.assert_not_called()
        self.idt.assert_not_called()
        self.image.assert_not_called()

    def test_legacy_raw_adoption_and_global_csv_diagnostic(self):
        name = f'1_{self.source.stem}_reduced'
        legacy = self.root / 'IDT/data' / f'{name}.avi'
        legacy.write_bytes(b'complete legacy video')
        table = self.root / 'IDT/features/mbh_features.csv'
        table.write_text(f',name,0\n0,{name},0.25\n', encoding='utf-8')
        with self.assertRaisesRegex(p.CacheMiss, 'no frame indices'):
            self.prepare(adopt_preprocessing=True)
        save_tracks(self.root / 'IDT/features' / f'{name}-trajectory.npy')
        self.prepare(adopt_preprocessing=True, adopt_idt=True)
        self.reducer.assert_not_called()
        self.idt.assert_not_called()
        self.image.assert_not_called()
        self.assertTrue(table.exists())

    def test_fv_code_change_keeps_completed_idt(self):
        self.prepare(allow_idt=True)
        (self.root / 'toolbox/feature_extraction.py').write_text('new FV code', encoding='utf-8')
        self.prepare()
        self.reducer.assert_called_once()
        self.idt.assert_called_once()
        self.assertEqual(self.extractor.call_count, 2)


class DockerTests(unittest.TestCase):
    def test_image_reused_and_built_only_for_missing_or_changed_label(self):
        with patch.object(p.subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess([], 0, stdout='signature\n')
            self.assertEqual(p.ensure_image('path with spaces', 'signature'), 'ravee-idt:signature')
            self.assertEqual(run.call_count, 1)
            run.reset_mock()
            run.return_value = subprocess.CompletedProcess([], 1, stdout='')
            p.ensure_image('path with spaces', 'signature')
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.kwargs['cwd'], 'path with spaces')
            self.assertNotIn('shell', run.call_args.kwargs)

    def test_build_fingerprint_ignores_videos_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'features').mkdir()
            (root / 'Dockerfile').write_text('FROM base', encoding='utf-8')
            first = p.build_signature(root)
            (root / 'data/boxing.avi').touch()
            (root / 'features/mbh_features.csv').touch()
            self.assertEqual(first, p.build_signature(root))
            (root / 'Dockerfile').write_text('FROM new-base', encoding='utf-8')
            self.assertNotEqual(first, p.build_signature(root))

    def test_container_sees_only_current_input_and_paths_stay_single_arguments(self):
        with tempfile.TemporaryDirectory(prefix="RAVEE ' & spaces ") as directory:
            root = Path(directory)
            prepared = root / "1_user's gift video_reduced.avi"
            prepared.write_bytes(b'prepared')
            (root / 'person01_boxing_d1_uncomp.avi').write_bytes(b'sample')

            def docker(argv, **kwargs):
                self.assertEqual(argv[:3], ['docker', 'run', '--rm'])
                mounts = [argv[i + 1] for i, value in enumerate(argv) if value == '-v']
                data = Path(mounts[0].split(':/densetrack')[0])
                output = Path(mounts[1].split(':/densetrack')[0])
                self.assertEqual([path.name for path in data.iterdir()], [prepared.name])
                self.assertTrue(mounts[0].endswith(':/densetrack/data:ro'))
                self.assertIn('RAVEE_RAW_ONLY=1', argv)
                self.assertTrue(kwargs['check'])
                save_tracks(output / f'{prepared.stem}-trajectory.npy')

            destination = root / 'tracks.npy'
            with patch.object(p.subprocess, 'run', side_effect=docker):
                p.extract_idt(prepared, destination, 'mock-image')
            p.validate_tracks(destination, 100)
            self.assertEqual((root / 'person01_boxing_d1_uncomp.avi').read_bytes(), b'sample')


def orchestration_namespace(relative, **dependencies):
    """Execute actual orchestration definitions without optional model/GUI imports."""
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = dict(__file__=str(path), __name__='test_orchestration', os=os, sys=sys,
                     np=np, pd=pd, re=re, time=time, **dependencies)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class OrchestrationTests(unittest.TestCase):
    def test_raw_reader_uses_named_fields_and_numpy_scalar_frame_numbers(self):
        ns = orchestration_namespace('toolbox/feature_extraction.py', Path=Path, measure=measure)
        # A different field order is legitimate; positional indexing is unsafe.
        raw = tracks()
        raw['hog'] = 1
        raw['hof'] = 2
        combined = ns['read_descriptors'](raw, 'all', None)
        self.assertEqual(combined.shape, (70, 426))
        np.testing.assert_array_equal(combined[:, 30:126], 1)
        np.testing.assert_array_equal(combined[:, 126:234], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.npy'
            np.save(path, raw)
            ns['compute_PCA'] = Mock(side_effect=lambda data: np.zeros((len(data), 64)))
            ns['compute_GMM'] = Mock()
            ns['fisher_vector'] = Mock(return_value=np.zeros(128))
            with contextlib.redirect_stdout(io.StringIO()):
                result = ns['extract_framewise_features'](str(path), directory)
            self.assertEqual(result.shape, (84, 64))
            self.assertTrue((Path(directory) / 'raw-frame_fv.txt').exists())

    def test_labeling_and_both_event_logs_still_run_on_cache_hit(self):
        labels = SimpleNamespace(LABEL={}, counter=1)
        ns = orchestration_namespace('toolbox/main.py', gui_elements=labels)
        with tempfile.TemporaryDirectory() as directory:
            for key in ('processedVideoOutput', 'idtOutput', 'fvOutput', 'finchOutput',
                        'eventlogOutput', 'tempDirectoryPath'):
                ns[key] = str(Path(directory) / key)
            case = {'video_path': str(Path(directory) / 'exact_gift.avi'),
                    'case_id': 1, 'video_name': 'gift-wrap-with-hyphens'}
            ns['prepare_features'] = Mock(return_value=(Mock(), {'1-gift-wrap': case}))
            ns['cached_finch'] = Mock(return_value=(np.zeros((4, 1), dtype=int), [1], None))
            ns['FINCH'] = Mock()
            ns['reduce_video_size'] = Mock()
            ns['extract_framewise_features'] = Mock()
            ns['FPS_RESIZE'], ns['WIDTH_RESIZE'], ns['HEIGHT_RESIZE'] = 15, 640, 480

            def label(*args):
                labels.LABEL[0] = 'wrap gift'
                return 0

            ns['run_snippets'] = Mock(side_effect=label)
            ns['createDF'] = Mock(return_value=pd.DataFrame())
            detailed = pd.DataFrame({'activity': ['wrap gift']})
            ns['extractLog'] = Mock(return_value=detailed)
            ns['extractTaskLog'] = Mock(return_value=detailed.copy())
            ns['review_semantic_task_log'] = Mock(return_value=detailed.copy())
            for name in ('exportCSV', 'exportXES', 'showProcessModel'):
                ns[name] = Mock()
            with contextlib.redirect_stdout(io.StringIO()):
                ns['main']()
            self.assertEqual(ns['run_snippets'].call_args.args[0], case['video_path'])
            self.assertEqual(len(ns['run_snippets'].call_args.args), 3)
            self.assertEqual(labels.LABEL, {})
            self.assertEqual(ns['exportCSV'].call_count, 2)
            self.assertEqual(ns['exportXES'].call_count, 2)
            self.assertEqual([entry.args[1] for entry in ns['exportCSV'].call_args_list],
                             ['all_videos_detailed', 'all_videos_task'])
            self.assertEqual(ns['extractLog'].call_args.args[0], 1)
            ns['review_semantic_task_log'].assert_called_once()
            self.assertEqual(ns['review_semantic_task_log'].call_args.args[0], case['video_path'])
            ns['showProcessModel'].assert_called_once()
            self.assertTrue(ns['showProcessModel'].call_args.args[0].endswith('all_videos_task_eventlog.xes'))
            self.assertTrue((Path(ns['finchOutput']) / 'gift-wrap-with-hyphens-c.csv').exists())

    def test_raw_mode_emits_atomic_trajectories_and_default_csv_mode_is_preserved(self):
        with patch.dict(os.environ, {'RAVEE_RAW_ONLY': '1'}):
            ns = orchestration_namespace('IDT/run_idt.py')
        self.assertFalse(ns['HOG_FISHER_VECTOR'])
        self.assertFalse(ns['HOF_FISHER_VECTOR'])
        self.assertFalse(ns['MBH_FISHER_VECTOR'])
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory) / 'data', Path(directory) / 'features'
            data.mkdir()
            output.mkdir()
            (data / 'gift.avi').touch()
            ns['DATA_DIRECTORY'], ns['TARGET_DIRECTORY'] = str(data), str(output)
            ns['read_video'] = Mock(return_value=np.zeros((100, 2, 2)))
            ns['densetrack'] = SimpleNamespace(densetrack=Mock(return_value=tracks()))
            with contextlib.redirect_stdout(io.StringIO()):
                ns['main']()
            p.validate_tracks(output / 'gift-trajectory.npy', 100)
            self.assertFalse(list(output.glob('*.partial')))
            self.assertFalse(list(output.glob('*.csv')))
            ns['save_as_csv'](pd.DataFrame({'name': ['gift'], '0': [1.0]}), str(output / 'mbh_features.csv'))
            self.assertIn('gift', p.descriptor_tables(output)['mbh'])
            self.assertFalse(list(output.glob('*.partial')))
        with patch.dict(os.environ, {'RAVEE_RAW_ONLY': '0'}):
            defaults = orchestration_namespace('IDT/run_idt.py')
        self.assertTrue(defaults['MBH_FISHER_VECTOR'])

    def test_preprocessing_validator_rejects_short_but_readable_legacy_video(self):
        source = {'frames': 2153, 'fps': 25, 'width': 1920, 'height': 1080}
        complete = {'frames': 1539, 'fps': 15, 'width': 640, 'height': 480}
        with patch.object(p, 'video_info', return_value=complete):
            p.validate_preprocessed('mock.avi', source, (15, 640, 480))
        with patch.object(p, 'video_info', return_value={**complete, 'frames': 100}):
            with self.assertRaises(ValueError):
                p.validate_preprocessed('mock.avi', source, (15, 640, 480))


if __name__ == '__main__':
    unittest.main(verbosity=2)
