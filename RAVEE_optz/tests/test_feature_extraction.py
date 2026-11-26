"""Lossless FV preparation and optional profiling; synthetic tracks, mocked fits."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from threading import Event, get_ident
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from toolbox import feature_extraction as f
from toolbox import pipeline_cache as p
from toolbox.fv_profiling import FVProfile
from tests import test_pipeline_cache as fixtures
from gui.video_selection import VideoSelectionController


FIELDS = ('trajectory', 'hog', 'hof', 'mbh_x', 'mbh_y')


def synthetic_tracks(count=70, dtype='f4'):
    # Deliberately different storage order from the descriptor feature order.
    kinds = [dtype] * 5 if isinstance(dtype, str) else dtype
    sizes = ((15, 2), (96,), (108,), (96,), (96,))
    fields = [(name, kind, size) for name, kind, size in zip(FIELDS, kinds, sizes)]
    value = np.zeros(count, dtype=[fields[3], ('frame_num', 'i4', (1,)),
                                   fields[1], fields[0], fields[4], fields[2]])
    rng = np.random.default_rng(20261117)
    for name in FIELDS:
        # Fractional, signed values exercise float32/64 and field boundaries.
        value[name] = rng.standard_normal(value[name].shape) * 10
    value['frame_num'][:, 0] = np.arange(15, 15 + count)
    return value


def legacy_descriptors(tracks, indices=None):
    """Frozen pre-optimization list construction and concatenation order."""
    rows = range(len(tracks)) if indices is None else indices
    parts = [np.array([tracks[index][name] for index in rows]) for name in FIELDS]
    parts[0] = parts[0].reshape(len(rows), 30)
    combine = parts[0]
    for part in parts[1:]:
        combine = np.concatenate((combine, part), axis=1)
    return combine


def frame_mapping(tracks):
    result = {}
    for index in range(len(tracks)):
        end = int(np.asarray(tracks[index]['frame_num']).item())
        for frame in range(end - 14, end + 1):
            result.setdefault(frame, []).append(index)
    return result


def fixed_gmm():
    """Fixed model, no fitting, no random initialization or external work."""
    return SimpleNamespace(weights_=np.array([0.25, 0.75]),
                           means_=np.ones((2, 64)), covariances_=np.ones((2, 64)),
                           predict_proba=lambda x: np.tile([0.25, 0.75], (len(x), 1)))


class DescriptorTests(unittest.TestCase):
    def test_original_and_optimized_arrays_are_exact(self):
        for kind in ('f4', 'f8', ['i1', 'u1', 'f2', 'f8', 'f4']):
            value = synthetic_tracks(dtype=kind)
            for tracks in (value, value[::2], value[::-1], value[:1]):
                with self.subTest(kind=kind, shape=tracks.shape, strides=tracks.strides):
                    expected = legacy_descriptors(tracks)
                    actual = f.read_descriptors(tracks, 'all', None)
                    self.assertTrue(np.array_equal(expected, actual))
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.shape, (len(tracks), 426))
                    self.assertTrue(actual.flags.c_contiguous)

    def test_field_order_boundaries_and_input_not_modified(self):
        tracks = synthetic_tracks()
        for number, name in enumerate(FIELDS, 1):
            tracks[name] = number
        original = tracks.copy()
        result = f.read_descriptors(tracks, 'all', None)
        for number, (start, end) in enumerate(((0, 30), (30, 126), (126, 234),
                                               (234, 330), (330, 426)), 1):
            self.assertTrue(np.array_equal(result[:, start:end], np.full((70, end-start), number)))
        self.assertTrue(np.array_equal(tracks, original))

    def test_empty_all_has_correct_dimensions_and_dtype(self):
        for kind in ('f4', 'f8'):
            result = f.read_descriptors(synthetic_tracks(0, kind), 'all', None)
            self.assertEqual(result.shape, (0, 426))
            self.assertEqual(result.dtype, np.dtype(kind))

    def test_indexing_preserves_order_duplicates_and_local_pca_input(self):
        tracks = synthetic_tracks()
        all_desc = f.read_descriptors(tracks, 'all', None)
        for indices in ([69, 0, 69, 1] * 17, np.array([2, 2, 0, -1] * 17)):
            expected = legacy_descriptors(tracks, indices)
            self.assertTrue(np.array_equal(all_desc[indices], expected))
            with patch.object(f, 'compute_PCA', side_effect=lambda x: x[:, :64]) as pca, \
                 patch.object(f, 'fisher_vector', return_value=np.zeros(258)):
                f.read_descriptors(tracks, indices, None)
            self.assertTrue(np.array_equal(pca.call_args.args[0], expected))

    def test_empty_and_threshold_keep_original_zero_fallback(self):
        tracks = synthetic_tracks()
        for count in (0, 1, 63, 64):
            with self.subTest(count=count), patch.object(f, 'compute_PCA', side_effect=lambda x: x[:, :64]) as pca, \
                 patch.object(f, 'fisher_vector', return_value=np.zeros(258)) as fv, \
                 contextlib.redirect_stdout(io.StringIO()):
                f.read_descriptors(tracks, list(range(count)), None)
                self.assertEqual(pca.call_count, int(count > 63))
                data = fv.call_args.args[0]
                if count <= 63:
                    self.assertEqual(data.dtype, np.dtype('f8'))
                    self.assertTrue(np.array_equal(data, np.zeros((64, 64))))
                else:
                    self.assertTrue(np.array_equal(data, legacy_descriptors(tracks, range(64))[:, :64]))

    def test_out_of_bounds_is_not_silently_accepted(self):
        with self.assertRaises(IndexError):
            f.read_descriptors(synthetic_tracks(), [70], None)


class ExtractionTests(unittest.TestCase):
    def run_extraction(self, profile=None, progress=None):
        tracks = synthetic_tracks()
        # Non-monotonic end frames exercise dict insertion order, overlapping
        # memberships, first/last inclusive boundaries and both PCA branches.
        tracks['frame_num'][:, 0] = [80] * 65 + [15, 79, 120, 0, 80]
        matrices = []

        def pca(matrix, **kwargs):
            matrices.append(matrix.copy())
            return matrix[:, :64].copy()

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(f, 'compute_PCA', side_effect=pca), \
             patch.object(f, 'compute_GMM', return_value=fixed_gmm()), \
             patch.object(f, 'read_descriptors', wraps=f.read_descriptors) as reader, \
             contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / 'synthetic tracks.npy'
            np.save(path, tracks)
            result = f.extract_framewise_features(path, directory, profile=profile, progress=progress)
            written = np.loadtxt(Path(directory) / 'synthetic tracks-frame_fv.txt')
            self.assertTrue(np.array_equal(result, written))
            reader.assert_called_once()  # No per-frame structured reconstruction.
        return tracks, matrices, result

    def test_pipeline_matches_legacy_frame_inputs_and_fisher_results(self):
        tracks, matrices, result = self.run_extraction()
        mapping = frame_mapping(tracks)
        self.assertEqual(next(iter(mapping)), 66)  # Never sort frames.
        self.assertIn(-14, mapping)
        self.assertIn(120, mapping)
        self.assertNotIn(121, mapping)
        all_desc = legacy_descriptors(tracks)
        self.assertTrue(np.array_equal(matrices[0], all_desc))
        local = iter(matrices[1:-1])
        legacy_fvs = []
        for indices in mapping.values():
            expected = legacy_descriptors(tracks, indices)
            if len(indices) > 63:
                self.assertTrue(np.array_equal(next(local), expected))
                reduced = expected[:, :64].copy()
            else:
                reduced = np.zeros((64, 64))
            legacy_fvs.append(f.fisher_vector(reduced, fixed_gmm()))
        with self.assertRaises(StopIteration):
            next(local)
        expected = np.array(legacy_fvs)
        self.assertTrue(np.array_equal(matrices[-1], expected))
        self.assertTrue(np.array_equal(result, expected[:, :64]))
        # One trajectory belongs to each of its 15 inclusive frame memberships.
        self.assertEqual(sum(indices.count(0) for indices in mapping.values()), 15)

    def test_optional_profile_and_throttled_progress_do_not_change_results(self):
        _, _, baseline = self.run_extraction()
        profile = FVProfile()
        messages = []
        with patch.object(f, 'perf_counter', return_value=1.0):
            tracks, _, actual = self.run_extraction(profile, messages.append)
        self.assertTrue(np.array_equal(baseline, actual))
        self.assertEqual(profile.metadata['descriptors']['shape'], [70, 426])
        self.assertEqual(profile.metadata['frame_count'], len(frame_mapping(tracks)))
        self.assertEqual(profile.metadata['frame_descriptor_count'], 70 * 15)
        for stage in ('trajectory_loading', 'descriptor_assembly', 'trajectory_frame_mapping',
                      'frame_descriptor_preparation', 'gmm_posterior', 'fisher_statistics',
                      'feature_text_serialization'):
            self.assertIn(stage, profile.timings)
            self.assertGreaterEqual(profile.timings[stage]['seconds'], 0)
        frame_messages = [message for message in messages if message.startswith('Computing framewise')]
        self.assertEqual(len(frame_messages), 2)  # Start and completion, no flood.
        self.assertTrue(frame_messages[-1].split(': ')[1].split('/')[0] != '0')
        self.assertIn('Loading trajectories', messages)
        self.assertIn('Fitting PCA', messages)
        self.assertIn('Fitting GMM', messages)
        self.assertIn('Finalizing features: fitting final PCA', messages)

    def test_disabled_profiling_and_progress_do_not_read_clocks(self):
        with patch('toolbox.fv_profiling.perf_counter', side_effect=AssertionError('profile enabled')), \
             patch.object(f, 'perf_counter', side_effect=AssertionError('progress enabled')):
            self.run_extraction()

    def test_progress_throttle_reports_intermediate_frames(self):
        messages = []
        with patch.object(f, 'perf_counter', side_effect=(number * 0.25 for number in range(1000))):
            tracks, _, _ = self.run_extraction(progress=messages.append)
        frame_messages = [message for message in messages if message.startswith('Computing framewise')]
        total = len(frame_mapping(tracks))
        self.assertGreater(len(frame_messages), 2)
        self.assertLessEqual(len(frame_messages), total // 2 + 2)

    def test_real_pca_api_scope_solver_and_fit_transform_order_unchanged(self):
        data = np.ones((70, 426), dtype='f4')
        for profile in (None, FVProfile()):
            model = Mock()
            model.fit.return_value = model
            model.transform.return_value = data[:, :64]
            with patch.object(f, 'PCA', return_value=model) as constructor:
                result = f.compute_PCA(data, profile=profile, stage='global_pca')
            constructor.assert_called_once_with(n_components=64)
            model.fit.assert_called_once_with(data)
            model.transform.assert_called_once_with(data)
            self.assertEqual([call[0] for call in model.method_calls], ['fit', 'transform'])
            self.assertTrue(np.array_equal(result, data[:, :64]))
            if profile is not None:
                self.assertEqual(set(profile.timings), {'global_pca_fit', 'global_pca_transform'})

    def test_complete_profile_contains_all_training_and_frame_stages(self):
        tracks = synthetic_tracks()
        tracks['frame_num'] = 80  # 15 frames, each with 70 trajectories.
        profile = FVProfile()
        def pca(n_components):
            model = Mock()
            model.fit.return_value = model
            model.transform.side_effect = lambda value: value[:, :n_components].copy()
            return model
        gmm = fixed_gmm()
        gmm.fit = Mock(return_value=gmm)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(f, 'PCA', side_effect=pca), patch.object(f, 'GMM', return_value=gmm), \
             contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / 'tracks.npy'
            np.save(path, tracks)
            f.extract_framewise_features(path, directory, profile=profile)
        for stage in ('global_pca_fit', 'global_pca_transform', 'gmm_fit',
                      'frame_pca_fit', 'frame_pca_transform', 'final_pca_fit', 'final_pca_transform'):
            self.assertIn(stage, profile.timings)
        for stage in ('frame_descriptor_preparation', 'frame_pca_fit', 'gmm_posterior', 'fisher_statistics'):
            self.assertEqual(profile.timings[stage]['calls'], 15)
        self.assertEqual(profile.metadata['gmm_sample']['shape'], [70, 64])
        self.assertEqual(profile.metadata['frame_fisher_vectors']['shape'], [15, 258])
        self.assertEqual(profile.metadata['features']['shape'], [15, 64])

    def test_fv_substages_cross_gui_worker_queue_and_block_duplicate_work(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'selected.mp4'
            source.touch()
            started, release, finished = Event(), Event(), Event()
            workers = []
            def process(source, progress):
                workers.append(get_ident())
                started.set()
                release.wait(3)
                try:
                    return self.run_extraction(progress=progress)[2]
                finally:
                    finished.set()
            controller = VideoSelectionController(lambda _: {}, process)
            try:
                controller.select(source)
                self.assertTrue(controller.start())
                self.assertTrue(started.wait(3))
                self.assertTrue(controller.busy)
                self.assertFalse(controller.start())
                self.assertNotEqual(workers, [get_ident()])
                release.set()
                self.assertTrue(finished.wait(3))
                events = []
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    events.extend(controller.poll())
                    if any(kind in ('success', 'error') for kind, _ in events):
                        break
                    time.sleep(.005)
                self.assertFalse(any(kind == 'error' for kind, _ in events), events)
                messages = [value for kind, value in events if kind == 'progress']
                for expected in ('Loading trajectories', 'Fitting PCA', 'Fitting GMM',
                                 'Finalizing features: fitting final PCA'):
                    self.assertIn(expected, messages)
                self.assertTrue(any(kind == 'success' for kind, _ in events))
                self.assertFalse(controller.busy)
                self.assertEqual(len(workers), 1)
            finally:
                release.set()
                controller.close()
                controller.worker.join(3)

    def test_gmm_sampling_and_constructor_are_unchanged(self):
        data = np.ones((70, 64), dtype='f4')
        for sample_size in (69, 70):
            for profile in (None, FVProfile()):
                rng = Mock()
                rng.choice.return_value = data[:sample_size]
                model = Mock()
                model.fit.return_value = model
                with patch.object(f, 'SAMPLE_SIZE', sample_size), \
                     patch.object(f.np.random, 'default_rng', return_value=rng) as random, \
                     patch.object(f, 'GMM', return_value=model) as constructor:
                    f.compute_GMM(data, profile=profile)
                constructor.assert_called_once_with(n_components=64, covariance_type='diag')
                if sample_size < len(data):
                    random.assert_called_once_with()
                    rng.choice.assert_called_once_with(data, sample_size)
                    self.assertIs(model.fit.call_args.args[0], rng.choice.return_value)
                else:
                    random.assert_not_called()
                    self.assertIs(model.fit.call_args.args[0], data)
                if profile is not None:
                    self.assertEqual(set(profile.timings), {'gmm_fit'})

    def test_fisher_equations_bitwise_equal_with_fixed_model(self):
        xx = np.random.default_rng(7).normal(size=(70, 64))
        gmm = fixed_gmm()
        q = gmm.predict_proba(xx)
        q_sum = np.sum(q, 0)[:, np.newaxis] / len(xx)
        q_xx = np.dot(q.T, xx) / len(xx)
        q_xx_2 = np.dot(q.T, xx ** 2) / len(xx)
        expected = np.hstack((q_sum.squeeze() - gmm.weights_,
                              (q_xx - q_sum * gmm.means_).flatten(),
                              (-q_xx_2 - q_sum * gmm.means_ ** 2 + q_sum * gmm.covariances_
                               + 2 * q_xx * gmm.means_).flatten()))
        for profile in (None, FVProfile()):
            self.assertTrue(np.array_equal(f.fisher_vector(xx, gmm, profile=profile), expected))


class ProfilingCacheTests(unittest.TestCase):
    def test_profile_does_not_change_identity_and_hits_remain_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = p.StageCache(directory)
            inputs = {'extractor': 'exact-source-signature'}
            producer = Mock(return_value={'features': np.ones((84, 2))})
            validator = lambda arrays: p.validate_fv(arrays, 84)
            profile = FVProfile()
            actual = cache.arrays('fisher_vectors', inputs, producer, validator, profile=profile)
            path, manifest = cache.location('fisher_vectors', inputs, 'arrays.npz')
            original = (path.read_bytes(), manifest.read_bytes())
            hit_profile = FVProfile()
            cached = cache.arrays('fisher_vectors', inputs, producer, validator, profile=hit_profile)
            self.assertTrue(np.array_equal(actual['features'], cached['features']))
            producer.assert_called_once()
            self.assertEqual((path.read_bytes(), manifest.read_bytes()), original)
            self.assertFalse(profile.metadata['cache_hit'])
            self.assertTrue(hit_profile.metadata['cache_hit'])
            self.assertEqual(profile.timings['cache_serialization']['calls'], 1)
            self.assertNotIn('cache_serialization', hit_profile.timings)
            path.write_bytes(b'corrupt')
            cache.arrays('fisher_vectors', inputs, producer, validator, profile=hit_profile)
            self.assertEqual(producer.call_count, 2)
            self.assertFalse(hit_profile.metadata['cache_hit'])

    def test_producer_failure_never_publishes_complete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = p.StageCache(directory)
            def fail():
                raise RuntimeError('fit interrupted')
            with self.assertRaisesRegex(RuntimeError, 'fit interrupted'):
                cache.arrays('fisher_vectors', {}, fail, lambda _: None, profile=FVProfile())
            self.assertIsNone(cache.lookup('fisher_vectors', {}, 'arrays.npz'))

    def test_profile_aggregates_without_retaining_matrices_and_times_errors(self):
        profile = FVProfile()
        matrix = np.zeros((2, 426), dtype='f4')
        profile.matrix('descriptors', matrix)
        with patch('toolbox.fv_profiling.perf_counter', side_effect=[1.0, 1.2, 2.0, 2.3]):
            with profile.measure('repeat'):
                pass
            with self.assertRaises(ValueError), profile.measure('repeat'):
                raise ValueError('interrupted')
        self.assertEqual(profile.timings['repeat']['calls'], 2)
        self.assertAlmostEqual(profile.timings['repeat']['seconds'], .5)
        self.assertEqual(profile.metadata['descriptors']['bytes'], matrix.nbytes)
        json.dumps(profile.as_dict())


class PipelineIntegrationTests(unittest.TestCase):
    setUp = fixtures.PipelineTests.setUp
    reduce = fixtures.PipelineTests.reduce
    prepare = fixtures.PipelineTests.prepare

    def test_new_progress_hooks_and_profile_are_forwarded_without_breaking_cache(self):
        received = []
        def extractor(raw, work, *, progress=None, profile=None):
            received.append(profile)
            progress('Loading trajectories')
            progress('Fitting GMM')
            return np.ones((84, 2))
        self.extractor = extractor
        messages = []
        with patch.dict(p.os.environ, {'RAVEE_PROFILE_FV': '1'}):
            cache, _ = self.prepare(allow_idt=True, progress=messages.append)
            before = {path: path.read_bytes() for path in cache.root.rglob('*') if path.is_file()}
            self.prepare(progress=messages.append)
        self.assertEqual(len(received), 1)
        self.assertIsInstance(received[0], FVProfile)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertIn(f'Loading trajectories: {self.source.name}', messages)
        self.assertIn(f'Fitting GMM: {self.source.name}', messages)
        self.assertIn('cache_serialization', received[0].timings)

    def test_legacy_two_argument_extractor_remains_compatible_with_profiling(self):
        self.prepare(allow_idt=True, fv_profile=True)
        self.extractor.assert_called_once()
        self.assertEqual(self.extractor.call_args.kwargs, {})

    def test_code_change_preserves_old_fv_and_finch_artifacts(self):
        cache, cases = self.prepare(allow_idt=True)
        case = next(iter(cases.values()))
        finch = Mock(return_value=(np.zeros((84, 1), dtype=int), [1], None))
        p.cached_finch(cache, case, None, finch, self.root)
        old = {path: path.read_bytes() for stage in ('fisher_vectors', 'finch')
               for path in (cache.root / stage).rglob('*') if path.is_file()}
        (self.root / 'toolbox/feature_extraction.py').write_text('optimized source', encoding='utf-8')
        _, newer = self.prepare()
        p.cached_finch(cache, next(iter(newer.values())), None, finch, self.root)
        self.assertEqual(self.extractor.call_count, 2)
        self.assertEqual(finch.call_count, 2)
        self.assertEqual(len(list((cache.root / 'fisher_vectors').glob('*/complete.json'))), 2)
        self.assertEqual(len(list((cache.root / 'finch').glob('*/complete.json'))), 2)
        self.assertEqual(old, {path: path.read_bytes() for path in old})
        self.idt.assert_called_once()
        self.reducer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
