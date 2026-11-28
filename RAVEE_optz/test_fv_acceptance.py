"""Offline Fisher Vector acceptance tests."""

from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from tests.test_feature_extraction import synthetic_tracks
from toolbox import feature_extraction as fv, pipeline_cache


class FisherVectorAcceptanceTests(unittest.TestCase):
    def test_pipeline_defaults_to_validated_fv_and_can_select_legacy(self):
        keys = ('mode', 'random_state', 'fit_samples', 'gmm_tol', 'workers', 'posterior')
        project = Path(fv.__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {}, clear=True):
            source = Path(directory) / 'tracks.npy'
            np.save(source, synthetic_tracks())
            default = pipeline_cache.fv_inputs_for(project, source)
            with patch.dict('os.environ', {'RAVEE_FV_MODE': 'legacy'}):
                legacy = pipeline_cache.fv_inputs_for(project, source)

        self.assertEqual(tuple(default[key] for key in keys), ('optimized', 0, 100000, .003, 6, 'numpy'))
        self.assertEqual(tuple(legacy[key] for key in keys), ('legacy', None, 0, .001, 1, 'sklearn'))

    def test_same_pca_and_gmm_give_equivalent_frame_vectors(self):
        rng = np.random.default_rng(0)
        raw_frames = [rng.normal(size=(rows, 12)).astype('f4') for rows in (72, 72, 95)]
        with threadpool_limits(limits=1):
            frames = [(raw, PCA(n_components=4, random_state=1).fit(raw)) for raw in raw_frames]
            training = np.concatenate([pca.transform(raw) for raw, pca in frames])
            gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=2).fit(training)
            frames.append((np.zeros((64, 4)), None))  # The original float64 fallback.
            old = np.stack([fv.fisher_vector(raw if pca is None else pca.transform(raw), gmm)
                            for raw, pca in frames]).astype('f8')
            new = np.stack(fv._encode_frame_batch(frames, gmm, profile=None)).astype('f8')
            direct = np.stack(fv._encode_frame_batch(frames, gmm, profile=None,
                                                    posterior=fv._DiagonalPosterior(gmm))).astype('f8')

        self.assertEqual(new.shape, old.shape)
        self.assertTrue(np.isfinite(new).all())
        cosine = (old * new).sum(axis=1) / (np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1))
        self.assertGreaterEqual(float(cosine.min()), 0.999)
        np.testing.assert_array_equal(new, direct)

    def test_parallel_frame_fit_preserves_the_randomized_pca_stream(self):
        rows = np.random.default_rng(4)
        training = [(rows.normal(size=(n, 426)).astype('f4'), solver)
                    for n, solver in ((180, 'randomized'), (640, None), (80, 'full'))]
        serial_rng, parallel_rng = np.random.RandomState(7), np.random.RandomState(7)
        with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=3) as pool:
            serial = fv._fit_frame_pcas(training, serial_rng, None, None)
            parallel = fv._fit_frame_pcas(training, parallel_rng, pool, None)
        for first, second in zip(serial, parallel):
            np.testing.assert_array_equal(first.components_, second.components_)
        np.testing.assert_array_equal(serial_rng.normal(size=10), parallel_rng.normal(size=10))

    def test_explicit_seed_repeats_with_different_global_rng_state(self):
        tracks = synthetic_tracks(600)
        tracks['frame_num'][:, 0] = np.repeat(np.arange(15, 90), 8)
        self.addCleanup(np.random.set_state, np.random.get_state())

        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1), \
                patch.object(fv, 'SAMPLE_SIZE', 250), redirect_stdout(io.StringIO()):
            source = Path(directory) / 'tracks.npy'
            np.save(source, tracks)

            def extract(seed):
                return fv.extract_framewise_features(source, directory, mode='optimized',
                                                     random_state=seed, fit_samples=0,
                                                     gmm_tol=fv.DEFAULT_GMM_TOL)

            np.random.seed(3)
            first = extract(7)
            np.random.seed(47)
            repeat = extract(7)
            other = extract(8)

        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, other))


if __name__ == '__main__':
    unittest.main()
