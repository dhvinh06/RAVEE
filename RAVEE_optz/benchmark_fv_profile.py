"""Profile Fisher Vector extraction from one saved trajectory NPY per process.

Run with: python -B benchmark_fv_profile.py PATH_TO_TRAJECTORY.npy
JSON goes to stdout and progress goes to stderr. The extractor's compatibility
text output is written to a temporary directory and hashed for comparison.
"""

import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np
import psutil
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_info, threadpool_limits

from toolbox import feature_extraction as fv
from toolbox.fv_profiling import FVProfile


STAGES = {
    'load_and_descriptors': ('trajectory_loading', 'descriptor_assembly', 'trajectory_frame_mapping'),
    'pca': tuple(f'{scope}_pca_{operation}' for scope in ('global', 'frame', 'final')
                 for operation in ('fit', 'transform')),
    'gmm_fit': ('gmm_fit',),
    'encoding_excluding_pca': ('frame_descriptor_preparation', 'gmm_posterior',
                               'fisher_statistics', 'frame_vector_assembly'),
    'post_normalization': ('power_normalize', 'L2_normalize'),
}


def peak_rss_bytes(process):
    if sys.platform == 'win32':
        return process.memory_info().peak_wset

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024


def compare_rows(expected, actual):
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape or not (np.isfinite(expected).all() and np.isfinite(actual).all()):
        raise ValueError('Fisher Vector comparison requires equal shapes and finite values')
    product = (expected * actual).sum(axis=-1)
    lengths = np.linalg.norm(expected, axis=-1) * np.linalg.norm(actual, axis=-1)
    both_zero = (~expected.any(axis=-1) & ~actual.any(axis=-1)).astype(float)
    cosine = np.clip(np.divide(product, lengths, out=both_zero, where=lengths != 0), -1, 1)
    return {
        'rows': len(expected), 'min_cosine': float(cosine.min()),
        'mean_cosine': float(cosine.mean()), 'max_abs_difference': float(np.abs(expected - actual).max()),
        'equal': bool(np.array_equal(expected, actual)),
    }


def profile_trajectory(trajectory, seed, *, mode='legacy', reference=None, check_fixed_models=False,
                       fit_samples=None, gmm_tol=None, workers=None, posterior=None):
    audit = FVProfile()
    pca_shapes = {}
    gmm_params = {}
    last_e_step = 0.0
    compute_pca, compute_gmm = fv.compute_PCA, fv.compute_GMM
    encode_batch = fv._encode_frame_batch
    fixed = {'rows': 0, 'min_cosine': 1.0, 'max_abs_difference': 0.0, 'equal': True, 'seconds': 0.0}

    def observe_pca(descriptors, dim_PCA=fv.PCA_DIM, *, profile=None, stage='pca', random_state=None):
        reduced = compute_pca(descriptors, dim_PCA, profile=profile, stage=stage, random_state=random_state)
        key = (stage, str(descriptors.dtype), str(reduced.dtype),
               descriptors.shape[1], reduced.shape[1])
        rows = descriptors.shape[0]
        stats = pca_shapes.setdefault(key, {
            'stage': stage, 'input_dtype': key[1], 'output_dtype': key[2],
            'input_columns': key[3], 'output_columns': key[4],
            'min_rows': rows, 'max_rows': rows, 'calls': 0,
        })
        stats['min_rows'] = min(stats['min_rows'], rows)
        stats['max_rows'] = max(stats['max_rows'], rows)
        stats['calls'] += 1
        return reduced

    def observe_gmm(descriptors, **options):
        def observe_e_step(model, *args, **kwargs):
            nonlocal last_e_step
            before = audit.timings.get('gmm_e_step', {}).get('seconds', 0.0)
            with audit.measure('gmm_e_step'):
                result = e_step(model, *args, **kwargs)
            last_e_step = audit.timings['gmm_e_step']['seconds'] - before
            return result

        e_step = fv.GMM._e_step
        # Scope method wrapping to model training. Encoding uses predict_proba.
        with ExitStack() as hooks:
            for cls, method, timing in (
                (fv.GMM, '_initialize_parameters', 'gmm_initialization'),
                (KMeans, 'fit', 'gmm_kmeans'),
                (fv.GMM, '_m_step', 'gmm_m_step'),
            ):
                hooks.enter_context(patch.object(cls, method, timed_call(getattr(cls, method), timing)))
            hooks.enter_context(patch.object(fv.GMM, '_e_step', observe_e_step))
            model = compute_gmm(descriptors, **options)
        params = model.get_params()
        gmm_params.update({name: params[name] for name in (
            'n_components', 'covariance_type', 'max_iter', 'n_init',
            'random_state', 'init_params', 'tol', 'reg_covar',
        )})
        if isinstance(params['random_state'], np.random.RandomState):
            gmm_params['random_state'] = f'per-extraction RandomState({seed})'
        gmm_params.update(n_iter=int(model.n_iter_), converged=bool(model.converged_))
        return model

    def compare_batch(frames, gmm, *, profile, posterior=None):
        actual = encode_batch(frames, gmm, profile=profile, posterior=posterior)
        started = perf_counter()
        expected = []
        for rows, pca in frames:
            if pca is not None:
                rows = pca.transform(rows)
            expected.append(fv.fisher_vector(rows, gmm))
        comparison = compare_rows(np.asarray(expected), np.asarray(actual))
        fixed['rows'] += comparison['rows']
        fixed['min_cosine'] = min(fixed['min_cosine'], comparison['min_cosine'])
        fixed['max_abs_difference'] = max(fixed['max_abs_difference'], comparison['max_abs_difference'])
        fixed['equal'] &= comparison['equal']
        fixed['seconds'] += perf_counter() - started
        return actual

    def timed_call(function, name):
        def observed(*args, **kwargs):
            with audit.measure(name):
                return function(*args, **kwargs)

        return observed

    def progress(message):
        if not message.startswith('Computing framewise') or message.rsplit(' ', 1)[-1].startswith('0/'):
            print(message, file=sys.stderr, flush=True)

    process = psutil.Process()
    with ExitStack() as stack:
        output = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='ravee-fv-profile-')))
        stack.enter_context(redirect_stdout(sys.stderr))
        stack.enter_context(patch.object(fv, 'compute_PCA', observe_pca))
        stack.enter_context(patch.object(fv, 'compute_GMM', observe_gmm))
        for name in STAGES['post_normalization']:
            stack.enter_context(patch.object(fv, name, timed_call(getattr(fv, name), name)))
        if check_fixed_models:
            stack.enter_context(patch.object(fv, '_encode_frame_batch', compare_batch))
        rss_before = process.memory_info().rss
        peak_before = peak_rss_bytes(process)
        started = perf_counter()
        features = fv.extract_framewise_features(trajectory, output, progress=progress, profile=audit,
                                                 mode=mode, random_state=seed, fit_samples=fit_samples,
                                                 gmm_tol=gmm_tol, workers=workers, posterior=posterior)
        total = perf_counter() - started
        peak = peak_rss_bytes(process)
        with (output / f'{trajectory.stem}-frame_fv.txt').open('rb') as stream:
            output_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()

    for name in STAGES['post_normalization']:
        audit.timings.setdefault(name, {'seconds': 0.0, 'calls': 0})

    timings = audit.timings
    groups = {stage: sum(timings[name]['seconds'] for name in names if name in timings)
              for stage, names in STAGES.items()}
    initialization = timings.get('gmm_initialization', {}).get('seconds', 0.0)
    em = (timings.get('gmm_e_step', {}).get('seconds', 0.0) - last_e_step
          + timings.get('gmm_m_step', {}).get('seconds', 0.0))
    gmm_fit_breakdown = {
        'initialization': initialization,
        'kmeans_within_initialization': timings.get('gmm_kmeans', {}).get('seconds', 0.0),
        'em_e_and_m_steps': em,
        'final_e_step': last_e_step,  # sklearn.fit_predict recomputes labels after EM.
        'other_fit_overhead': groups['gmm_fit'] - initialization - em - last_e_step,
    }
    breakdown = {
        'load_and_prepare': groups['load_and_descriptors'],
        'pca_with_training_sample_selection': groups['pca'] + sum(
            timings.get(stage, {}).get('seconds', 0.0) for stage in ('global_fit_sampling', 'local_fit_sampling')),
        'gmm_fit': groups['gmm_fit'],
        'encoding_excluding_pca': groups['encoding_excluding_pca'],
        'normalization_and_export': groups['post_normalization'] + timings['feature_text_serialization']['seconds'],
    }
    breakdown['unallocated_overhead'] = total - fixed['seconds'] - sum(breakdown.values())
    return {
        'trajectory': str(trajectory), 'mode': mode, 'seed': seed, 'total_seconds': total,
        'extraction_seconds': total - fixed['seconds'],
        'fixed_model_comparison': fixed if check_fixed_models else None,
        'reference_comparison': compare_rows(np.loadtxt(reference), features) if reference else None,
        'stages_seconds': groups, 'breakdown_seconds': breakdown, 'gmm_fit_breakdown_seconds': gmm_fit_breakdown,
        'pca': list(pca_shapes.values()), 'gmm_parameters': gmm_params,
        'ram_bytes': {'rss_before_fv': rss_before, 'peak_before_fv': peak_before,
                      'process_peak_at_fv_end': peak},
        'versions': {'numpy': np.__version__, 'scikit-learn': sklearn.__version__},
        'threadpools': threadpool_info(), 'feature_text_sha256': output_sha256,
        **audit.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--workers', type=int, help='Parallel frame PCA workers with one BLAS thread each')
    parser.add_argument('--posterior', choices=('sklearn', 'numpy'))
    parser.add_argument('--mode', choices=('legacy', 'optimized'), default='legacy')
    parser.add_argument('--fit-samples', type=int, help='Experimental fit sample size (0 disables sampling)')
    parser.add_argument('--gmm-tol', type=float, help='Experimental GMM convergence tolerance')
    parser.add_argument('--reference-text', type=Path)
    parser.add_argument('--check-fixed-models', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.check_fixed_models and args.mode != 'optimized':
        parser.error('--check-fixed-models requires --mode optimized')
    if args.mode != 'optimized' and any(value is not None for value in (
            args.fit_samples, args.gmm_tol, args.workers, args.posterior)):
        parser.error('FV tuning options require --mode optimized')
    with threadpool_limits(limits=args.threads):
        result = profile_trajectory(args.trajectory.resolve(strict=True), args.seed, mode=args.mode,
                                    reference=args.reference_text, check_fixed_models=args.check_fixed_models,
                                    fit_samples=args.fit_samples, gmm_tol=args.gmm_tol,
                                    workers=args.workers, posterior=args.posterior)
    print(json.dumps(result, indent=2))
    for key in ('fixed_model_comparison', 'reference_comparison'):
        comparison = result[key]
        if comparison is not None and comparison['min_cosine'] < 0.999:
            raise SystemExit(f'{key} failed the 0.999 minimum row cosine threshold')


if __name__ == '__main__':
    main()
