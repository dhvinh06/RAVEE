import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from toolbox.fv_profiling import measure
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits

# Number of clusters for Gaussian mixture - k∈{1,...,K} where each Gaussian k has a μ (defining center), cov (definies
# width) and p(mixing probability)
K = 64
# Number of dimensions to reduce to w PCA
PCA_DIM = 64
# Number of samples to take to fit the GMM
SAMPLE_SIZE = 200000
FRAME_BATCH_BYTES = 32 * 1024**2
DEFAULT_GMM_TOL = 1e-3


def fv_runtime_settings(mode, workers=None, posterior=None):
    if mode != 'optimized':
        return 1, 'sklearn'
    if workers is None:
        workers = os.environ.get('RAVEE_FV_WORKERS', '1')
    if isinstance(workers, str):
        workers = int(workers)
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError('RAVEE_FV_WORKERS must be a positive integer')
    if posterior is None:
        posterior = os.environ.get('RAVEE_FV_POSTERIOR', 'sklearn')
    if posterior not in ('sklearn', 'numpy'):
        raise ValueError('RAVEE_FV_POSTERIOR must be sklearn or numpy')
    return int(workers), posterior


def fv_settings(mode=None, random_state=None, fit_samples=None, gmm_tol=None):
    mode = os.environ.get('RAVEE_FV_MODE', 'legacy') if mode is None else mode
    if mode not in ('legacy', 'optimized'):
        raise ValueError('RAVEE_FV_MODE must be legacy or optimized')
    if random_state is None:
        random_state = os.environ.get('RAVEE_FV_SEED')
    if random_state is not None:
        if isinstance(random_state, str):
            random_state = int(random_state)
        if not isinstance(random_state, (int, np.integer)) or not 0 <= random_state < 2**32:
            raise ValueError('Fisher Vector seed must be an integer in [0, 2**32 - 1]')
        random_state = int(random_state)
    if mode == 'legacy':
        return mode, random_state, 0, DEFAULT_GMM_TOL

    if fit_samples is None:
        fit_samples = os.environ.get('RAVEE_FV_FIT_SAMPLES', '0')
    if isinstance(fit_samples, str):
        fit_samples = int(fit_samples)
    if (not isinstance(fit_samples, (int, np.integer)) or fit_samples < 0
            or 0 < fit_samples < max(PCA_DIM, K)):
        raise ValueError(f'Fisher Vector fit sample count must be 0 or an integer >= {max(PCA_DIM, K)}')
    if gmm_tol is None:
        gmm_tol = os.environ.get('RAVEE_FV_GMM_TOL', DEFAULT_GMM_TOL)
    gmm_tol = float(gmm_tol)
    if not math.isfinite(gmm_tol) or gmm_tol <= 0:
        raise ValueError('Fisher Vector GMM tolerance must be finite and positive')
    if (fit_samples or gmm_tol != DEFAULT_GMM_TOL) and random_state is None:
        random_state = 0
    return mode, random_state, int(fit_samples), gmm_tol


def fisher_vector(xx, gmm, *, profile=None):
    """
    Computes the fisher vector on a set of descriptors and GMM.

    :param xx: the descriptor (e.g. HOG/HOF/MBH) for each video.
    :type xx: array_like, shape (N, D) or (D, )
                The set of descriptors

    :param gmm: Gauassian mixture model of the descriptors.
    :type gmm: instance of sklearn mixture.GMM object

    :return: Fisher vector of the given descriptor.
    :rtype: array_like, shape (K + 2 * D * K ) where K is the number of GMM
            components and D is the dimension of each descriptor.

    This function was taken from this GitHub gist: https://gist.github.com/danoneata/9927923
    """
    xx = np.atleast_2d(xx)
    # Compute posterior probabilities.
    with measure(profile, 'gmm_posterior'):
        posterior = gmm.predict_proba(xx)
    return _fisher_statistics(xx, posterior, gmm, profile=profile)


def _fisher_statistics(xx, Q, gmm, *, profile=None):
    N = xx.shape[-2]
    with measure(profile, 'fisher_statistics'):
        # Compute the sufficient statistics of descriptors.
        Q_sum = np.sum(Q, axis=-2)[..., np.newaxis] / N
        if xx.ndim == 2:
            Q_xx = np.dot(Q.T, xx) / N
            Q_xx_2 = np.dot(Q.T, xx ** 2) / N
        else:
            transposed = Q.swapaxes(-1, -2)
            Q_xx = (transposed @ xx) / N
            Q_xx_2 = (transposed @ (xx ** 2)) / N

        # Compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum[..., 0] - gmm.weights_
        d_mu = Q_xx - Q_sum * gmm.means_
        d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_
            + 2 * Q_xx * gmm.means_)

        # Merge derivatives into a vector.
        leading = d_pi.shape[:-1]
        return np.concatenate((d_pi, d_mu.reshape(*leading, -1),
                               d_sigma.reshape(*leading, -1)), axis=-1)


def read_descriptors(descriptors, index, gmm):
    """
        Extracts the trajectories (D=30), HOG (D=96), HOF (D=108) & MBH (D=192) from tracks.

        :param descriptors: the improved dense trajectories returned by
                            densetrack.densetrack method.
        :type descriptors: array_like, shape (N, ) where N is the number of
                           trajectories. Value of each element is a np.void.

        :param index: Index of the descriptor to be read
        :type index: list, with number of descriptor indices; or 'all' if iteration over all indices

        :return: combination of all descriptors
        :rtype: array_like, shape (N, D) where N is the number of
                      descriptors and D is the dimension
    """
    all_rows = isinstance(index, str) and index == 'all'
    selection = slice(None) if all_rows else index
    # Field views avoid five Python loops. One concatenate preserves named field
    # order, row order and NumPy's dtype promotion without intermediate matrices.
    trajectory = descriptors['trajectory'][selection]
    parts = (trajectory.reshape(len(trajectory), 30),
             descriptors['hog'][selection], descriptors['hof'][selection],
             descriptors['mbh_x'][selection], descriptors['mbh_y'][selection])
    if all(part.dtype == parts[0].dtype for part in parts):
        combine = np.concatenate(parts, axis=1)
    else:
        # Unusual mixed field types retain historical intermediate promotion
        # and rounding, which need not match a single multi-type concatenate.
        combine = parts[0]
        for part in parts[1:]:
            combine = np.concatenate((combine, part), axis=1)
    return combine if all_rows else _frame_fisher_vector(combine, gmm)


def _frame_fisher_vector(combine, gmm, *, profile=None, **pca_options):
    # Keep the original local fitting scope and the <=63 descriptor fallback.
    if combine.shape[0] > 63:
        if profile is not None:
            pca_options.update(profile=profile, stage='frame_pca')
        combine = compute_PCA(combine, **pca_options)
    else:
        combine = np.zeros((64, 64))
        print("Warning: FV derived from np.zeros - if done to often clustering is random")
    return (fisher_vector(combine, gmm) if profile is None else
            fisher_vector(combine, gmm, profile=profile))


def _fit_pca(descriptor, dim, *, profile, stage, random_state=None, solver=None):
    options = {} if random_state is None else {'random_state': random_state}
    if solver is not None:
        options['svd_solver'] = solver
    with measure(profile, stage + '_fit'):
        return PCA(n_components=int(dim), **options).fit(descriptor)


def compute_PCA(descriptor,dim_PCA=PCA_DIM, *, profile=None, stage='pca', random_state=None):
    '''
    Compute PCA of given descriptor

    :param descriptor: descriptor(s) that have to be reduced in dimensionality
    :type descriptor: array_like, shape (N, D) where N is the number of
                      descriptors and D is the dimension of each descriptor
                      (e.g. 96 for HOG).

    :param dim_PCA: Dimensions that PCA has to reduce descriptor(s) to
    :type dim_PCA: int value (for our purpose set to 64 as default
    :return: reduced descriptor
    '''
    pca = _fit_pca(descriptor, dim_PCA, profile=profile, stage=stage, random_state=random_state)
    with measure(profile, stage + '_transform'):
        return pca.transform(descriptor)

def compute_GMM(descriptor, *, profile=None, random_state=None, sample_rng=None,
                fit_sample=False, tol=DEFAULT_GMM_TOL):
    '''
    Computes GMM for K components from features given by descriptor

    :param descriptor: The descriptors to fit the GMM with
    :type descriptor: array_like, shape (N, D) where N is the number of
                      descriptors and D is the dimension of each descriptor
                      (e.g. 96 for HOG).
    :return:  GMM
    '''
    #Get a random sample of SAMPLE_SIZE for creation of GMM
    if not fit_sample and descriptor.shape[0] > SAMPLE_SIZE:
        rng = np.random.default_rng() if sample_rng is None else sample_rng
        sample = rng.choice(descriptor,SAMPLE_SIZE)
    else:
        sample = descriptor
    if profile is not None:
        profile.matrix('gmm_sample', sample)
    options = {} if random_state is None else {'random_state': random_state}
    if tol != DEFAULT_GMM_TOL:
        options['tol'] = tol
    with measure(profile, 'gmm_fit'):
        gmm = GMM(n_components=K, covariance_type='diag', **options).fit(sample)
    if profile is not None:
        for name in ('n_iter_', 'converged_'):
            value = getattr(gmm, name, None)
            if isinstance(value, (bool, int, np.bool_, np.integer)):
                profile.metadata['gmm_' + name.rstrip('_')] = value.item() if isinstance(value, np.generic) else value
    return gmm

def power_normalize(xx, alpha=0.5):
    """Computes a alpha-power normalization for the matrix xx."""
    return np.sign(xx) * np.abs(xx) ** alpha


def L2_normalize(xx):
    """L2-normalizes each row of the data xx."""
    Zx = np.sum(xx * xx, 1)
    xx_norm = xx / np.sqrt(Zx[:, np.newaxis])
    xx_norm[np.isnan(xx_norm)] = 0
    return xx_norm

def normalize_fv(fv):
    '''
    Normalize the fisher vector with power & L2 normalization (power first see:
        https://lear.inrialpes.fr/pubs/2010/PSM10/PSM10_0766.pdf)
    :param fv: fisher vector of the descriptors
    :return: normalized fisher vector

    Functions from https://bitbucket.org/doneata/fv4a/src/9cd355701c1657eff11a71f8ce4cc42ddd381113/evaluate.py#lines-50:85
    '''
    power_norm = power_normalize(fv) #alpha value see: https://hal.inria.fr/hal-00830491v2/document
    l2_norm = L2_normalize(power_norm)
    return l2_norm

def _frame_indices(ends, optimized):
    # For ordered tracks, all rows belonging to a frame form one contiguous slice.
    if (optimized and len(ends) and ends.dtype.kind in 'iu' and ends.dtype.itemsize < 8
            and np.all(ends[1:] >= ends[:-1])):
        ends = ends.astype(np.int64, copy=False)
        unique_ends = np.unique(ends)
        starts = unique_ends - 14
        starts[1:] = np.maximum(starts[1:], unique_ends[:-1] + 1)
        frames = np.concatenate([np.arange(start, end + 1) for start, end in zip(starts, unique_ends)])
        first = np.searchsorted(ends, frames, side='left')
        last = np.searchsorted(ends, frames + 14, side='right')
        return {int(frame): slice(int(start), int(stop)) for frame, start, stop in zip(frames, first, last)}

    result = {}
    for index, end in enumerate(ends):
        end = int(end)
        for frame in range(end - 14, end + 1):
            result.setdefault(frame, []).append(index)
    return result


def _index_count(indices):
    return indices.stop - indices.start if isinstance(indices, slice) else len(indices)


def _temporal_fit_mask(ends, count, rng):
    """Take a proportional, uniform sample without replacement in 15-frame bins."""
    bins = ends.astype(np.int64, copy=False) // 15
    order = np.argsort(bins, kind='stable')
    boundaries = np.r_[0, np.flatnonzero(np.diff(bins[order])) + 1, len(ends)]
    allocation = (boundaries * count + len(ends) // 2) // len(ends)
    mask = np.zeros(len(ends), dtype=bool)
    for lo, hi, take in zip(boundaries[:-1], boundaries[1:], np.diff(allocation)):
        if take:
            mask[order[lo + rng.choice(hi - lo, size=take, replace=False)]] = True
    return mask


def _local_pca_solver(rows):
    n, width = rows.shape
    if width <= 1000 and n >= 10 * width:
        return 'covariance_eigh'
    if max(rows.shape) > 500 and PCA_DIM < .8 * min(rows.shape):
        return 'randomized'
    return 'full'


def _local_fit_rows(rows, selected, rng):
    # Keep the existing 64-component PCA valid in sparsely sampled frames.
    missing = min(len(rows), PCA_DIM + 1) - np.count_nonzero(selected)
    if missing > 0:
        selected = selected.copy()
        selected[rng.choice(np.flatnonzero(~selected), size=missing, replace=False)] = True
    if selected.all():
        return rows, None
    # Match auto selection on the original frame size so small samples don't
    # inadvertently switch a randomized PCA fit to a full SVD.
    return rows[selected], _local_pca_solver(rows)


def _frame_batches(frame_indices, descriptors):
    batch = []
    used = 0
    row_bytes = descriptors.shape[1] * max(descriptors.dtype.itemsize, 4)
    for indices in frame_indices:
        rows = _index_count(indices)
        size = 64 * 64 * 8 if rows <= 63 else (rows + PCA_DIM + 1) * row_bytes
        if batch and used + size > FRAME_BATCH_BYTES:
            yield batch
            batch = []
            used = 0
        batch.append(indices)
        used += size
    if batch:
        yield batch


def _project_frame_batch(frames, *, profile):
    """Apply each frame's own PCA, grouping matrices of the same shape."""
    projected = [None] * len(frames)
    groups = {}
    for index, (rows, pca) in enumerate(frames):
        if pca is None:
            projected[index] = rows  # The existing float64 zero fallback.
        else:
            groups.setdefault((rows.shape, rows.dtype, pca.components_.dtype), []).append(index)

    for indices in groups.values():
        with measure(profile, 'frame_pca_transform'):
            if len(indices) == 1:
                index = indices[0]
                rows, pca = frames[index]
                axes = pca.components_.T
                projected[index] = rows @ axes
                projected[index] -= pca.mean_.reshape(1, -1) @ axes
                continue
            inputs = np.stack([frames[index][0] for index in indices])
            axes = np.stack([frames[index][1].components_ for index in indices]).swapaxes(1, 2)
            means = np.stack([frames[index][1].mean_ for index in indices])[:, None, :]
            # Same order as sklearn's non-whitened PCA transform.
            reduced = inputs @ axes
            reduced -= means @ axes
            for index, rows in zip(indices, reduced):
                projected[index] = rows
    return projected


class _DiagonalPosterior:
    """Reuse diagonal-GMM constants and the sklearn float32 operation order."""

    def __init__(self, gmm):
        self.gmm = gmm
        self.valid = gmm.covariance_type == 'diag' and all(
            matrix.dtype == np.float32 for matrix in (gmm.means_, gmm.precisions_cholesky_, gmm.weights_))
        if not self.valid:
            return
        means, chol = gmm.means_, gmm.precisions_cholesky_
        self.precision = chol ** 2
        self.mean_precision = (means * self.precision).T
        self.mean_squared = np.sum(means ** 2 * self.precision, axis=1)
        self.log_det = np.sum(np.log(chol), axis=1)
        self.log_weight = np.log(gmm.weights_)
        self.gaussian_constant = means.shape[1] * math.log(2 * math.pi)

    def __call__(self, rows):
        if not self.valid or rows.dtype != np.float32:
            return self.gmm.predict_proba(rows)
        logits = rows @ self.mean_precision
        logits *= -2.0
        logits += self.mean_squared
        logits += (rows ** 2) @ self.precision.T
        logits += self.gaussian_constant
        logits *= -0.5
        logits += self.log_det
        logits += self.log_weight

        # Keep sklearn's log-sum-exp order, including ties at the maximum.
        largest = logits.max(axis=1, keepdims=True)
        if not np.isfinite(largest).all():
            return self.gmm.predict_proba(rows)
        shifted = logits - largest
        ties = shifted == 0
        count = ties.sum(axis=1, keepdims=True, dtype=logits.dtype)
        shifted[ties] = -np.inf
        with np.errstate(under='ignore'):
            np.exp(shifted, out=shifted)
            log_norm = np.log1p(shifted.sum(axis=1, keepdims=True) / count) + np.log(count) + largest
            logits -= log_norm
            np.exp(logits, out=logits)
        return logits


def _encode_frame_batch(frames, gmm, *, profile, posterior=None):
    projected = _project_frame_batch(frames, profile=profile)
    predict = gmm.predict_proba if posterior is None else posterior
    groups = {}
    for index, rows in enumerate(projected):
        groups.setdefault(rows.dtype, []).append(index)
    vectors = [None] * len(projected)
    for indices in groups.values():
        blocks = [projected[index] for index in indices]
        offsets = np.cumsum([len(rows) for rows in blocks])[:-1]
        with measure(profile, 'gmm_posterior'):
            probabilities_batch = predict(np.concatenate(blocks) if len(blocks) > 1 else blocks[0])
        same_size = {}
        for index, rows, probabilities in zip(indices, blocks, np.split(probabilities_batch, offsets)):
            same_size.setdefault(len(rows), []).append((index, rows, probabilities))
        for group in same_size.values():
            frame_ids, rows, probabilities = zip(*group)
            if len(group) == 1:
                vectors[frame_ids[0]] = _fisher_statistics(rows[0], probabilities[0], gmm, profile=profile)
            else:
                # Matrix products are independent across frames; no padding or cross-frame sums.
                encoded = _fisher_statistics(np.stack(rows), np.stack(probabilities), gmm, profile=profile)
                for index, vector in zip(frame_ids, encoded):
                    vectors[index] = vector
    return vectors


def _fit_frame_pcas(training, random_state, executor, profile):
    def fit(rows, solver, rng, timing=None):
        return _fit_pca(rows, PCA_DIM, profile=timing, stage='frame_pca', random_state=rng, solver=solver)

    if executor is None or len(training) < 2:
        return [fit(rows, solver, random_state, profile) for rows, solver in training]

    rng = check_random_state(random_state)
    initial_state = rng.get_state()
    jobs = []
    with measure(profile, 'frame_pca_fit'):
        for rows, solver in training:
            worker_rng = np.random.RandomState(0)
            worker_rng.set_state(rng.get_state())
            if (solver or _local_pca_solver(rows)) == 'randomized':
                # Randomized PCA draws one Gaussian starting matrix using the
                # smaller input dimension and sklearn's 10 default oversamples.
                rng.normal(size=(min(rows.shape), PCA_DIM + 10))
            expected = rng.get_state()
            with measure(profile, 'pca_worker_submit'):
                future = executor.submit(fit, rows, solver, worker_rng)
            jobs.append((future, worker_rng, expected))

        with measure(profile, 'pca_worker_wait'):
            models = [future.result() for future, _, _ in jobs]
        for _, worker_rng, expected in jobs:
            state = worker_rng.get_state()
            if state[2:] != expected[2:] or not np.array_equal(state[1], expected[1]):
                # A different sklearn solver or RNG draw must not alter the
                # seed stream used by subsequent frames and the final PCA.
                rng.set_state(initial_state)
                if profile is not None:
                    profile.metadata['serial_pca_fallback_batches'] += 1
                return [fit(rows, solver, random_state) for rows, solver in training]
        if profile is not None:
            profile.metadata['parallel_pca_batches'] += 1
    return models


def _batched_fisher_vectors(descriptors, indices, gmm, *, profile, random_state, report_frames,
                           fit_mask=None, fit_rng=None, executor=None, posterior=None):
    result = None
    completed = 0
    pca_fit_rows = 0
    for batch in _frame_batches(indices.values(), descriptors):
        frames = []
        training_rows = []
        for frame_indices in batch:
            with measure(profile, 'frame_descriptor_preparation'):
                rows = descriptors[frame_indices]
            if len(rows) > 63:
                training, solver = rows, None
                if fit_mask is not None:
                    with measure(profile, 'local_fit_sampling'):
                        training, solver = _local_fit_rows(rows, fit_mask[frame_indices], fit_rng)
                pca_fit_rows += len(training)
                training_rows.append((training, solver))
                has_pca = True
            else:
                rows = np.zeros((64, 64))
                has_pca = False
                print("Warning: FV derived from np.zeros - if done to often clustering is random")
            frames.append((rows, has_pca))
        fitted = iter(_fit_frame_pcas(training_rows, random_state, executor, profile))
        frames = [(rows, next(fitted) if has_pca else None) for rows, has_pca in frames]
        if posterior is None:
            vectors = _encode_frame_batch(frames, gmm, profile=profile)
        else:
            vectors = _encode_frame_batch(frames, gmm, profile=profile, posterior=posterior)
        with measure(profile, 'frame_vector_assembly'):
            packed = np.array(vectors)
            if result is None:
                result = np.empty((len(indices), packed.shape[1]), dtype=packed.dtype)
            else:
                dtype = np.result_type(result.dtype, packed.dtype)
                if dtype != result.dtype:
                    promoted = np.empty(result.shape, dtype=dtype)
                    promoted[:completed] = result[:completed]
                    result = promoted
            result[completed:completed + len(packed)] = packed
        completed += len(packed)
        report_frames(completed)
    if profile is not None:
        profile.metadata['local_pca_fit_rows'] = pca_fit_rows
    return result


def extract_framewise_features(input_path, outputPath, *, progress=None, profile=None,
                               mode=None, random_state=None, fit_samples=None, gmm_tol=None,
                               workers=None, posterior=None):
    """Extract framewise FVs with optional batching and a per-extraction integer seed.

    ``profile`` accepts an FVProfile; omitted profiling does not read the clock
    for timings. Progress callbacks run on the caller's worker thread. The video
    selector already forwards these messages to Tkinter through its event queue.
    """
    mode, seed, fit_samples, gmm_tol = fv_settings(mode, random_state, fit_samples, gmm_tol)
    workers, posterior = fv_runtime_settings(mode, workers, posterior)
    if profile is not None:
        profile.metadata.update(frame_workers=workers, posterior=posterior,
                                parallel_pca_batches=0, serial_pca_fallback_batches=0)
    options = dict(progress=progress, profile=profile, mode=mode, seed=seed,
                   fit_samples=fit_samples, gmm_tol=gmm_tol, posterior=posterior)
    if workers == 1:
        return _extract_frames(input_path, outputPath, executor=None, **options)
    with threadpool_limits(limits=1), ExitStack() as stack:
        with measure(profile, 'pca_worker_pool_setup'):
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
        return _extract_frames(input_path, outputPath, executor=executor, **options)


def _extract_frames(input_path, outputPath, *, progress, profile, mode, seed, fit_samples, gmm_tol,
                    executor, posterior):
    optimized = mode == 'optimized'
    rng = None if seed is None else np.random.RandomState(seed)
    pca_options = {} if rng is None else {'random_state': rng}
    gmm_options = {} if rng is None else {'random_state': rng, 'sample_rng': np.random.default_rng(seed)}
    if fit_samples:
        gmm_options['fit_sample'] = True
    if gmm_tol != DEFAULT_GMM_TOL:
        gmm_options['tol'] = gmm_tol
    name = Path(input_path).stem
    report = progress or (lambda message: None)
    report('Loading trajectories')
    with measure(profile, 'trajectory_loading'):
        tracks = np.load(input_path)
    if profile is not None:
        profile.matrix('tracks', tracks)
        profile.metadata.update(trajectory_count=len(tracks), gmm_components=K,
                                pca_dimensions=PCA_DIM, gmm_sample_limit=SAMPLE_SIZE,
                                mode=mode, seed=seed, fit_samples=fit_samples, gmm_tol=gmm_tol)
    report('Preparing descriptors')
    with measure(profile, 'descriptor_assembly'):
        all_desc = read_descriptors(tracks, 'all', 'nothing')
        ends = tracks['frame_num'].reshape(len(tracks))
        if optimized:
            ends = ends.copy()
            del tracks
    print("Descriptor Shape:", all_desc.shape)
    if profile is not None:
        profile.matrix('descriptors', all_desc)
    training = all_desc
    fit_mask = fit_rng = None
    if 0 < fit_samples < len(all_desc):
        fit_rng = np.random.default_rng(seed)
        with measure(profile, 'global_fit_sampling'):
            fit_mask = _temporal_fit_mask(ends, fit_samples, fit_rng)
            training = all_desc[fit_mask]
    report('Fitting PCA')
    red_desc = (compute_PCA(training, **pca_options) if profile is None else
                compute_PCA(training, profile=profile, stage='global_pca', **pca_options))
    del training
    if profile is not None:
        profile.matrix('global_pca_output', red_desc)
    report('Fitting GMM')
    if profile is not None:
        gmm_options['profile'] = profile
    gmm = compute_GMM(red_desc, **gmm_options)
    if optimized:
        del red_desc
    predict = _DiagonalPosterior(gmm) if posterior == 'numpy' else None

    report('Mapping trajectories to frames')
    # Keep insertion order, inclusive 15-frame membership and trajectory order.
    with measure(profile, 'trajectory_frame_mapping'):
        dict_desc = _frame_indices(ends, optimized)
    total = len(dict_desc)
    if profile is not None:
        counts = [_index_count(indices) for indices in dict_desc.values()]
        profile.metadata.update(frame_count=total,
                                frame_descriptor_count=sum(counts),
                                frame_descriptors_min=min(counts, default=0),
                                frame_descriptors_max=max(counts, default=0),
                                frame_pca_count=sum(count > 63 for count in counts),
                                zero_fallback_count=sum(count <= 63 for count in counts))
    report(f'Computing framewise Fisher Vectors: 0/{total}')
    last_report = perf_counter() if progress is not None else None

    def report_frames(completed):
        nonlocal last_report
        if progress is not None:
            now = perf_counter()
            if completed == total or now - last_report >= 0.5:
                report(f'Computing framewise Fisher Vectors: {completed}/{total}')
                last_report = now

    if optimized:
        np_frame_descriptors = _batched_fisher_vectors(
            all_desc, dict_desc, gmm, profile=profile, random_state=rng, report_frames=report_frames,
            fit_mask=fit_mask, fit_rng=fit_rng, executor=executor, posterior=predict)
        del all_desc
    else:
        frame_descriptors = []
        for completed, indices in enumerate(dict_desc.values(), 1):
            with measure(profile, 'frame_descriptor_preparation'):
                combine = all_desc[indices]
            frame_descriptors.append(_frame_fisher_vector(combine, gmm, profile=profile, **pca_options))
            report_frames(completed)
        with measure(profile, 'frame_vector_assembly'):
            np_frame_descriptors = np.array(frame_descriptors)
    if profile is not None:
        profile.matrix('frame_fisher_vectors', np_frame_descriptors)
    report('Finalizing features: fitting final PCA')
    np_frame_desc_red = (compute_PCA(np_frame_descriptors, **pca_options) if profile is None else
                         compute_PCA(np_frame_descriptors, profile=profile, stage='final_pca', **pca_options))
    if profile is not None:
        profile.matrix('features', np_frame_desc_red)
    with measure(profile, 'feature_text_serialization'):
        np.savetxt(os.path.join(outputPath, name + '-frame_fv.txt'), np_frame_desc_red)
    return np_frame_desc_red
