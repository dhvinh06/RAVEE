import os
from pathlib import Path
from time import perf_counter
from toolbox.fv_profiling import measure
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA

# Number of clusters for Gaussian mixture - k∈{1,...,K} where each Gaussian k has a μ (defining center), cov (definies
# width) and p(mixing probability)
K = 64
# Number of dimensions to reduce to w PCA
PCA_DIM = 64
# Number of samples to take to fit the GMM
SAMPLE_SIZE = 200000


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
    N = xx.shape[0]

    # Compute posterior probabilities.
    with measure(profile, 'gmm_posterior'):
        Q = gmm.predict_proba(xx)  # NxK

    with measure(profile, 'fisher_statistics'):
        # Compute the sufficient statistics of descriptors.
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_xx = np.dot(Q.T, xx) / N
        Q_xx_2 = np.dot(Q.T, xx ** 2) / N

        # Compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum.squeeze() - gmm.weights_
        d_mu = Q_xx - Q_sum * gmm.means_
        d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_
            + 2 * Q_xx * gmm.means_)

        # Merge derivatives into a vector.
        return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


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


def _frame_fisher_vector(combine, gmm, *, profile=None):
    # Keep the original local fitting scope and the <=63 descriptor fallback.
    if combine.shape[0] > 63:
        combine = (compute_PCA(combine) if profile is None else
                   compute_PCA(combine, profile=profile, stage='frame_pca'))
    else:
        combine = np.zeros((64, 64))
        print("Warning: FV derived from np.zeros - if done to often clustering is random")
    return (fisher_vector(combine, gmm) if profile is None else
            fisher_vector(combine, gmm, profile=profile))


def compute_PCA(descriptor,dim_PCA=PCA_DIM, *, profile=None, stage='pca'):
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
    with measure(profile, stage + '_fit'):
        pca = PCA(n_components=int(dim_PCA)).fit(descriptor)
    with measure(profile, stage + '_transform'):
        return pca.transform(descriptor)

def compute_GMM(descriptor, *, profile=None):
    '''
    Computes GMM for K components from features given by descriptor

    :param descriptor: The descriptors to fit the GMM with
    :type descriptor: array_like, shape (N, D) where N is the number of
                      descriptors and D is the dimension of each descriptor
                      (e.g. 96 for HOG).
    :return:  GMM
    '''
    #Get a random sample of SAMPLE_SIZE for creation of GMM
    if descriptor.shape[0] > SAMPLE_SIZE:
        rng = np.random.default_rng()
        sample = rng.choice(descriptor,SAMPLE_SIZE)
    else:
        sample = descriptor
    if profile is not None:
        profile.matrix('gmm_sample', sample)
    with measure(profile, 'gmm_fit'):
        gmm = GMM(n_components=K, covariance_type='diag').fit(sample)
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

def extract_framewise_features(input_path, outputPath, *, progress=None, profile=None):
    """Extract unchanged framewise FVs, optionally reporting progress and timings.

    ``profile`` accepts an FVProfile; omitted profiling does not read the clock
    for timings. Progress callbacks run on the caller's worker thread. The video
    selector already forwards these messages to Tkinter through its event queue.
    """
    name = Path(input_path).stem
    report = progress or (lambda message: None)
    report('Loading trajectories')
    with measure(profile, 'trajectory_loading'):
        tracks = np.load(input_path)
    if profile is not None:
        profile.matrix('tracks', tracks)
        profile.metadata.update(trajectory_count=len(tracks), gmm_components=K,
                                pca_dimensions=PCA_DIM, gmm_sample_limit=SAMPLE_SIZE)
    report('Preparing descriptors')
    with measure(profile, 'descriptor_assembly'):
        all_desc = read_descriptors(tracks, 'all', 'nothing')
    print("Descriptor Shape:", all_desc.shape)
    if profile is not None:
        profile.matrix('descriptors', all_desc)
    report('Fitting PCA')
    red_desc = (compute_PCA(all_desc) if profile is None else
                compute_PCA(all_desc, profile=profile, stage='global_pca'))
    if profile is not None:
        profile.matrix('global_pca_output', red_desc)
    report('Fitting GMM')
    gmm = (compute_GMM(red_desc) if profile is None else
           compute_GMM(red_desc, profile=profile))

    report('Mapping trajectories to frames')
    # Keep insertion order, inclusive 15-frame membership and trajectory order.
    with measure(profile, 'trajectory_frame_mapping'):
        dict_desc = {}
        for index in range(tracks.shape[0]):
            end_frame = int(np.asarray(tracks[index]['frame_num']).item())
            frame_range = range(int(end_frame - 14), int(end_frame + 1))
            for j in frame_range:
                if j in dict_desc:
                    dict_desc[j].append(index)
                else:
                    dict_desc[j] = [index]
    total = len(dict_desc)
    if profile is not None:
        counts = [len(indices) for indices in dict_desc.values()]
        profile.metadata.update(frame_count=total,
                                frame_descriptor_count=sum(counts),
                                frame_descriptors_min=min(counts, default=0),
                                frame_descriptors_max=max(counts, default=0),
                                frame_pca_count=sum(count > 63 for count in counts),
                                zero_fallback_count=sum(count <= 63 for count in counts))
    report(f'Computing framewise Fisher Vectors: 0/{total}')
    last_report = perf_counter() if progress is not None else None
    frame_descriptors = []
    for completed, key in enumerate(dict_desc, 1):
        with measure(profile, 'frame_descriptor_preparation'):
            # Index RAW descriptors, never the globally PCA-transformed matrix.
            # Advanced indexing preserves duplicates/order and copies one frame.
            combine = all_desc[dict_desc[key]]
        frame_descriptors.append(_frame_fisher_vector(combine, gmm, profile=profile))
        if progress is not None:
            now = perf_counter()
            if completed == total or now - last_report >= 0.5:
                report(f'Computing framewise Fisher Vectors: {completed}/{total}')
                last_report = now
    with measure(profile, 'frame_vector_assembly'):
        np_frame_descriptors = np.array(frame_descriptors)
    if profile is not None:
        profile.matrix('frame_fisher_vectors', np_frame_descriptors)
    report('Finalizing features: fitting final PCA')
    np_frame_desc_red = (compute_PCA(np_frame_descriptors) if profile is None else
                         compute_PCA(np_frame_descriptors, profile=profile, stage='final_pca'))
    if profile is not None:
        profile.matrix('features', np_frame_desc_red)
    with measure(profile, 'feature_text_serialization'):
        np.savetxt(os.path.join(outputPath, name + '-frame_fv.txt'), np_frame_desc_red)
    del dict_desc
    del tracks
    return np_frame_desc_red
