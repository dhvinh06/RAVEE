import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
import cv2
import re
import time

import densetrack

# Choose which descriptors you would like to run. If all are set to False,
# raw IDT features will be outputted.
HOG_FISHER_VECTOR = False
HOF_FISHER_VECTOR = False
MBH_FISHER_VECTOR = True

# RAVEE temporal segmentation needs raw trajectories with frame indices.
# Keep standalone HOG/HOF/MBH CSV behavior unless the pipeline requests raw mode.
if os.environ.get('RAVEE_RAW_ONLY') == '1':
    HOG_FISHER_VECTOR = HOF_FISHER_VECTOR = MBH_FISHER_VECTOR = False

DATA_DIRECTORY = 'data'  # the directory of the input video files
TARGET_DIRECTORY = 'features'  # targets will be put in this directory

K = 128  # GMM components for the Fisher Vector


def read_video(file):
    """
    Reads the frames from a video file.

    :param file: a filename in the data directory or an absolute path.
    :type file: String

    :return: the gray video as a numpy array.
    :rtype: array_like, shape (D, H, W) where D is the number of frames,
            H and W are the resolution.
    """
    vidcap = cv2.VideoCapture(os.path.join(DATA_DIRECTORY, file))

    video = []
    success, image = vidcap.read()
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        video.append(gray)
        success, image = vidcap.read()

    return np.array(video)


def fisher_vector(xx, gmm):
    """
    Computes the fisher vector on a set of descriptors and GMM.

    :param xx: the descriptor (e.g. HOG/HOF/MBH) for each video.
    :type xx: array_like, shape (N, D) where N is the number of descriptors
              and D is the dimension of each descriptor (e.g. 96 for HOG).

    :param gmm: Gauassian mixture model of the descriptors.
    :type gmm: instance of sklearn mixture.GMM object

    :return: Fisher vector of the given descriptor.
    :rtype: array_like, shape (K + 2 * D * K, ) where K is the number of GMM
            components and D is the dimension of each descriptor.

    This function was taken from this GitHub gist: https://gist.github.com/danoneata/9927923
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

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


def read_hog(descriptors):
    """
    Extracts the HOG descriptors from the improved dense trajectory features.

    :param descriptors: the improved dense trajectories returned by
                        densetrack.densetrack method.
    :type descriptors: array_like, shape (N, ) where N is the number of
                       trajectories. Value of each element is a np.void.

    :return: HOG descriptors.
    :rtype: array_like, shape (N, ) where N is the number of trajectories.
    """
    return np.array([descriptors[i][-5] for i in
                    range(descriptors.shape[0])])


def read_hof(descriptors):
    """
    Extracts the HOF descriptors from the improved dense trajectory features.

    :param descriptors: the improved dense trajectories returned by
                        densetrack.densetrack method.
    :type descriptors: array_like, shape (N, ) where N is the number of
                       trajectories. Value of each element is a np.void.

    :return: HOF descriptors.
    :rtype: array_like, shape (N, ) where N is the number of trajectories.
    """
    return np.array([descriptors[i][-4] for i in
                     range(descriptors.shape[0])])


def read_mbh(descriptors):
    """
    Extracts the MBH descriptors from the improved dense trajectory features.

    :param descriptors: the improved dense trajectories returned by
                        densetrack.densetrack method.
    :type descriptors: array_like, shape (N, ) where N is the number of
                       trajectories. Value of each element is a np.void.

    :return: MBH descriptors.
    :rtype: array_like, shape (N, ) where N is the number of trajectories.
    """
    mbh_x_descriptors = np.array([descriptors[i][-3] for i in
                                  range(descriptors.shape[0])])
    mbh_y_descriptors = np.array([descriptors[i][-2] for i in
                                  range(descriptors.shape[0])])

    return mbh_x_descriptors, mbh_y_descriptors


def fisher_descriptor(descriptor):
    """
    Compute the fisher vector based on a set of descriptors.

    As described in the original Improved Dense Trajectory papers, the dimension
    of the descriptors is halved with PCA. The default value of the GMM
    components, K (defined at the top of the script), is 256.

    :param descriptor: the descriptor (e.g. HOG/HOF/MBH) for each video.
    :type descriptor: array_like, shape (N, D) where N is the number of
                      descriptors and D is the dimension of each descriptor
                      (e.g. 96 for HOG).

    :return: fisher vector of the descriptors.
    :rtype: array_like, shape (K + 2 * K * D/2) since the dimension is halved
            with a PCA.
    """
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture as GMM

    descriptor_pca = PCA(n_components=int(len(descriptor[0])/2))\
        .fit(descriptor).transform(descriptor)
    gmm = GMM(n_components=K, covariance_type='diag').fit(descriptor_pca)
    return fisher_vector(descriptor_pca, gmm)


def add_row_to_table(data_frame, name, features):
    """
    Adds a row to the data frame.

    If empty, creates the data frame. Otherwise, add name of the video and
    corresponding descriptor's features.

    :param data_frame: The data frame of a descriptor (HOG/HOF/MBH).
    :type data_frame: DataFrame

    :param name: the name of the video.
    :type name: String

    :param features: the features of a descriptor (HOG/HOF/MBH)
    :type features: array_like, shape (D, ) where D is the dimension of the
                    descriptor.

    :return: DataFrame with the new row appended.
    :rtype: DataFrame
    """
    import pandas as pd

    if data_frame.empty:
        data_frame = pd.DataFrame(columns=['name'] + [str(i) for i in
                                                      range(len(features))])
    data_frame = data_frame.append(pd.Series(
        [name] + list(features), index=data_frame.columns), ignore_index=True)

    return data_frame


def save_as_csv(data_frame, path):
    """
    Creates a CSV file from a data frame and does nothing if frame is empty.

    :param data_frame: The final data frame of a descriptor (HOG/HOG/MBH) for
                       all videos.
    :type data_frame: DataFrame

    :param path: The path to write the DataFrame to.
    :type path: String
    """
    if not data_frame.empty:
        temporary = path + '.partial'
        data_frame.to_csv(temporary)
        os.replace(temporary, path)

def sorted_alphanumeric(data):
    """
    Sorts alphanumerically.

    :param data: data to be sorted e.g. a list
    :return: sorted data alphanumerically e.g. "User_2" before "User_10"
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def track_video(path, filename, *, profile=False):
    read_started = time.perf_counter()
    video = read_video(path)
    print('------------------------------------------')
    print(f'Running: {filename} of shape {video.shape}')
    if not profile:
        return densetrack.densetrack(video, adjust_camera=True)
    report_timing('decode_video', time.perf_counter() - read_started)
    started, cpu_started = time.perf_counter(), time.process_time()
    tracks = densetrack.densetrack(video, adjust_camera=True)
    report_timing('sequential', time.perf_counter() - started,
                  cpu_s=f'{time.process_time() - cpu_started:.6f}')
    return tracks


def raw_worker_count():
    if os.environ.get('RAVEE_RAW_ONLY') != '1':
        return 1
    try:
        workers = int(os.environ.get('RAVEE_IDT_WORKERS', '1'))
        if 1 <= workers <= 6:
            return workers
    except ValueError:
        pass
    raise ValueError('RAVEE_IDT_WORKERS must be an integer from 1 to 6')


def initialize_chunk_worker():
    cv2.setNumThreads(1)


def report_timing(stage, seconds, **details):
    fields = ' '.join(f'{key}={value}' for key, value in details.items())
    print(f'IDT timing stage={stage} wall_s={seconds:.6f} {fields}'.rstrip(), flush=True)


def extract_chunk(video_path, output_path, start, end):
    started, cpu_started = time.perf_counter(), time.process_time()
    video = np.load(video_path, mmap_mode='r', allow_pickle=False)
    load_start = max(0, start - 30)
    tracking_started = time.perf_counter()
    tracks = densetrack.densetrack(video[load_start:end], adjust_camera=True)
    tracking_seconds = time.perf_counter() - tracking_started
    frames = tracks['frame_num'].reshape(-1)
    frames += load_start
    # DenseTrack emits completed trajectories in frame order.
    first, last = np.searchsorted(frames, (start, end))
    kept = tracks[first:last]
    kept['t_pos'].reshape(-1)[:] = np.clip(
        (frames[first:last] - 7.5) / len(video), 0, 0.999)
    np.save(output_path, kept, allow_pickle=False)
    report_timing('chunk', time.perf_counter() - started, start=start, end=end,
                  load_start=load_start, tracks=len(kept), tracking_s=f'{tracking_seconds:.6f}',
                  cpu_s=f'{time.process_time() - cpu_started:.6f}')
    return len(kept), kept.dtype


def extract_parallel_video(local_input, local_output, workers):
    started = time.perf_counter()
    video = read_video(local_input.resolve())
    report_timing('decode_video', time.perf_counter() - started)
    if video.ndim != 3 or len(video) == 0:
        raise ValueError(f'No decoded frames in {local_input.name}')
    frame_count = len(video)
    video_path = local_input.parent / 'video_gray.npy'
    started = time.perf_counter()
    np.save(video_path, video, allow_pickle=False)
    del video
    report_timing('write_gray_mmap', time.perf_counter() - started)

    chunk_count = min(workers, frame_count)
    chunks = [local_input.parent / f'chunk_{index}.npy' for index in range(chunk_count)]
    # Spawned Python processes inherit these before importing NumPy/OpenCV.
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    print(f'Running: {local_input.name} with {chunk_count} IDT workers, {frame_count} frames')
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=chunk_count, mp_context=get_context('spawn'),
                             initializer=initialize_chunk_worker) as executor:
        jobs = [executor.submit(extract_chunk, video_path, path,
                                idx * frame_count // chunk_count,
                                (idx + 1) * frame_count // chunk_count)
                for idx, path in enumerate(chunks)]
        results = [job.result() for job in jobs]
    report_timing('chunk_pool', time.perf_counter() - started)
    video_path.unlink()

    started = time.perf_counter()
    dtype = results[0][1]
    merged = np.lib.format.open_memmap(
        local_output, mode='w+', dtype=dtype, shape=(sum(count for count, _ in results),))
    cursor = 0
    for path, (count, _) in zip(chunks, results):
        chunk = np.load(path, mmap_mode='r', allow_pickle=False)
        merged[cursor:cursor + count] = chunk
        cursor += count
        del chunk
        path.unlink()
    merged.flush()
    del merged
    report_timing('merge_tracks', time.perf_counter() - started)


def extract_raw_video(file, destination):
    from pathlib import Path
    import shutil
    import tempfile
    from uuid import uuid4

    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix='ravee-idt-') as scratch:
        local_input = Path(scratch) / file
        local_output = Path(scratch) / destination.name
        started = time.perf_counter()
        shutil.copyfile(Path(DATA_DIRECTORY) / file, local_input)
        report_timing('stage_input', time.perf_counter() - started)
        workers = raw_worker_count()
        if workers > 1:
            extract_parallel_video(local_input, local_output, workers)
        else:
            np.save(local_output, track_video(local_input.resolve(), file, profile=True))

        # The partial file must be on the destination filesystem for rename.
        pending = destination.with_name(f'{destination.name}.{uuid4().hex}.partial')
        output = pending.open('xb')
        started = time.perf_counter()
        try:
            with output, local_output.open('rb') as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            os.replace(pending, destination)
            report_timing('publish_tracks', time.perf_counter() - started)
        finally:
            pending.unlink(missing_ok=True)


def report_container_peak_memory():
    for filename in ('/sys/fs/cgroup/memory.peak',
                     '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        try:
            with open(filename) as memory:
                peak = int(memory.read())
        except (OSError, ValueError):
            continue
        print(f'IDT container peak memory bytes: {peak}', flush=True)
        break


def main():
    if not os.listdir(DATA_DIRECTORY):
        print('No input video files are present. Please put your files in the '
              '"data" directory, rebuild the image and run the container.')
        return

    files = sorted_alphanumeric(os.listdir(DATA_DIRECTORY))
    if not (HOG_FISHER_VECTOR or HOF_FISHER_VECTOR or MBH_FISHER_VECTOR):
        for file in files:
            name = os.path.splitext(file)[0]
            extract_raw_video(file, os.path.join(TARGET_DIRECTORY, name + '-trajectory.npy'))
            print(f'Completed {file}')
        if os.environ.get('RAVEE_RAW_ONLY') == '1':
            report_container_peak_memory()
        return

    export_csv_features(files)


def export_csv_features(files):
    import pandas as pd

    hog_df, hof_df, mbh_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i, file in enumerate(files, 1):
        name = os.path.splitext(file)[0]
        tracks = track_video(file, file)
        if HOG_FISHER_VECTOR:
            hog_descriptors = read_hog(tracks)
            hog_fv = fisher_descriptor(hog_descriptors)
            hog_df = add_row_to_table(hog_df, name, hog_fv)
        if HOF_FISHER_VECTOR:
            hof_descriptors = read_hof(tracks)
            hof_fv = fisher_descriptor(hof_descriptors)
            hof_df = add_row_to_table(hof_df, name, hof_fv)
        if MBH_FISHER_VECTOR:
            mbh_x_descriptors, mbh_y_descriptors = read_mbh(tracks)
            mbh_x_fv = fisher_descriptor(mbh_x_descriptors)
            mbh_y_fv = fisher_descriptor(mbh_y_descriptors)
            mbh_fv = np.concatenate((mbh_x_fv, mbh_y_fv))
            mbh_df = add_row_to_table(mbh_df, name, mbh_fv)
        del tracks

        print(f'Completed {file}')
        if i % 10 == 0:
            print(f'{i} files were completed.')
        save_as_csv(hog_df, 'features/hog_features.csv')
        save_as_csv(hof_df, 'features/hof_features.csv')
        save_as_csv(mbh_df, 'features/mbh_features.csv')


if __name__ == '__main__':
    main()
