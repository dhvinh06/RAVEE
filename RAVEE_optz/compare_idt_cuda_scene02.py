"""Compare CPU W=1 and CUDA IDT on the same 300 grayscale Solve4X Scene 02 frames."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np


FRAME_COUNT = 300
TRACK_LENGTH = 15
SCRIPT = 'compare_idt_cuda_scene02.py'
DESCRIPTORS = ('hog', 'hof', 'mbh_x', 'mbh_y')
CUDA_COUNTS = re.compile(r'IDT CUDA optical flow: uploads=(\d+) raw_pairs=(\d+) warped_pairs=(\d+)')
CUDA_PROFILE_MARKER = 'IDT CUDA profile: '


def container_peak_ram():
    for path in ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes'):
        try:
            return int(Path(path).read_text())
        except (OSError, ValueError):
            continue
    return None


def run_in_container(backend, work):
    import cv2
    import densetrack

    sample = np.load(work / 'sample.npy', mmap_mode='r', allow_pickle=False)
    if sample.shape != (FRAME_COUNT, 480, 640) or sample.dtype != np.uint8:
        raise ValueError(f'Expected {FRAME_COUNT} grayscale frames at 640x480')
    started = time.perf_counter()
    tracks = densetrack.densetrack(sample, track_length=TRACK_LENGTH, adjust_camera=True)
    elapsed = time.perf_counter() - started
    np.save(work / f'{backend}.npy', tracks, allow_pickle=False)
    stats = {'densetrack_seconds': elapsed, 'opencv': cv2.__version__,
             'container_peak_ram_bytes': container_peak_ram()}
    (work / f'{backend}.json').write_text(json.dumps(stats), encoding='utf-8')
    print(f'{backend}: {len(tracks)} tracks in {elapsed:.2f} seconds', flush=True)


@contextmanager
def sample_vram(path):
    with path.open('w', encoding='utf-8') as log:
        try:
            process = subprocess.Popen([
                'nvidia-smi', '-i', '0', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits', '-lms', '200',
            ], stdout=log, stderr=subprocess.DEVNULL)
        except OSError:
            yield
            return
        try:
            yield
        finally:
            process.terminate()
            process.wait()


def run_backend(work, image, backend, run_logged, *, profile=False):
    argv = ['docker', 'run', '--rm', '--pull', 'never', '--memory=6g', '--memory-swap=6g',
            '-e', 'RAVEE_IDT_WORKERS=1', '-e', f'RAVEE_IDT_BACKEND={backend}']
    if backend == 'cuda':
        argv.extend(['--gpus', 'all'])
        if profile:
            argv.extend(['-e', 'RAVEE_IDT_CUDA_PROFILE=1'])
    argv.extend(['--mount', f'type=bind,source={work},target=/check',
                 '--entrypoint', '/bin/sh', image, '-ec',
                 'export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
                 'export PYTHONPATH="/densetrack${PYTHONPATH:+:$PYTHONPATH}"; '
                 f'exec python3 -u /check/{SCRIPT} --inside-container {backend}'])
    lines = run_logged(argv, work / f'{backend}.log')
    result = json.loads((work / f'{backend}.json').read_text(encoding='utf-8'))
    if backend == 'cuda':
        counts = [tuple(map(int, match.groups())) for line in lines if (match := CUDA_COUNTS.search(line))]
        if counts != [(FRAME_COUNT, FRAME_COUNT - 1, FRAME_COUNT - 1)]:
            raise RuntimeError(f'CUDA flow did not report the expected uploads and two passes: {counts}')
        result['frame_uploads'], result['raw_flow_pairs'], result['warped_flow_pairs'] = counts[0]
        if profile:
            profiles = [json.loads(line[len(CUDA_PROFILE_MARKER):]) for line in lines
                        if line.startswith(CUDA_PROFILE_MARKER)]
            if (len(profiles) != 1 or profiles[0]['measured_frames'] <= 0
                    or profiles[0]['measured_frames'] + profiles[0]['warmup_frames'] != FRAME_COUNT):
                raise RuntimeError('Missing or incomplete CUDA event profile')
            result['profile'] = profiles[0]
    return result


def end_frames(tracks):
    frames = tracks['frame_num'].reshape(-1)
    if frames.min() < TRACK_LENGTH or frames.max() >= FRAME_COUNT:
        raise ValueError('Unexpected trajectory end-frame indices in the short clip')
    return np.unique(frames)


def compare_descriptor(cpu, cuda):
    cpu_mean = cpu.mean(axis=0, dtype=np.float64)
    cuda_mean = cuda.mean(axis=0, dtype=np.float64)
    norm_product = np.linalg.norm(cpu_mean) * np.linalg.norm(cuda_mean)
    low = float(min(cpu.min(), cuda.min()))
    high = float(max(cpu.max(), cuda.max()))
    bins = np.linspace(low, high if high > low else low + 1, 33)
    cpu_hist = np.histogram(cpu, bins=bins)[0] / cpu.size
    cuda_hist = np.histogram(cuda, bins=bins)[0] / cuda.size
    return {
        'cpu_p05_p50_p95': np.percentile(cpu, [5, 50, 95]).tolist(),
        'cuda_p05_p50_p95': np.percentile(cuda, [5, 50, 95]).tolist(),
        'mean_vector_cosine': float(np.dot(cpu_mean, cuda_mean) / norm_product) if norm_product else None,
        'mean_bin_absolute_delta': float(np.mean(np.abs(cpu_mean - cuda_mean))),
        'coefficient_histogram_total_variation': float(np.abs(cpu_hist - cuda_hist).sum() / 2),
    }


def compare_outputs(work, validate_tracks):
    for backend in ('cpu', 'cuda'):
        validate_tracks(work / f'{backend}.npy', FRAME_COUNT)
    cpu = np.load(work / 'cpu.npy', mmap_mode='r', allow_pickle=False)
    cuda = np.load(work / 'cuda.npy', mmap_mode='r', allow_pickle=False)
    if cpu.dtype != cuda.dtype:
        raise ValueError(f'Trajectory dtype mismatch. CPU={cpu.dtype.descr}, CUDA={cuda.dtype.descr}')
    cpu_frames, cuda_frames = end_frames(cpu), end_frames(cuda)
    return {
        'dtype': cpu.dtype.descr,
        'cpu_tracks': len(cpu), 'cuda_tracks': len(cuda),
        'cuda_over_cpu_track_count': len(cuda) / len(cpu),
        'cpu_end_frame_range': [int(cpu_frames[0]), int(cpu_frames[-1])],
        'cuda_end_frame_range': [int(cuda_frames[0]), int(cuda_frames[-1])],
        'common_end_frames': len(np.intersect1d(cpu_frames, cuda_frames)),
        'cpu_distinct_end_frames': len(cpu_frames), 'cuda_distinct_end_frames': len(cuda_frames),
        'descriptors': {name: compare_descriptor(cpu[name], cuda[name]) for name in DESCRIPTORS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--cpu-image', help='Existing CPU W=1 IDT image tag or ID')
    parser.add_argument('--profile-cuda', action='store_true',
                        help='Profile CUDA on the same clip without rerunning the CPU baseline')
    parser.add_argument('--inside-container', choices=('cpu', 'cuda'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside_container:
        run_in_container(args.inside_container, Path('/check'))
        return 0

    from profile_idt_scene02 import CPU_IMAGE, SCENE_PATH, read_sample, run_logged
    from toolbox.pipeline_cache import build_signature, ensure_image, validate_tracks

    source = args.video or Path(tempfile.gettempdir()) / SCENE_PATH
    cpu_image = args.cpu_image or CPU_IMAGE
    if not args.profile_cuda:
        subprocess.run(['docker', 'image', 'inspect', cpu_image], stdout=subprocess.DEVNULL, check=True)
    work = Path(tempfile.mkdtemp(prefix='ravee-idt-cuda-300-'))
    print(f'Output directory: {work}', flush=True)
    sample, start = read_sample(source, FRAME_COUNT)
    np.save(work / 'sample.npy', sample, allow_pickle=False)
    shutil.copy2(Path(__file__), work / SCRIPT)

    idt = Path(__file__).resolve().parent / 'IDT'
    cuda_image = ensure_image(idt, build_signature(idt, backend='cuda'), backend='cuda')
    vram_log = work / 'vram_mib.txt'
    with sample_vram(vram_log):
        cuda = run_backend(work, cuda_image, 'cuda', run_logged, profile=args.profile_cuda)
    samples = [int(line) for line in vram_log.read_text().splitlines() if line.strip().isdecimal()]
    cuda['device_peak_sampled_vram_mib'] = max(samples, default=0) or None

    report = dict(source=str(source.resolve()), sample_start_frame=start, sample_frames=FRAME_COUNT,
                  cuda_image=cuda_image, cuda=cuda)
    if args.profile_cuda:
        validate_tracks(work / 'cuda.npy', FRAME_COUNT)
        tracks = np.load(work / 'cuda.npy', mmap_mode='r', allow_pickle=False)
        report['cuda_tracks'] = len(tracks)
        out_dir = idt.parent / 'temp' / 'idt_cuda_profiles'
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f'{work.name}.json'
    else:
        cpu = run_backend(work, cpu_image, 'cpu', run_logged)
        report.update(compare_outputs(work, validate_tracks), cpu_image=cpu_image, cpu=cpu,
                      cpu_over_cuda_densetrack_time=cpu['densetrack_seconds'] / cuda['densetrack_seconds'])
        report_path = work / 'comparison.json'
    output = json.dumps(report, indent=2, allow_nan=False) + '\n'
    report_path.write_text(output, encoding='utf-8')
    if args.profile_cuda:
        print(f'CUDA profile saved to {report_path}', flush=True)
    else:
        print(output, end='', flush=True)
    vram = cuda['device_peak_sampled_vram_mib']
    if vram is not None and vram >= 3.5 * 1024:
        print('Sampled GPU memory use reached the 3.5 GiB limit.', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(str(error))
