"""Sweep CUDA IDT on Solve4X and check the five phase 4A acceptance criteria."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from statistics import mean, stdev
import subprocess
import sys
import tempfile
from threading import Event, Thread
import time
from unittest.mock import patch
from uuid import uuid4

PROJECT = Path(__file__).resolve().parent
BASELINE = PROJECT / 'idt-phase3-20261126-205155.bak'
TEMP = Path(tempfile.gettempdir())
SCENES = {'scene01': (5882, 680.68), 'scene02': (3310, 413.89)}
SCOPES = (*SCENES, 'combined')
METRICS = ('raw_ari', 'oracle_ari', 'oracle_accuracy')
SEEDS = tuple(range(10))
VRAM_LIMIT_MIB = 3584
RAM_LIMIT_BYTES = int(6.5 * 1024**3)

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from toolbox import pipeline_cache as cache


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    pending = path.with_suffix('.partial')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gpu_query(query):
    return subprocess.check_output([
        'nvidia-smi', '-i', '0', f'--query-gpu={query}', '--format=csv,noheader,nounits',
    ], text=True, timeout=5).strip()


def power_status():
    if sys.platform != 'win32':
        return {}
    status = (ctypes.c_ubyte * 12)()
    ac = status[0] if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)) else None
    scheme = subprocess.check_output(['powercfg', '/getactivescheme'], text=True, timeout=5)
    return {'ac_connected': {0: False, 1: True}.get(ac), 'power_scheme': scheme.strip()}


def monitor_docker_stats(name, path, finished, started):
    with path.open('w', encoding='utf-8') as out:
        while not finished.is_set():
            try:
                result = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{json .}}', name],
                                        capture_output=True, text=True, timeout=4)
                if result.returncode == 0 and result.stdout.strip():
                    sample = json.loads(result.stdout)
                    if sample.get('MemUsage'):
                        sample['elapsed_seconds'] = time.perf_counter() - started
                        print(json.dumps(sample), file=out, flush=True)
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
                pass
            finished.wait(0.5)


class TrialStopped(RuntimeError):
    pass


def stop_container(name, process):
    for _ in range(4):
        if process.poll() is not None:
            return
        subprocess.run(['docker', 'kill', name], capture_output=True, timeout=5, check=False)
        try:
            process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue
    process.kill()
    process.wait()
    raise RuntimeError(f'Could not verify Docker cleanup for {name}')


def measure(prepared, output, image, workers, folder, *, time_limit=None):
    folder.mkdir(parents=True, exist_ok=True)
    name = f'ravee-idt-4a-{uuid4().hex[:12]}'
    finished = Event()
    started = time.perf_counter()
    stats_path = folder / 'docker-stats.jsonl'
    peak = None
    samples = 0
    stats = None
    result = {'workers': workers, 'status': 'complete', 'container': name,
              'scene': prepared.stem.split('_')[1], 'trajectory': str(output),
              'time_limit_seconds': time_limit}
    print(f'{folder.name} | CUDA W={workers} | {name}', flush=True)

    def run_container(command, *, check, stdout, stderr):
        nonlocal peak, samples, stats
        command[3:3] = ['--name', name, '--pull', 'never']
        stats = Thread(target=monitor_docker_stats, args=(name, stats_path, finished, started), daemon=True)
        with subprocess.Popen(command, stdout=stdout, stderr=stderr) as process:
            stats.start()
            try:
                next_progress = 60
                with (folder / 'nvidia-smi.csv').open('w', encoding='utf-8') as gpu_log:
                    print('elapsed_seconds,vram_used_mib', file=gpu_log, flush=True)
                    while process.poll() is None:
                        elapsed = time.perf_counter() - started
                        try:
                            sample = int(gpu_query('memory.used'))
                        except (ValueError, OSError, subprocess.SubprocessError) as error:
                            raise TrialStopped(f'VRAM monitoring failed: {error}') from error
                        peak = sample if peak is None else max(peak, sample)
                        samples += 1
                        print(f'{elapsed:.3f},{sample}', file=gpu_log, flush=True)
                        if sample >= VRAM_LIMIT_MIB:
                            raise TrialStopped(f'VRAM {sample} MiB reached {VRAM_LIMIT_MIB} MiB')
                        if time_limit is not None and elapsed >= time_limit and process.poll() is None:
                            raise TrialStopped(f'Elapsed longer than fastest Scene 02: {time_limit:.2f} s')
                        if elapsed >= next_progress:
                            print(f'{folder.name} | {elapsed:.0f} s | VRAM peak {peak} MiB', flush=True)
                            next_progress += 60
                        try:
                            process.wait(timeout=0.35)
                        except subprocess.TimeoutExpired:
                            continue
                if check and process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                finished.set()
                if process.poll() is None:
                    stop_container(name, process)

    log_path = folder / 'container.log'
    with log_path.open('w', encoding='utf-8') as log:
        try:
            cache.extract_idt(prepared, output, image, workers=workers, backend='cuda',
                              container_log=log, run_container=run_container)
        except TrialStopped as error:
            result.update(status='stopped', reason=str(error))
        except subprocess.CalledProcessError as error:
            result.update(status='error', reason=f'Docker exit {error.returncode}. See {log_path}')
    elapsed = time.perf_counter() - started
    if stats is not None and stats.ident is not None:
        stats.join(timeout=5)
    log_contents = log_path.read_text(encoding='utf-8', errors='replace')
    match = re.search(r'IDT container peak memory bytes: (\d+)', log_contents)
    result.update(wall_seconds=elapsed, vram_peak_mib=peak, vram_samples=samples,
                  ram_peak_bytes=int(match[1]) if match else None,
                  docker_stats_samples=len(stats_path.read_text(encoding='utf-8').splitlines())
                  if stats_path.is_file() else 0)
    if result['status'] == 'complete':
        try:
            cache.validate_tracks(output, SCENES[result['scene']][0] + 1)
            result['trajectory_sha256'] = hash_file(output)
        except (ValueError, OSError) as error:
            result.update(status='error', reason=str(error))
    print(f"{folder.name}: {result['status']} | {elapsed:.2f} s | peak VRAM {peak} MiB"
          + (f" | {result['reason']}" if 'reason' in result else ''), flush=True)
    return result


def within_resources(trial):
    return (trial['status'] == 'complete' and trial['vram_peak_mib'] is not None
            and trial['vram_peak_mib'] < VRAM_LIMIT_MIB and trial['docker_stats_samples'] > 0
            and trial['ram_peak_bytes'] is not None and trial['ram_peak_bytes'] < RAM_LIMIT_BYTES)


def run_trial(prepared, image, workers, folder, time_limit=None):
    path = folder / 'result.json'
    if path.is_file():
        previous = read_json(path)
        if (previous['status'] == 'stopped' and not previous['reason'].startswith('VRAM monitoring failed')
                and previous['time_limit_seconds'] == time_limit):
            return previous
        if previous['status'] == 'complete':
            trajectory = Path(previous['trajectory'])
            if within_resources(previous) and trajectory.is_file() and hash_file(trajectory) == previous['trajectory_sha256']:
                return previous
    output = folder / f'{prepared.stem}-trajectory.npy'
    result = measure(prepared, output, image, workers, folder, time_limit=time_limit)
    write_json(path, result)
    return result


def sweep(videos, image, output, report):
    best = None
    for workers in range(1, cache.MAX_CUDA_IDT_WORKERS + 1):
        cutoff = best['wall_seconds'] if workers >= 3 else None
        trial = run_trial(videos['scene02'], image, workers, output / f'scene02-w{workers}', cutoff)
        report['sweep'].append(trial)
        write_json(output / 'acceptance.json', report)
        if not within_resources(trial) or (best and trial['wall_seconds'] >= best['wall_seconds']):
            break
        best = trial
    if best is None:
        return None
    workers = best['workers']
    print(f"Selected CUDA W={workers} from Scene 02 ({best['wall_seconds']:.2f} s)", flush=True)
    report.update(best_workers=workers, best_runs={'scene02': best})
    write_json(output / 'acceptance.json', report)
    # The winning Scene 02 trial already processed the entire video from scratch.
    report['best_runs']['scene01'] = run_trial(videos['scene01'], image, workers,
                                               output / f'scene01-w{workers}')
    return report['best_runs']


def legacy_acceptance():
    if str(BASELINE) not in sys.path:
        sys.path.insert(0, str(BASELINE))
    import accept_idt
    return accept_idt


def check_trajectories(runs, references, *, max_track_pct, max_l2_pct):
    import numpy as np

    prior = legacy_acceptance()
    rows = {}
    for scene, (end, _) in SCENES.items():
        if runs[scene]['status'] != 'complete':
            continue
        reference, cpu = references[scene]
        observed, _ = prior.inspect(reference, cpu, None, end + 1)
        _, row = prior.inspect(Path(runs[scene]['trajectory']), runs[scene], observed, end + 1)
        actual = np.load(runs[scene]['trajectory'], mmap_mode='r', allow_pickle=False)
        row['cuda_track_count'] = len(actual)
        row['cpu_track_count'] = observed[2]
        row['format_pass'] = (row['dtype_equal'] and row['ordered'] and row['frame_ends'] == [15, end]
                              and row['unique_frame_ends'] == end - 14
                              and row['added_frame_ends'] == row['missing_frame_ends'] == 0)
        row['drift_pass'] = abs(row['track_delta_pct']) <= max_track_pct and all(
            value[key] <= max_l2_pct for value in row['descriptors'].values()
            for key in ('mean_l2_delta_pct', 'std_l2_delta_pct')
        )
        rows[scene] = row
    return rows


def evaluate_seed_scene(seed, scene, trajectory, output):
    import numpy as np

    prior = legacy_acceptance()
    load = np.load

    def load_mmap(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file) == trajectory and not args:
            kwargs.setdefault('mmap_mode', 'r')
        return load(file, *args, **kwargs)

    with patch.object(np, 'load', load_mmap):
        prior.downstream(trajectory, output, scene, SCENES[scene][0] + 1, seed, progress=False)
    return scene, seed


def check_quality(runs, output, cpu_scores, scorer_path, dataset, jobs):
    spec = importlib.util.spec_from_file_location('phase3_oracle_scoring', scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    prior = legacy_acceptance()
    identity = cache.fingerprint({scene: row['trajectory_sha256'] for scene, row in runs.items()})[:16]
    output = output / f'quality-{identity}'
    output.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for scene in SCENES:
            tasks = [(seed, scene, Path(runs[scene]['trajectory']), output / f'seed-{seed}' / scene)
                     for seed in SEEDS]
            pending = [pool.submit(evaluate_seed_scene, *task) for task in tasks
                       if not all((task[3] / name).is_file() for name in
                                  ('complete.json', 'segments.csv', 'prediction_seconds.csv'))]
            print(f'{scene} | {len(pending)} Fisher Vector + TW-FINCH runs pending', flush=True)
            for task in pending:
                complete_scene, seed = task.result()
                print(f'{complete_scene} | seed {seed} complete', flush=True)
    cuda_scores = {seed: scorer.score_run(output / f'seed-{seed}', SCENES, dataset, prior.load_seconds)
                   for seed in SEEDS}
    rows = []
    for scope in SCOPES:
        for metric in METRICS:
            cpu = [cpu_scores[seed, scope][metric] for seed in SEEDS]
            cuda = [cuda_scores[seed][scope][metric] for seed in SEEDS]
            center, sd = mean(cpu), stdev(cpu)
            score = mean(cuda)
            rows.append({'scope': scope, 'metric': metric, 'cpu_w1_mean': center, 'cpu_w1_sd': sd,
                         'cuda_mean': score, 'cuda_sd': stdev(cuda), 'lower': center - sd,
                         'upper': center + sd, 'pass': center - sd <= score <= center + sd})
    write_json(output / 'scores-by-seed.json', cuda_scores)
    with (output / 'mean-sd.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def offline_tests(output):
    codes = {}
    for name, args in {
        'discovery': ['discover', '-s', 'tests', '-t', '.', '-v'],
        'mock_docker': ['-v', 'test_idt_cuda_backend'],
    }.items():
        path = output / f'unittest-{name}.log'
        with path.open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-B', '-m', 'unittest', *args],
                                    cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=False)
        codes[name] = result.returncode
        print(f'Offline {name}: exit {result.returncode} | {path}', flush=True)
    return codes


def benchmark_inputs(args):
    initial = read_json(BASELINE / 'acceptance_run' / 'seed-0' / 'acceptance.json')
    references = {}
    videos = {}
    for scene, (end, _) in SCENES.items():
        stem = f'1_{scene}_cam2_15fps_reduced'
        videos[scene] = args.videos / scene / 'preprocessed' / f'{stem}.avi'
        info = cache.video_info(videos[scene])
        if info['frames'] != end + 1 or abs(info['fps'] - 15) > 1e-3:
            raise ValueError(f'{scene} needs {end + 1} frames at 15 FPS: {info}')
        reference = BASELINE / 'acceptance_input' / f'w1-{stem}-trajectory.npy'
        cpu = initial['scenes'][scene]['1']
        if hash_file(reference) != cpu['trajectory_sha256']:
            raise ValueError(f'CPU W=1 reference differs from B3 acceptance inputs: {reference}')
        references[scene] = (reference, cpu)
    with (BASELINE / 'multiseed_results' / 'scores_by_seed.csv').open(encoding='utf-8', newline='') as stream:
        scores = {(int(row['seed']), row['scope']): {metric: float(row[metric]) for metric in METRICS}
                  for row in csv.DictReader(stream) if int(row['workers']) == 1}
    if set(scores) != {(seed, scope) for seed in SEEDS for scope in SCOPES}:
        raise ValueError('CPU W=1 reference must include 10 seeds for both scenes and combined')
    return videos, references, scores


def existing_image():
    signature = cache.build_signature(PROJECT / 'IDT', backend='cuda')
    result = subprocess.run(['docker', 'image', 'inspect', '--format',
                             '{{index .Config.Labels "ravee.build"}} {{.Id}}', cache.CUDA_IDT_IMAGE],
                            capture_output=True, text=True, timeout=15, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    label, _, image = result.stdout.strip().partition(' ')
    if label != signature or not image.startswith('sha256:'):
        raise RuntimeError(f'{cache.CUDA_IDT_IMAGE} is out of date. Build the CUDA image before measuring.')
    return image, signature


@contextmanager
def benchmark_lock(path):
    with path.open('a+b') as lock:
        lock.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another benchmark is using {path.parent}') from error
        yield


def run(args):
    output = args.output.resolve()
    image, signature = existing_image()
    gpu = gpu_query('name,memory.total')
    if 'GTX 1650' not in gpu or gpu.rsplit(',', 1)[-1].strip() != '4096':
        raise ValueError(f'Expected GTX 1650 with 4096 MiB VRAM, found {gpu}')
    power = power_status()
    if power.get('ac_connected') is False:
        raise RuntimeError('Connect AC power and use Performance mode before starting the cold run.')
    print(f'GPU: {gpu} | {power}', flush=True)
    videos, references, cpu_scores = benchmark_inputs(args)
    for path in (args.scorer, *(args.dataset / scene / f'{scene}_vid.json' for scene in SCENES)):
        if not path.is_file():
            raise FileNotFoundError(path)
    identity = {'image': image, 'image_signature': signature, 'gpu': gpu,
                'power': power, 'videos': {scene: hash_file(path) for scene, path in videos.items()}}
    output.mkdir(parents=True, exist_ok=True)
    with benchmark_lock(output / 'benchmark.lock'):
        manifest = output / 'inputs.json'
        if manifest.exists() and read_json(manifest) != identity:
            raise ValueError('Image, videos or power settings changed. Pass a new --output directory.')
        write_json(manifest, identity)
        with (output / 'cuda-smoke.log').open('w', encoding='utf-8') as log:
            subprocess.run(['docker', 'run', '--rm', '--pull', 'never', '--gpus', 'all',
                            '--entrypoint', '/usr/local/bin/idt-cuda-smoke', image],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        print(f'Acceptance output: {output}', flush=True)
        report = {'inputs': identity, 'status': 'running', 'sweep': [],
                  'thresholds': {'track_delta_pct': args.max_track_delta_pct, 'descriptor_l2_pct': args.max_l2_pct,
                                 'vram_mib_exclusive': VRAM_LIMIT_MIB, 'ram_bytes_exclusive': RAM_LIMIT_BYTES}}
        report['offline_tests'] = offline_tests(output)
        runs = sweep(videos, image, output, report)
        trajectory = report['trajectory'] = {}
        quality = report['quality'] = []
        timing = report['cold_vs_b3'] = {}
        if runs:
            trajectory.update(check_trajectories(runs, references, max_track_pct=args.max_track_delta_pct,
                                                 max_l2_pct=args.max_l2_pct))
            for scene, (_, baseline) in SCENES.items():
                trial = runs[scene]
                seconds = trial['wall_seconds'] if trial['status'] == 'complete' else None
                timing[scene] = {'cpu_b3_seconds': baseline, 'cuda_seconds': seconds,
                                 'trial_status': trial['status'],
                                 'speedup': baseline / seconds if seconds else None,
                                 'pass': within_resources(trial) and trial['wall_seconds'] < baseline}
        report['gates'] = gates = {
            'npy_format_and_frames': len(trajectory) == len(SCENES) and all(row['format_pass'] for row in trajectory.values()),
            'descriptor_and_track_drift': len(trajectory) == len(SCENES) and all(row['drift_pass'] for row in trajectory.values()),
            'quality_10_seeds': None,
            'speed_and_resources': len(timing) == len(SCENES) and all(row['pass'] for row in timing.values()),
            'offline_regression': all(code == 0 for code in report['offline_tests'].values()),
        }
        write_json(output / 'acceptance.json', report)
        if runs and gates['npy_format_and_frames']:
            quality.extend(check_quality(runs, output, cpu_scores, args.scorer, args.dataset, args.quality_jobs))
            gates['quality_10_seeds'] = all(row['pass'] for row in quality)
        report['status'] = 'PASS' if all(gates.values()) else 'NOT PASS'
        write_json(output / 'acceptance.json', report)
        print(f"4A {report['status']} | {output / 'acceptance.json'}", flush=True)
        return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', type=Path, default=TEMP / 'RAVEE-optz-cold-0eb23d8e69ef')
    parser.add_argument('--dataset', type=Path, default=Path('D:/Solve4X'))
    parser.add_argument('--scorer', type=Path,
                        default=TEMP / 'RAVEE-optz-benchmark-runner-20261126' / 'score_ravee_optz.py')
    parser.add_argument('--output', type=Path, default=PROJECT / 'temp' / 'idt_cuda_acceptance')
    parser.add_argument('--quality-jobs', type=int, default=2)
    parser.add_argument('--max-track-delta-pct', type=float, default=15)
    parser.add_argument('--max-l2-pct', type=float, default=10)
    args = parser.parse_args()
    if args.quality_jobs < 1 or not 0 <= args.max_track_delta_pct <= 100 or not 0 <= args.max_l2_pct <= 100:
        parser.error('Use a positive job count and finite tolerances between 0 and 100 percent.')
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f'Benchmark blocked: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
