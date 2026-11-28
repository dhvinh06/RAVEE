"""Validate the restored Fisher Vector settings against saved IDT W=6 runs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys
import tempfile
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parent
CPU = PROJECT / 'idt-phase3-20261126-205155.bak'
CUDA = PROJECT / 'temp/idt_cuda_w6_trial'
ARCHIVE = Path(tempfile.gettempdir()) / 'ravee-fv-step4-388b44dc'
MULTICORE = ARCHIVE.with_name('ravee-fv-multicore-9fcd5d7b')
SCENES = {'scene01': 5883, 'scene02': 3311}
METRICS = ('raw_ari', 'oracle_ari', 'oracle_accuracy')
OPTIONS = dict(mode='optimized', fit_samples=100000, gmm_tol=.003, workers=6, posterior='numpy')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs(profile, scene):
    stem = f'1_{scene}_cam2_15fps_reduced-trajectory.npy'
    if profile == 'cpu':
        old = read(CPU / 'acceptance_run/seed-0/acceptance.json')['scenes'][scene]['6']
        return CPU / 'acceptance_input' / f'w6-{stem}', old['trajectory_sha256']
    folder = CUDA / f'{scene}-w6'
    return folder / stem, read(folder / 'result.json')['trajectory_sha256']


def trial(output, profile, scene, seed, repeat):
    import accept_idt as prior
    import psutil
    from threadpoolctl import threadpool_limits

    source, expected = inputs(profile, scene)
    source_sha = digest(source)
    if source_sha != expected:
        raise ValueError(f'Trajectory checksum changed for {source}')
    signature = {'trajectory_sha256': source_sha, 'fv_code_sha256': digest(Path(prior.fv.__file__)),
                 'options': OPTIONS, 'seed': seed}
    destination = output / profile / f'seed-{seed}'
    folder = destination / (f'{scene}-repeat' if repeat else scene)
    record_path = folder / 'step2.json'
    if record_path.is_file():
        if read(record_path)['signature'] != signature:
            raise ValueError(f'Existing trial uses different code or settings: {record_path}')
        return
    destination.mkdir(parents=True, exist_ok=True)
    extract = prior.fv.extract_framewise_features
    process = psutil.Process()
    peak = 0

    def configured(path, target, **options):
        nonlocal peak
        features = extract(path, target, random_state=seed, **OPTIONS, **options)
        peak = process.memory_info().peak_wset
        return features

    print(f'START {profile} {scene} seed={seed} repeat={repeat}', flush=True)
    with tempfile.TemporaryDirectory(prefix='fv-step2-') as scratch, threadpool_limits(limits=1):
        work = Path(scratch)
        with patch.object(prior.fv, 'extract_framewise_features', configured):
            timing = prior.downstream(source, work, scene, SCENES[scene], seed, progress=False)
        record = {'signature': signature, 'timing': timing, 'peak_ram_bytes': peak,
                  'fv_sha256': digest(work / f'{source.stem}-frame_fv.txt')}
        if seed == 0:
            archived = (ARCHIVE / f'sample100k_tol003/seed-0/{scene}/stage4.json' if profile == 'cpu'
                        else MULTICORE / f'{profile}-{scene}-mixed.json')
            record['matches_archived_sha256'] = record['fv_sha256'] == read(archived)['feature_text_sha256']
        folder.mkdir(exist_ok=True)
        for filename in ('segments.csv', 'prediction_seconds.csv'):
            (folder / filename).write_bytes((work / filename).read_bytes())
        prior.write_json(record_path, record)
    print(f'DONE {profile} {scene} seed={seed} repeat={repeat} '
          f'{timing["fisher_vector_seconds"]:.2f} s, {peak / 2**30:.3f} GiB, '
          f'FV SHA-256 {record["fv_sha256"]}; archive match {record.get("matches_archived_sha256")}', flush=True)


def summarize(output):
    import accept_idt as prior
    import idt_multiseed as multi

    scorer = multi.load_scorer()
    cuda_reference, = CUDA.glob('quality-*/scores-by-seed.json')
    reference = read(cuda_reference)
    cuda = output / 'cuda'
    scores = {str(seed): scorer.score_run(cuda / f'seed-{seed}', tuple(SCENES), multi.DATASET,
                                         prior.load_seconds) for seed in range(10)}
    report = {'checks': {}, 'cpu_quality': read(ARCHIVE / 'summary.json')['quality'],
              'cuda_quality': {}, 'cuda_scores_by_seed': scores}
    for profile in ('cpu', 'cuda'):
        report['checks'][profile] = {}
        for scene in SCENES:
            root = output / profile / 'seed-0'
            first = read(root / scene / 'step2.json')
            second = read(root / f'{scene}-repeat' / 'step2.json')
            pairs = (first, second)
            peak = max(x['peak_ram_bytes'] for x in pairs)
            equal = first['fv_sha256'] == second['fv_sha256']
            report['checks'][profile][scene] = {
                'equal': equal, 'cosine': 1.0 if equal else None,
                'sha256': [row['fv_sha256'] for row in pairs],
                'matches_archive': all(row['matches_archived_sha256'] for row in pairs),
                'seconds': [row['timing']['fisher_vector_seconds'] for row in pairs],
                'peak_gib': peak / 2**30, 'ram_pass': peak < 1.7 * 2**30,
            }
    for scope in (*SCENES, 'combined'):
        metrics = report['cuda_quality'][scope] = {}
        for metric in METRICS:
            old = [reference[seed][scope][metric] for seed in scores]
            new = [scores[seed][scope][metric] for seed in scores]
            center, sd, average = mean(old), stdev(old), mean(new)
            metrics[metric] = {'mean': average, 'sd': stdev(new), 'reference_mean': center,
                               'reference_sd': sd, 'pass': abs(average - center) <= sd}
    prior.write_json(output / 'summary.json', report)
    print(output / 'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--trial', nargs=3, metavar=('PROFILE', 'SCENE', 'SEED'))
    parser.add_argument('--repeat', action='store_true')
    args = parser.parse_args()
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'
    sys.path[:0] = [str(PROJECT), str(CPU)]
    if sys.flags.no_site:
        sys.path.append(str(Path(sys.executable).resolve().parent.parent / 'Lib/site-packages'))
    if args.trial:
        profile, scene, seed = args.trial
        if profile not in ('cpu', 'cuda') or scene not in SCENES or seed not in tuple(map(str, range(10))):
            parser.error('Expected cpu/cuda, scene01/scene02 and seed 0..9')
        trial(args.output, profile, scene, int(seed), args.repeat)
    else:
        summarize(args.output)


if __name__ == '__main__':
    main()
