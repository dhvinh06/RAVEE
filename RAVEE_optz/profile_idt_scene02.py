"""Measure the IDT C++ core on Solve4X Scene 02 with the existing CPU image."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


CPU_IMAGE = 'ravee-idt:bdae91ff9c7afb9995f3'
WARMUP = 30
FRAME_COUNT = 430
CORE_MARKER = 'IDT_CORE_PROFILE '
RESULT_MARKER = 'IDT_SCENE02_PROFILE '
SCENE_PATH = ('RAVEE-optz-cold-0eb23d8e69ef/scene02/preprocessed/'
              '1_scene02_cam2_15fps_reduced.avi')


def read_sample(video_path, frame_count=FRAME_COUNT):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if (width, height) != (640, 480) or abs(fps - 15) > 0.001 or frames < frame_count:
            raise ValueError(f'Expected a 640x480, 15 FPS AVI. Got {width}x{height}, {fps} FPS, {frames} frames.')
        start = (frames - frame_count) // 2
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start) or cap.get(cv2.CAP_PROP_POS_FRAMES) != start:
            raise ValueError(f'Could not seek to frame {start} in {video_path}')
        sample = np.empty((frame_count, height, width), dtype=np.uint8)
        for index in range(frame_count):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Could not decode frame {start + index} in {video_path}')
            sample[index] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()
    return sample, start


def profile_in_container(video_path):
    # Load cv2 first, as the CPU image's normal IDT entry point does.
    sample, start = read_sample(video_path)
    import densetrack

    started = time.perf_counter()
    tracks = densetrack.densetrack(sample, adjust_camera=True)
    elapsed = time.perf_counter() - started
    print(RESULT_MARKER + json.dumps({
        'sample_start_frame': start, 'sample_frames': len(sample),
        'warmup_frames': WARMUP, 'width': sample.shape[2], 'height': sample.shape[1],
        'fps': 15, 'workers': 1, 'track_count': len(tracks),
        'densetrack_seconds_including_warmup_and_packing': elapsed,
    }), flush=True)


def run_logged(command, log_path):
    output = []
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                output.append(line)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    return output


def summarize(output):
    core, = [json.loads(line[len(CORE_MARKER):]) for line in output if line.startswith(CORE_MARKER)]
    result, = [json.loads(line[len(RESULT_MARKER):]) for line in output if line.startswith(RESULT_MARKER)]
    total = core['core_seconds']
    if core['frames'] != FRAME_COUNT - WARMUP or total <= 0 or result['track_count'] == 0:
        raise ValueError(f'Incomplete C++ profile: {core}')
    groups = {
        'camera': core['surf_seconds'] + core['camera_match_warp_seconds'],
        'farneback': core['flow_raw_seconds'] + core['flow_warped_seconds'] + core['flow_state_seconds'],
        'tracking': core['tracking_seconds'],
        'hog_hof_mbh': core['descriptor_seconds'],
    }
    groups['other'] = total - sum(groups.values())
    if min(groups.values()) < 0:
        raise ValueError(f'Invalid timing breakdown: {groups}')
    flow_fraction = groups['farneback'] / total
    return dict(result, core=core, seconds=groups,
                percent={name: 100 * seconds / total for name, seconds in groups.items()},
                farneback_only_core_speedup_ceiling=1 / (1 - flow_fraction))


def stage_source(source, staged):
    idt = source.parent / 'IDT'
    shutil.copytree(idt / 'src', staged / 'src')
    shutil.copy2(idt / 'setup.py', staged)
    shutil.copy2(source, staged)
    cpp = staged / 'src' / 'DenseTrackStab.cpp'
    cpp.write_bytes(cpp.read_bytes().replace(b'\r\n', b'\n'))
    patch = source.with_name('idt_core_profile.patch')
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=staged, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=staged, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=Path(tempfile.gettempdir()) / SCENE_PATH)
    parser.add_argument('--cpu-image', default=CPU_IMAGE)
    parser.add_argument('--inside-container', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inside_container:
        profile_in_container(args.inside_container)
        return 0

    video = args.video.resolve(strict=True)
    scratch = Path(tempfile.mkdtemp(prefix='ravee-idt-4a-profile-'))
    print(f'Profile output: {scratch}', flush=True)
    subprocess.run(['docker', 'image', 'inspect', args.cpu_image],
                   stdout=subprocess.DEVNULL, check=True)
    gpu_ok = False
    gpu_probe = (
        'if command -v nvidia-smi >/dev/null 2>&1; then exec nvidia-smi -L; fi; '
        'if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then exec /usr/lib/wsl/lib/nvidia-smi -L; fi; '
        'echo "Docker did not expose nvidia-smi" >&2; exit 127'
    )
    try:
        gpu_output = run_logged([
            'docker', 'run', '--rm', '--pull', 'never', '--gpus', 'all',
            '-e', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
            '--entrypoint', '/bin/sh', args.cpu_image, '-ec', gpu_probe,
        ], scratch / 'gpu.log')
        gpu_ok = any('GTX 1650' in line for line in gpu_output)
    except subprocess.CalledProcessError:
        print('GPU probe failed. Continuing with the CPU profile.', file=sys.stderr, flush=True)
    print(f'GTX 1650 in Docker: {"PASS" if gpu_ok else "UNVERIFIED"}', flush=True)

    staged = scratch / 'source'
    stage_source(Path(__file__).resolve(), staged)
    runtime = (
        'mkdir -p /tmp/idt-profile && cp -R /profile-source/. /tmp/idt-profile/ '
        '&& cd /tmp/idt-profile && '
        'export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && '
        'IDT_CPU_FLAGS="-mavx2 -mfma -ffp-contract=off" '
        'python3 setup.py build_ext --force --inplace && '
        'python3 -u profile_idt_scene02.py --inside-container /scene02.avi'
    )
    print('Compiling the instrumented extension in the existing CPU image.', flush=True)
    output = run_logged([
        'docker', 'run', '--rm', '--pull', 'never', '--memory', '6656m',
        '--memory-swap', '6656m', '-e', 'RAVEE_IDT_WORKERS=1',
        '--mount', f'type=bind,source={staged},target=/profile-source,readonly',
        '--mount', f'type=bind,source={video},target=/scene02.avi,readonly',
        '--entrypoint', '/bin/sh', args.cpu_image, '-ec', runtime,
    ], scratch / 'cpu.log')
    summary = dict(summarize(output), docker_gtx_1650=gpu_ok, cpu_image=args.cpu_image)
    rendered = json.dumps(summary, indent=2) + '\n'
    (scratch / 'profile.json').write_text(rendered, encoding='utf-8')
    print(rendered, end='', flush=True)
    return 0 if gpu_ok else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise SystemExit(str(error))
