"""Run and archive one isolated, seed-0 Solve4X cold scene."""

import argparse
from collections import Counter
import ctypes
from datetime import datetime
from hashlib import file_digest
import json
from pathlib import Path
import math
import os
import re
import shutil
import subprocess
import sys
from threading import Event, Thread

sys.dont_write_bytecode = True

parse = argparse.ArgumentParser(description=__doc__)
parse.add_argument("--repo", required=True, type=Path)
parse.add_argument("--input-video", required=True, type=Path)
parse.add_argument("--scratch", required=True, type=Path)
parse.add_argument("--archive", required=True, type=Path)
parse.add_argument("--scene", required=True, choices=("scene01", "scene02"))
parse.add_argument("--backend", required=True, choices=("cpu", "cuda"))
args = parse.parse_args()

root = args.repo.resolve(strict=True)
project = root / "RAVEE_optz"
sys.path[:0] = [str(project), str(project / "toolbox"), str(root)]
os.environ.update(RAVEE_IDT_BACKEND=args.backend, RAVEE_IDT_WORKERS="6", RAVEE_FV_SEED="0",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
from threadpoolctl import threadpool_limits

from benchmark_ravee_orig import Profile, trajectory_frame_indices, video_timing, write_csv
from finch.finch import FINCH
from toolbox.feature_extraction import extract_framewise_features
from toolbox.pipeline_cache import build_signature, extract_idt, idt_backend, validate_tracks, video_info, validate_preprocessed
from toolbox.preprocessing import FPS_RESIZE, HEIGHT_RESIZE, WIDTH_RESIZE, reduce_video_size


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def power_status():
    status = (ctypes.c_ubyte * 12)()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise ctypes.WinError()
    return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "ac": status[0],
            "power_scheme": subprocess.check_output(["powercfg", "/getactivescheme"], text=True).strip()}


def on_ac_high_performance(power):
    return power["ac"] == 1 and "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c" in power["power_scheme"].lower()


def sample_gpu(done, samples, errors):
    while not done.is_set():
        try:
            row = subprocess.check_output(
                ["nvidia-smi", "-i", "0", "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
            samples.append(tuple(map(int, row.split(","))))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            errors.append(str(error))
        done.wait(0.5)


def existing_image(backend):
    signature = build_signature(project / "IDT", backend=backend)
    tag = "ravee-idt:cuda" if backend == "cuda" else f"ravee-idt:{signature[:20]}"
    image, label = subprocess.check_output(
        ["docker", "image", "inspect", "--format", '{{.Id}} {{index .Config.Labels "ravee.build"}}', tag],
        text=True).split()
    if signature != label:
        raise RuntimeError(f"Image build label does not match the IDT source: {tag}")
    return image, tag, signature


def run_scene(scene, source, output, image, backend, archive, identity):
    video_dir, idt_dir, fv_dir, finch_dir = (output / name for name in ("preprocessed", "idt", "fv", "finch"))
    for directory in (video_dir, idt_dir, fv_dir, finch_dir):
        directory.mkdir(parents=True)
    profile = Profile(output, "full_pipeline", source)
    profile.have_nvidia_smi = False  # A single sampler records both utilization and VRAM for E2.
    profile.metadata.update(video_id=scene, requested_clusters=8, implementation="RAVEE_optz",
                            image=image, backend=backend, idt_workers=6, seed=0, **identity)
    input_fps, input_frames = video_timing(source)
    samples, errors = [], []
    done = Event()
    gpu = Thread(target=sample_gpu, args=(done, samples, errors), daemon=True)
    profile.metadata["power_before"] = before = power_status()
    if not on_ac_high_performance(before):
        raise RuntimeError(f"Cold-run power conditions are unmet: {before}")
    if backend == "cuda":
        gpu.start()
    try:
        if input_fps != FPS_RESIZE:
            raise ValueError(f"Expected 15 FPS for {source}, found {input_fps:g}.")
        reduced = video_dir / f"1_{source.stem}_reduced.avi"
        raw_name = f"{reduced.stem}-trajectory.npy"
        with profile.stage("Preprocessing", video_dir):
            reduce_video_size(str(source), str(video_dir), 1, source.stem)
            if not reduced.is_file() or reduced.stat().st_size == 0:
                raise RuntimeError("Preprocessing did not write a video.")
        validate_preprocessed(reduced, video_info(source), (FPS_RESIZE, WIDTH_RESIZE, HEIGHT_RESIZE))
        raw = idt_dir / raw_name
        container_metrics = idt_dir / "container_metrics.json"
        with profile.stage("IDT (Docker)", idt_dir, container_metrics=container_metrics):
            with (archive / "container.log").open("w", encoding="utf-8") as log:
                extract_idt(reduced, raw, image, workers=6, backend=backend, container_log=log,
                            run_container=run_local_image)
            report = (archive / "container.log").read_text(encoding="utf-8", errors="replace")
            peak = re.search(r"IDT container peak memory bytes: (\d+)", report)
            if peak:
                container_metrics.write_text(json.dumps({"peak_ram_bytes": int(peak[1])}), encoding="utf-8")
                profile.metadata["docker_peak_ram_bytes"] = int(peak[1])
        validate_tracks(raw, input_frames)
        with profile.stage("Fisher Vector", fv_dir):
            with threadpool_limits(limits=1):
                features = extract_framewise_features(str(raw), str(fv_dir), mode="optimized", random_state=0,
                                                      fit_samples=100000, gmm_tol=.003, workers=6,
                                                      posterior="numpy", progress=lambda *_: None)
        with profile.stage("TW-FINCH", finch_dir):
            hierarchy, counts, partition = FINCH(features, req_clust=8, tw_finch=True)
            partition = np.asarray(partition).reshape(-1)
            if np.unique(partition).size != 8:
                raise RuntimeError("TW-FINCH did not return 8 clusters.")
            np.savetxt(finch_dir / "hierarchy.csv", hierarchy, delimiter=",", fmt="%d")
            np.savetxt(finch_dir / "cluster_counts.csv", counts, fmt="%d")
            np.savetxt(finch_dir / "selected_partition.csv", partition, fmt="%d")

        profile.metadata["power_after"] = power_status()
        profile.metadata["four_stage_conditions_ok"] = on_ac_high_performance(profile.metadata["power_after"])
        print(f"POWER {scene} before={before} after={profile.metadata['power_after']}", flush=True)
        done.set()
        if backend == "cuda":
            gpu.join()
            profile.metadata["vram_peak_mib"] = max((row[0] for row in samples), default=None)
            profile.metadata["gpu_peak_celsius"] = max((row[1] for row in samples), default=None)
            profile.metadata["gpu_peak_utilization_percent"] = max((row[2] for row in samples), default=None)
            profile.metadata["gpu_samples"] = len(samples)
            profile.metadata["gpu_errors"] = errors
        fv_file = fv_dir / f"{raw.stem}-frame_fv.txt"
        profile.metadata["hashes"] = {label: sha256(path) for label, path in
                                       (("source", source), ("preprocessed", reduced),
                                        ("trajectory", raw), ("fv_text", fv_file))}
        frames = trajectory_frame_indices(raw)
        if len(frames) != len(partition) or np.any(np.diff(frames) <= 0):
            raise RuntimeError("Cannot align Fisher Vector rows with increasing video frames.")
        starts = np.r_[0, np.flatnonzero(partition[1:] != partition[:-1]) + 1]
        ends = np.r_[starts[1:], len(partition)]
        write_csv(output / "segments.csv", ("segment", "cluster", "start_frame", "end_frame"), (
            (index, int(partition[start]), int(frames[start]), int(frames[end - 1]))
            for index, (start, end) in enumerate(zip(starts, ends), 1)
        ))
        seconds = frames // FPS_RESIZE
        boundaries = np.r_[0, np.flatnonzero(seconds[1:] != seconds[:-1]) + 1, len(seconds)]
        clusters = {
            int(seconds[start]): int(Counter(partition[start:end]).most_common(1)[0][0])
            for start, end in zip(boundaries[:-1], boundaries[1:])
        }
        write_csv(output / "prediction_seconds.csv", ("video", "second", "cluster", "activity"), (
            (scene, second, clusters.get(second, -1), "Unassigned")
            for second in range(math.ceil(input_frames / input_fps))
        ))
        profile.metadata["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        profile.metadata.update(status="failed", error=str(error))
        raise
    finally:
        done.set()
        if gpu.ident is not None:
            gpu.join()
        if "power_after" not in profile.metadata:
            profile.metadata["power_after"] = power_status()
        profile.save()
        for name in ("profile.csv", "profile.json", "segments.csv", "prediction_seconds.csv"):
            path = output / name
            if path.is_file():
                shutil.copy2(path, archive / name)
        print(f"Profile: {output / 'profile.csv'}", flush=True)


def run_local_image(command, **kwargs):
    return subprocess.run(command[:3] + ["--pull", "never"] + command[3:], **kwargs)


def main():
    backend = idt_backend()
    source = args.input_video.resolve(strict=True)
    scratch = args.scratch.resolve(strict=True)
    temp_root = Path(os.environ["TEMP"]).resolve(strict=True)
    if scratch.parent != temp_root or not scratch.name.startswith("ravee-phase4-"):
        raise ValueError("Scratch must be a dedicated ravee-phase4-* directory directly under TEMP.")
    if source.name != f"{args.scene}_cam2_15fps.mp4" or source.is_relative_to(scratch):
        raise ValueError(f"Unexpected Solve4X camera 2 input: {source}")
    image, tag, signature = existing_image(backend)
    output = scratch / args.scene
    output.mkdir()
    archive = args.archive.resolve() / args.scene
    archive.mkdir(parents=True)
    print(f"START {backend} {args.scene} W=6 seed=0 | {output}", flush=True)
    run_scene(args.scene, source, output, image, backend, archive,
              {"image_tag": tag, "image_signature": signature})


if __name__ == "__main__":
    main()


