"""Run the original RAVEE stages in isolation and score aligned one-second labels.

Use the RAVEE_org virtual environment. Run this script with --help for commands.
Each benchmark needs a new output directory outside RAVEE_org and RAVEE_optz.
"""

import argparse
from collections import Counter
import csv
import ctypes
from contextlib import contextmanager
from datetime import datetime, time as clock_time, timedelta
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from threading import Event, Thread
import time


ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "RAVEE_org"
DEMO = ORIGINAL / "DemoFiles"
TABLE_4 = {
    "Solve4X": dict(ri=0.8921, ari=0.7067, accuracy=0.7113,
                    weighted_precision=0.7366, weighted_recall=0.7113),
    "IKEA": dict(ri=0.8777, ari=0.7193, accuracy=0.7722,
                 weighted_precision=0.7537, weighted_recall=0.7722),
}

# IDT's public runner defaults to whole-video MBH Fisher vectors. Select its
# existing raw trajectory mode so the original framewise FV can run on its output.
IDT_ENTRYPOINT = """
import json
import resource
import time
from pathlib import Path

started = time.perf_counter()
cpu_started = time.process_time()
import run_idt

run_idt.HOG_FISHER_VECTOR = False
run_idt.HOF_FISHER_VECTOR = False
run_idt.MBH_FISHER_VECTOR = False
try:
    run_idt.main()
finally:
    Path('features/idt_resources.json').write_text(json.dumps({
        'process_wall_s': time.perf_counter() - started,
        'cpu_s': time.process_time() - cpu_started,
        'peak_ram_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }))
"""


def rss_reader():
    if sys.platform != "win32":
        def linux_rss():
            return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return linux_rss

    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD), ("page_faults", wintypes.DWORD),
            *((name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
            )),
        ]

    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
    handle = current_process()

    def windows_rss():
        counters = MemoryCounters()
        counters.size = ctypes.sizeof(counters)
        if not get_memory(handle, ctypes.byref(counters), counters.size):
            raise ctypes.WinError()
        return counters.working_set

    return windows_rss


def gpu_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = [int(value) for value in result.stdout.split() if value.isdecimal()]
    return max(values, default=None)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def video_timing(path):
    import cv2

    video = cv2.VideoCapture(str(path))
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        video.release()
    if not math.isfinite(fps) or fps <= 0 or frames <= 0:
        raise RuntimeError(f"Could not determine FPS and frame count for {path}.")
    return fps, frames


class Profile:
    def __init__(self, output, mode, source):
        self.output = output
        self.stages = []
        self.read_rss = rss_reader()
        self.have_nvidia_smi = shutil.which("nvidia-smi") is not None
        self.metadata = dict(mode=mode, source=str(source), status="running",
                             platform=platform.platform(), python=sys.version,
                             host_logical_cores=os.cpu_count())

    def save(self):
        compute_total = sum(row["wall_s"] for row in self.stages
                            if row["status"] == "ok" and row["counts_toward_pipeline"])
        complete = self.metadata["mode"] == "full_pipeline" and self.metadata["status"] == "ok"
        for row in self.stages:
            row["pipeline_pct"] = (100 * row["wall_s"] / compute_total
                                   if complete and row["counts_toward_pipeline"]
                                   else None)
        self.metadata["measured_compute_s"] = compute_total
        self.metadata["stages"] = self.stages
        (self.output / "profile.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        if self.stages:
            columns = tuple(self.stages[0])
            write_csv(self.output / "profile.csv", columns,
                      ([row[column] for column in columns] for row in self.stages))

    @contextmanager
    def stage(self, name, artifact_dir=None, *, compute=True, container_metrics=None):
        print(f"\n--- {name} ---", flush=True)
        row = dict(stage=name, status="failed", counts_toward_pipeline=compute)
        self.stages.append(row)
        stop = Event()
        rss_samples = [self.read_rss()]
        gpu_samples = []

        def sample_rss():
            while not stop.wait(0.1):
                rss_samples.append(self.read_rss())

        def sample_gpu():
            while not stop.is_set():
                value = gpu_usage()
                if value is not None:
                    gpu_samples.append(value)
                stop.wait(1)

        samplers = [Thread(target=sample_rss, daemon=True)]
        if self.have_nvidia_smi:
            samplers.append(Thread(target=sample_gpu, daemon=True))
        for sampler in samplers:
            sampler.start()
        start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            yield
            row["status"] = "ok"
        finally:
            elapsed = time.perf_counter() - start
            host_cpu = time.process_time() - cpu_start
            rss_samples.append(self.read_rss())
            stop.set()
            for sampler in samplers:
                sampler.join()
            idt_stats = (json.loads(container_metrics.read_text())
                         if container_metrics is not None and container_metrics.is_file() else {})
            cpu_s = idt_stats.get("cpu_s") if container_metrics is not None else host_cpu
            cpu_window = idt_stats.get("process_wall_s", elapsed)
            artifacts = ([p for p in artifact_dir.iterdir() if p.is_file() and p != container_metrics]
                         if artifact_dir is not None else [])
            row.update(
                wall_s=elapsed, cpu_s=cpu_s,
                cpu_avg_pct_one_core=100 * cpu_s / cpu_window if cpu_s is not None else None,
                peak_ram_bytes=(idt_stats.get("peak_ram_bytes") if container_metrics is not None
                                else max(rss_samples)),
                ram_scope="container_process_peak_rss" if container_metrics is not None else "host_process_sampled_rss",
                gpu_host_avg_pct=sum(gpu_samples) / len(gpu_samples) if gpu_samples else None,
                gpu_host_peak_pct=max(gpu_samples, default=None), gpu_samples=len(gpu_samples),
                artifact_bytes=sum(p.stat().st_size for p in artifacts),
            )
            self.save()
            print(f"{name}: {elapsed:.3f}s ({row['status']})", flush=True)


def trajectory_frame_indices(trajectories):
    import numpy as np

    # Recover the exact frame order used by the original FV's dictionary.
    tracks = np.load(trajectories, mmap_mode="r")
    frame_ends = np.asarray(tracks[tracks.dtype.names[0]]).reshape(-1)
    _, first = np.unique(frame_ends, return_index=True)
    return np.array(list(dict.fromkeys(
        frame for end in frame_ends[np.sort(first)] for frame in range(int(end) - 14, int(end) + 1)
    )))


def collect_labels(output, reduced, segments, profile):
    from display_video_snippets import display_video_snippet
    from gui import gui_elements

    checkpoint = output / "segment_labels.json"
    activities = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else []
    if (not isinstance(activities, list) or len(activities) > len(segments)
            or any(not isinstance(label, str) or not label.strip() for label in activities)):
        raise ValueError(f"Invalid label checkpoint: {checkpoint}")
    if len(activities) == len(segments):
        return activities

    print(f"Saved labels: {len(activities)}/{len(segments)}. Enter a label, click Commit, then Exit.", flush=True)
    with profile.stage("Human labeling", compute=False):
        for index, (_, _, start, end) in enumerate(segments[len(activities):], len(activities) + 1):
            print(f"Segment {index}/{len(segments)}", flush=True)
            gui_elements.LABEL.clear()
            gui_elements.counter = 0
            display_video_snippet(str(reduced), start, end)
            label = list(gui_elements.LABEL.values())[-1].strip() if gui_elements.LABEL else ""
            if not label:
                raise RuntimeError(f"Segment {index} needs a label. Run resume to reopen it.")
            activities.append(label)
            temp = checkpoint.with_suffix(".tmp")
            temp.write_text(json.dumps(activities, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(checkpoint)
    return activities


def benchmark(args):
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ORIGINAL), str(ORIGINAL / "toolbox")]
    import numpy as np
    from preprocessing import FPS_RESIZE, reduce_video_size

    output = args.output.resolve()
    if any(output.is_relative_to(ROOT / folder) for folder in ("RAVEE_org", "RAVEE_optz", ".git")):
        raise ValueError("Choose a new output directory outside RAVEE_org, RAVEE_optz and .git.")
    resume = args.command == "resume"
    demo = args.command == "demo"
    if resume:
        saved = json.loads((output / "profile.json").read_text(encoding="utf-8"))
        stages = saved.pop("stages")
        if saved["mode"] != "full_pipeline" or not any(
                row["stage"] == "TW-FINCH" and row["status"] == "ok" for row in stages):
            raise ValueError("Resume needs a run that completed TW-FINCH.")
        if saved["status"] != "failed":
            raise ValueError("Resume needs a failed or interrupted run.")
        source = Path(saved["source"])
        profile = Profile(output, saved["mode"], source)
        profile.metadata.update(saved)
        profile.metadata.pop("error", None)
        profile.metadata["status"] = "running"
        profile.stages = [row for row in stages if row["stage"] != "Event Log export"]
    else:
        source = DEMO / "scan_video_reduced.avi" if demo else args.video.resolve(strict=True)
        output.mkdir(parents=True, exist_ok=False)
        profile = Profile(output, "demo_export" if demo else "full_pipeline", source)
        profile.metadata["requested_clusters"] = args.clusters
    video_id = args.video_id or profile.metadata.get("video_id") or source.stem
    profile.metadata["video_id"] = video_id
    try:
        if demo:
            partition = np.loadtxt(DEMO / "scan_video_reduced_c.csv", delimiter=",", dtype=int)[:, -1]
            with (DEMO / "scan_video_reduced_eventlog.csv").open(newline="") as file:
                activities = [row["activity"] for row in csv.DictReader(file)]
            reduced = source
            frame_indices = np.arange(partition.size)
        elif resume:
            reduced = output / "preprocessed" / f"1_{source.stem}_reduced.avi"
            video_timing(reduced)
            trajectories = output / "idt" / f"{reduced.stem}-trajectory.npy"
            frame_indices = trajectory_frame_indices(trajectories)
            partition = np.loadtxt(output / "finch" / "selected_partition.csv", dtype=int, ndmin=1)
            print("Loaded the saved TW-FINCH partition. Resuming at human labeling.", flush=True)
        else:
            from feature_extraction import extract_framewise_features
            from finch.finch import FINCH

            video_dir, idt_dir, fv_dir, finch_dir = (output / name for name in ("preprocessed", "idt", "fv", "finch"))
            for directory in (video_dir, idt_dir, fv_dir, finch_dir):
                directory.mkdir()
            reduced = video_dir / f"1_{source.stem}_reduced.avi"
            input_fps, input_frames = video_timing(source)
            with profile.stage("Preprocessing", video_dir):
                reduce_video_size(str(source), str(video_dir), 1, source.stem)
                if not reduced.is_file() or reduced.stat().st_size == 0:
                    raise RuntimeError("Original preprocessing did not write a video.")
            reduced_fps, reduced_frames = video_timing(reduced)
            input_duration = input_frames / input_fps
            reduced_duration = reduced_frames / reduced_fps
            profile.metadata["video_timing"] = dict(input_fps=input_fps, input_frames=input_frames,
                                                   reduced_fps=reduced_fps, reduced_frames=reduced_frames)
            if (reduced_fps != FPS_RESIZE
                    or abs(input_duration - reduced_duration) > max(1 / input_fps, 1 / reduced_fps) + 1e-9):
                raise RuntimeError(f"Preprocessing changed the time axis ({input_duration:.3f}s to "
                                   f"{reduced_duration:.3f}s). Resample the source to {FPS_RESIZE} FPS "
                                   "while preserving its duration, then rerun into a new directory.")
            print(f"Verified: {reduced_frames} frames at {reduced_fps:g} FPS ({reduced_duration:.3f}s)", flush=True)
            if args.build:
                with profile.stage("Docker image build", compute=False):
                    subprocess.run(["docker", "build", "-t", args.image, str(ORIGINAL / "IDT")], check=True)
            metrics = idt_dir / "idt_resources.json"
            with profile.stage("IDT (Docker)", idt_dir, container_metrics=metrics):
                subprocess.run([
                    "docker", "run", "--rm", "--entrypoint", "python3",
                    "-v", f"{video_dir}:/densetrack/data:ro",
                    "-v", f"{idt_dir}:/densetrack/features",
                    args.image, "-u", "-c", IDT_ENTRYPOINT,
                ], check=True)
                trajectories = idt_dir / f"{reduced.stem}-trajectory.npy"
                if not trajectories.is_file() or not metrics.is_file():
                    raise RuntimeError("IDT did not produce raw trajectories and resource metrics.")
            with profile.stage("Fisher Vector", fv_dir):
                features = extract_framewise_features(str(trajectories), str(fv_dir))
            with profile.stage("TW-FINCH", finch_dir):
                hierarchy, counts, partition = FINCH(features, req_clust=args.clusters, tw_finch=True)
                partition = np.asarray(partition).reshape(-1)
                if np.unique(partition).size != args.clusters:
                    raise RuntimeError(f"TW-FINCH did not return {args.clusters} distinct clusters.")
                np.savetxt(finch_dir / "hierarchy.csv", hierarchy, delimiter=",", fmt="%d")
                np.savetxt(finch_dir / "cluster_counts.csv", counts, fmt="%d")
                np.savetxt(finch_dir / "selected_partition.csv", partition, fmt="%d")
            frame_indices = trajectory_frame_indices(trajectories)

        if len(frame_indices) != len(partition) or np.any(np.diff(frame_indices) <= 0):
            raise RuntimeError("Cannot align the original FV rows with increasing video frames.")
        starts = np.r_[0, np.flatnonzero(partition[1:] != partition[:-1]) + 1]
        ends = np.r_[starts[1:], len(partition)]
        segments = [(index, int(partition[start]), int(frame_indices[start]), int(frame_indices[end - 1]))
                    for index, (start, end) in enumerate(zip(starts, ends), 1)]
        if not resume:
            write_csv(output / "segments.csv", ("segment", "cluster", "start_frame", "end_frame"), segments)
        if not demo:
            activities = collect_labels(output, reduced, segments, profile)

        from event_log_creation import exportCSV, exportXES, extractLog

        events = output / "event_log"
        events.mkdir(exist_ok=resume)
        with profile.stage("Event Log export", events):
            labels = np.array(list(enumerate(activities, 1)))
            today = datetime.now().date()
            log = extractLog(1, labels, partition[:, None], 0)
            # Keep the original rounded event times and present them as full dates to PM4Py.
            times = []
            for value in log["timestamp_start"]:
                clock = datetime.strptime(value, "%H:%M:%S").time()
                stamp = datetime.combine(times[-1].date() if times else today, clock)
                if times and stamp < times[-1]:
                    stamp += timedelta(days=1)
                times.append(stamp)
            log["timestamp_start"] = times
            exportCSV(log, source.stem, str(events))
            exportXES(log, source.stem, str(events))

        fps, video_frames = video_timing(reduced)
        if fps != FPS_RESIZE:
            raise RuntimeError(f"Expected {FPS_RESIZE} FPS after preprocessing, got {fps}.")
        frame_activities = np.repeat(activities, ends - starts)
        # Select the most frequent cluster and activity in each one-second bin.
        # Ties take the first value seen. Keep missing output on the time axis.
        seconds = frame_indices // FPS_RESIZE
        boundaries = np.r_[0, np.flatnonzero(seconds[1:] != seconds[:-1]) + 1, len(seconds)]
        predicted = {
            int(seconds[start]): (int(Counter(partition[start:end]).most_common(1)[0][0]),
                                  Counter(frame_activities[start:end]).most_common(1)[0][0])
            for start, end in zip(boundaries[:-1], boundaries[1:])
        }
        write_csv(output / "prediction_seconds.csv", ("video", "second", "cluster", "activity"), (
            (video_id, second, *predicted.get(second, (-1, "Unassigned")))
            for second in range(math.ceil(video_frames / fps))
        ))
        profile.metadata["status"] = "ok"
    except (Exception, KeyboardInterrupt) as error:
        profile.metadata.update(status="failed", error=str(error))
        raise
    finally:
        profile.save()
        print(f"Profile: {output / 'profile.json'}", flush=True)


def solve4x_truth(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    offset = datetime.fromisoformat(data["offset"].removesuffix("Z"))
    end = datetime.fromisoformat(data["end"].removesuffix("Z"))
    duration = (end - offset).total_seconds()
    if duration <= 0:
        raise ValueError(f"Invalid Solve4X scene duration in {path}.")

    # Sample the annotation timeline on the same 15-FPS grid as the predictions.
    # When annotations overlap, later entries in the source JSON take precedence.
    sample_fps = 15
    activities = ["Not specified"] * math.ceil(duration * sample_fps)
    for process in data["localisations"]:
        for event in process["localisation"]:
            start, stop = (datetime.combine(offset.date(), clock_time.fromisoformat(event[key]))
                           for key in ("tcin", "tcout"))
            first = max(0, math.ceil((start - offset).total_seconds() * sample_fps))
            last = min(len(activities), math.ceil((stop - offset).total_seconds() * sample_fps))
            if first < last:
                activities[first:last] = [event["data"]["activity"]] * (last - first)
    for start in range(0, len(activities), sample_fps):
        activity = Counter(activities[start:start + sample_fps]).most_common(1)[0][0]
        yield (data["scene_name"], start // sample_fps), activity


def load_seconds(paths, *, predictions=False):
    rows = {}
    for path in paths:
        if not predictions and path.suffix.lower() == ".json":
            for key, activity in solve4x_truth(path):
                if key in rows:
                    raise ValueError(f"Duplicate second in {path}: {key}.")
                rows[key] = activity
            continue
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"video", "second", "activity"} | ({"cluster"} if predictions else set())
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} needs these columns: {', '.join(sorted(required))}.")
            for row in reader:
                key = (row["video"].strip(), int(row["second"]))
                if key in rows or not key[0] or key[1] < 0 or not row["activity"].strip():
                    raise ValueError(f"Invalid or duplicate second in {path}: {key}.")
                rows[key] = (row["activity"].strip(), int(row["cluster"])) if predictions else row["activity"].strip()
    return rows


def score(args):
    from sklearn.metrics import (accuracy_score, adjusted_rand_score, precision_recall_fscore_support, rand_score)

    truth = load_seconds(args.truth)
    predicted = load_seconds(args.prediction, predictions=True)
    if not truth or truth.keys() != predicted.keys():
        raise ValueError(f"Expected identical video/second keys. Missing predictions: {len(truth.keys() - predicted.keys())}. "
                         f"Extra predictions: {len(predicted.keys() - truth.keys())}.")
    keys = sorted(truth)
    true_labels = [truth[key] for key in keys]
    predicted_labels = [predicted[key][0] for key in keys]
    clusters = [f"{key[0]}:{predicted[key][1]}" for key in keys]
    precision, recall, _, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=0,
    )
    activities = sorted(set(true_labels) | set(predicted_labels))
    class_precision, class_recall, _, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=activities, zero_division=0,
    )
    results = dict(ri=rand_score(true_labels, clusters), ari=adjusted_rand_score(true_labels, clusters),
                   accuracy=accuracy_score(true_labels, predicted_labels),
                   weighted_precision=precision, weighted_recall=recall)
    report = dict(seconds=len(keys), videos=sorted({key[0] for key in keys}), metrics=results,
                  per_activity={activity: dict(precision=float(p), recall=float(r), support=int(n))
                                for activity, p, r, n in zip(activities, class_precision, class_recall, support)})
    if args.reference:
        reference = TABLE_4[args.reference]
        report.update(reference_dataset=args.reference, table_4_reference=reference,
                      delta={key: results[key] - reference[key] for key in reference})
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Profile the original pipeline for one video.")
    run.add_argument("video", type=Path)
    run.add_argument("--clusters", type=int, required=True, help="Target cluster count from ground truth.")
    run.add_argument("--video-id", help="Scene name used in the one-second output.")
    run.add_argument("--output", type=Path, required=True, help="New output directory.")
    run.add_argument("--image", default="idt-orig:latest")
    run.add_argument("--build", action="store_true", help="Build the IDT image before running it.")
    resume = commands.add_parser("resume", help="Resume labeling from a failed run with saved TW-FINCH results.")
    resume.add_argument("--output", type=Path, required=True, help="Existing run directory.")
    resume.add_argument("--video-id", help="Scene name for runs created before the ID was saved.")
    demo = commands.add_parser("demo", help="Smoke-test export using the author's supplied demo data.")
    demo.add_argument("--output", type=Path, required=True)
    demo.set_defaults(video=None, clusters=None, video_id=None)
    metrics = commands.add_parser("score", help="Score matching video and second rows against ground truth.")
    metrics.add_argument("--truth", type=Path, nargs="+", required=True,
                         help="One-second CSV files or Solve4X video annotation JSON files.")
    metrics.add_argument("--prediction", type=Path, nargs="+", required=True)
    metrics.add_argument("--reference", choices=TABLE_4)
    args = parser.parse_args()
    if args.command == "run" and args.clusters < 2:
        parser.error("--clusters must be at least 2.")
    if args.command == "score":
        score(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
