import os
import argparse
import time
import sys

# Legacy helpers import their siblings by module name. Keep those imports working
# both for the direct IntelliJ script and package imports, without editing the
# fingerprinted preprocessing/encoding modules.
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Support direct IntelliJ / python toolbox/main.py execution from any cwd.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip

from finch.finch import FINCH
from gui import gui_elements
from toolbox.preprocessing import reduce_video_size, FPS_RESIZE, WIDTH_RESIZE, HEIGHT_RESIZE
from toolbox.feature_extraction import extract_framewise_features
from toolbox.pipeline_cache import prepare_features, cached_finch, inspect_video_cache
from toolbox.semantic_aggregation import review_semantic_task_log
from toolbox.display_video_snippets import run_snippets
from toolbox.event_log_creation import (
    createDF,
    extractLog,
    extractTaskLog,
    exportCSV,
    exportXES,
    showProcessModel,
)


# -----------------------------------------------------------------------------
# Directories for runtime
# -----------------------------------------------------------------------------
# Resolve runtime paths from this file so launching from another directory works.
projectDirectory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory of IDT feature extraction
targetIDT = os.path.join(projectDirectory, "IDT")

# Video input directory
raw_video_input = os.path.join(projectDirectory, "video_input")

# Processed video output
processedVideoOutput = os.path.join(targetIDT, "data")

# IDT output
idtOutput = os.path.join(targetIDT, "features")

# Fisher Vector output
fvOutput = os.path.join(projectDirectory, "finch", "input")

# FINCH output
finchOutput = os.path.join(projectDirectory, "finch", "output")

# Demo files
demoPath = os.path.join(projectDirectory, "DemoFiles")
# Event-log output for normal (non-demo) processing
eventlogOutput = os.path.join(projectDirectory, "EventLog_Output")

# Event-log output for DemoFiles
eventlogOutputDemo = demoPath

# Temporary directory to cache results if necessary
tempDirectoryPath = os.path.join(projectDirectory, "temp")


# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------
# Amount of clusters to be retrieved by FINCH. None lets FINCH determine it.
required_cluster = None

# Case ID counter
CASEID = 1

# Demo mode: True runs the precomputed DemoFiles dataset.
DEMO = False 



# -----------------------------------------------------------------------------
# Helper functions from the original RAVEE main.py
# -----------------------------------------------------------------------------
def concatenate(video_clip_paths, output_path, method="compose"):
    """Concatenate video clips and write the result to output_path."""
    clips = [VideoFileClip(path) for path in video_clip_paths]
    final_clip = concatenate_videoclips(clips, method=method)
    final_clip.write_videofile(output_path)


def getFrameCount(video_clip_path):
    """Return the number of frames in a video file."""
    cap = cv2.VideoCapture(video_clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_clip_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def reset_label_gui():
    """Clear labels between videos so labels do not leak into the next case."""
    gui_elements.LABEL.clear()
    gui_elements.counter = 1


def export_two_event_logs(detailed_df, task_df, output_dir, base_name):
    """Export detailed and task-level event logs as CSV and XES."""
    os.makedirs(output_dir, exist_ok=True)

    exportCSV(detailed_df, f"{base_name}_detailed", output_dir)
    exportXES(detailed_df, f"{base_name}_detailed", output_dir)

    exportCSV(task_df, f"{base_name}_task", output_dir)
    exportXES(task_df, f"{base_name}_task", output_dir)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def prepare_cases(*, videos=None, allow_idt=False, adopt_preprocessing=False,
                  adopt_idt=False, rebuild=False, progress=None):
    """One processing path for the GUI worker and CLI; never opens review windows."""
    cache, cases = prepare_features(
        projectDirectory, os.path.join(tempDirectoryPath, 'pipeline_cache'),
        reduce_video_size, (FPS_RESIZE, WIDTH_RESIZE, HEIGHT_RESIZE),
        extract_framewise_features, videos=videos, allow_idt=allow_idt,
        adopt_preprocessing=adopt_preprocessing, adopt_idt=adopt_idt,
        rebuild=rebuild, progress=progress)
    partitions = {}
    for key, case in cases.items():
        partitions[key] = cached_finch(cache, case, required_cluster, FINCH,
                                       projectDirectory, progress=progress)
    return cache, cases, partitions


def main(*, allow_idt=False, adopt_preprocessing=False, adopt_idt=False, rebuild=False,
         videos=None, prepared=None, prepare_only=False):
    global CASEID

    if DEMO:
        # ---------------------------------------------------------------------
        # DEMO MODE
        # Start from a precomputed feature set and run the labeling process.
        # ---------------------------------------------------------------------
        video_path = os.path.join(demoPath, "scan_video_reduced.avi")

        c = np.genfromtxt(
            os.path.join(demoPath, "scan_video_reduced_c.csv"),
            delimiter=",",
        )
        num_cluster = np.genfromtxt(
            os.path.join(demoPath, "scan_video_reduced_num_cluster.csv"),
            delimiter=",",
        )

        # Let the user select the clustering granularity and label segments.
        idx_selected = run_snippets(video_path, c, num_cluster)
        label = np.array(list(gui_elements.LABEL.items()), dtype=object)
        print("Label", label)

        # Reset GUI labels after collecting the annotations.
        reset_label_gui()

        caseID = 1

        # Create the detailed log: preserve every contiguous FINCH segment.
        df_detailed = extractLog(caseID, label, c, idx_selected)

        # Review semantic task groups without changing the original segments.
        df_task = review_semantic_task_log(
            video_path, df_detailed, eventlogOutputDemo, 'scan_video_reduced',
        )

        # Export both representations.
        export_two_event_logs(
            df_detailed,
            df_task,
            eventlogOutputDemo,
            "scan_video_reduced",
        )

        # Discover and visualize the process model from task-level activities.
        task_xes_path = os.path.join(
            eventlogOutputDemo,
            "scan_video_reduced_task_eventlog.xes",
        )
        showProcessModel(task_xes_path)

    else:
        # ---------------------------------------------------------------------
        # NON-DEMO MODE
        # Process the selected source, or the original directory in CLI mode.
        # ---------------------------------------------------------------------
        # The selector has already run these exact stages in its worker. The
        # CLI calls the same helper; neither path duplicates processing.
        cache, cases, partitions = prepared if prepared is not None else prepare_cases(
            videos=videos, allow_idt=allow_idt, adopt_preprocessing=adopt_preprocessing,
            adopt_idt=adopt_idt, rebuild=rebuild)
        if prepare_only:
            return cache, cases, partitions
        os.makedirs(finchOutput, exist_ok=True)
        os.makedirs(eventlogOutput, exist_ok=True)

        # Separate aggregate dataframes for detailed and task-level logs.
        eventlog_detailed = createDF()
        eventlog_task = pd.DataFrame()

        # ---------------------------------------------------------------------
        # FINCH clustering, manual labeling, and event-log extraction.
        # ---------------------------------------------------------------------
        for count, (key, case) in enumerate(cases.items()):
            start = time.time()

            c, num_clust, req_c = partitions[key]

            print(
                "Time Elapsed: {:2.2f} seconds".format(
                    time.time() - start
                )
            )

            # Save FINCH results.
            name = case['video_name']
            keyNr = str(case['case_id'])

            print("Writing back the results on the provided path ...")

            np.savetxt(
                os.path.join(finchOutput, name + "-c.csv"),
                c,
                delimiter=",",
                fmt="%d",
            )
            np.savetxt(
                os.path.join(finchOutput, name + "-num_cluster.csv"),
                np.array(num_clust),
                delimiter=",",
                fmt="%d",
            )

            if req_c is not None:
                np.savetxt(
                    os.path.join(finchOutput, name + "-req_c.csv"),
                    req_c,
                    delimiter=",",
                    fmt="%d",
                )

            # Use this input's exact cached video, never a stale case-ID match.
            video_path = case['video_path']
            video_path_name = os.path.basename(video_path)
            print(video_path)

            # Run the original RAVEE manual labeling interface.
            if req_c is None:
                idx_selected = run_snippets(video_path, c, num_clust)
            else:
                idx_selected = run_snippets(
                    video_path,
                    c,
                    num_clust,
                    req_c,
                )

            label = np.array(
                list(gui_elements.LABEL.items()),
                dtype=object,
            )
            print("Label", label)

            # Reset labels before processing the next video.
            reset_label_gui()

            caseID = int(keyNr)

            # Extract both event-log representations for this video.
            df_detailed = extractLog(
                caseID,
                label,
                c,
                idx_selected,
            )
            df_task = review_semantic_task_log(
                video_path, df_detailed, eventlogOutput, f'{caseID}_{name}',
            )

            eventlog_detailed = pd.concat(
                [eventlog_detailed, df_detailed],
                ignore_index=True,
            )
            eventlog_task = pd.concat(
                [eventlog_task, df_task],
                ignore_index=True,
            )

            print(f"Completed labeling and event-log extraction: {video_path_name}")

            if (count + 1) % 10 == 0:
                print(f"{count + 1} files were completed.")

        # Export aggregate event logs for all processed videos.
        export_two_event_logs(
            eventlog_detailed,
            eventlog_task,
            eventlogOutput,
            "all_videos",
        )

        # Discover the process model using the task-level event log.
        showProcessModel(
            os.path.join(
                eventlogOutput,
                "all_videos_task_eventlog.xes",
            )
        )

        # Preserve completed stage caches and all original input/IDT files.


def launch_interactive(*, adopt_preprocessing=False, adopt_idt=False, rebuild=False):
    """Keep Tk on the main thread; only prepare stages run in the worker."""
    from gui.video_selection import select_video_to_process

    def inspect(source):
        return inspect_video_cache(projectDirectory,
            os.path.join(tempDirectoryPath, 'pipeline_cache'), source,
            (FPS_RESIZE, WIDTH_RESIZE, HEIGHT_RESIZE), required_cluster)

    def process(source, progress):
        # Clicking Process Video explicitly authorizes missing stages, including
        # Docker extraction, for this one path. Selecting/browsing never does.
        return prepare_cases(videos=[source], allow_idt=True,
            adopt_preprocessing=adopt_preprocessing, adopt_idt=adopt_idt,
            rebuild=rebuild, progress=progress)

    selected = select_video_to_process(raw_video_input, inspect, process)
    if selected is None:
        return
    source, prepared = selected
    return main(videos=[source], prepared=prepared)


def entrypoint(argv=None):
    parser = argparse.ArgumentParser(description="RAVEE video selection and validated stage caches")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--gui', action='store_true', help='Open the startup video selector (default without arguments)')
    mode.add_argument('--no-gui', '--cli', dest='no_gui', action='store_true',
                      help='Bypass the selector; retain the original review workflow')
    parser.add_argument('--video', metavar='PATH', help='Process only this video in CLI mode, including an external path')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Run cached stages through FINCH without any GUI or review/export')
    parser.add_argument('--allow-idt', action='store_true',
                        help='Allow missing preprocessing / Docker extraction in CLI mode')
    parser.add_argument('--adopt-existing-preprocessing', action='store_true',
                        help='Validate and cache matching legacy IDT/data videos')
    parser.add_argument('--adopt-existing-idt', action='store_true',
                        help='Validate and cache matching legacy raw trajectory .npy files')
    parser.add_argument('--rebuild-idt', action='store_true',
                        help='Force image rebuild if Docker extraction is needed')
    args = parser.parse_args(argv)
    if args.gui and (args.video or args.prepare_only):
        parser.error('--video and --prepare-only select CLI mode and cannot be used with --gui')
    if DEMO and args.prepare_only:
        parser.error('--prepare-only requires DEMO = False')
    options = dict(adopt_preprocessing=args.adopt_existing_preprocessing,
                   adopt_idt=args.adopt_existing_idt, rebuild=args.rebuild_idt)
    # Legacy processing flags retain their original directory-processing meaning.
    # --gui can explicitly combine them with the new selector.
    legacy_cli = args.allow_idt or any(options.values())
    if DEMO or args.no_gui or args.video or args.prepare_only or (legacy_cli and not args.gui):
        return main(videos=[args.video] if args.video else None,
                    allow_idt=args.allow_idt, prepare_only=args.prepare_only, **options)
    return launch_interactive(**options)


if __name__ == "__main__":
    entrypoint()
