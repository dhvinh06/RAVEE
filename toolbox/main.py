import os
import argparse
import time

import cv2
import numpy as np
import pandas as pd
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip

from finch.finch import FINCH
from gui import gui_elements
from preprocessing import reduce_video_size, FPS_RESIZE, WIDTH_RESIZE, HEIGHT_RESIZE
from feature_extraction import extract_framewise_features
from pipeline_cache import prepare_features, cached_finch
from toolbox.semantic_aggregation import review_semantic_task_log
from display_video_snippets import run_snippets
from event_log_creation import (
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
def main(*, allow_idt=False, adopt_preprocessing=False, adopt_idt=False, rebuild=False):
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
        # Process all videos in video_input.
        # ---------------------------------------------------------------------
        os.makedirs(processedVideoOutput, exist_ok=True)
        os.makedirs(idtOutput, exist_ok=True)
        os.makedirs(fvOutput, exist_ok=True)
        os.makedirs(finchOutput, exist_ok=True)
        os.makedirs(eventlogOutput, exist_ok=True)

        # Keep video identity explicit. CSV names identify descriptor channels;
        # their named rows contain one global FV per video, not framewise data.
        cache, cases = prepare_features(
            projectDirectory,
            os.path.join(tempDirectoryPath, "pipeline_cache"),
            reduce_video_size,
            (FPS_RESIZE, WIDTH_RESIZE, HEIGHT_RESIZE),
            extract_framewise_features,
            allow_idt=allow_idt,
            adopt_preprocessing=adopt_preprocessing,
            adopt_idt=adopt_idt,
            rebuild=rebuild,
        )

        # Separate aggregate dataframes for detailed and task-level logs.
        eventlog_detailed = createDF()
        eventlog_task = pd.DataFrame()

        # ---------------------------------------------------------------------
        # FINCH clustering, manual labeling, and event-log extraction.
        # ---------------------------------------------------------------------
        for count, (key, case) in enumerate(cases.items()):
            start = time.time()

            c, num_clust, req_c = cached_finch(
                cache, case, required_cluster, FINCH, projectDirectory,
            )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAVEE with validated per-input stage caches")
    parser.add_argument('--allow-idt', action='store_true',
                        help='Allow Docker extraction when a raw IDT cache is missing')
    parser.add_argument('--adopt-existing-preprocessing', action='store_true',
                        help='Validate and cache matching legacy IDT/data videos')
    parser.add_argument('--adopt-existing-idt', action='store_true',
                        help='Validate and cache matching legacy raw trajectory .npy files')
    parser.add_argument('--rebuild-idt', action='store_true',
                        help='Force image rebuild if Docker extraction is needed')
    args = parser.parse_args()
    main(allow_idt=args.allow_idt,
         adopt_preprocessing=args.adopt_existing_preprocessing,
         adopt_idt=args.adopt_existing_idt, rebuild=args.rebuild_idt)
