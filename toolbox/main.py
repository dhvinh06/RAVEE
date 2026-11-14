import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip

from finch.finch import FINCH
from gui import gui_elements
from preprocessing import reduce_video_size
from feature_extraction import extract_framewise_features
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
# Current working directory. Keep the same working-directory setting that
# was used by the original RAVEE run, because its paths are relative to it.
currentDirectory = os.getcwd()

# System-specific separator
seperator = str(os.path.sep)

# Directory of IDT feature extraction
# Keep the original repository's directory layout assumptions.
targetIDT = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "IDT")
)

# Video input directory
raw_video_input = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "video_input")
)

# Processed video output
processedVideoOutput = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "IDT", "data")
)

# IDT output
idtOutput = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "IDT", "features")
)

# Fisher Vector output
fvOutput = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "finch", "input")
)

# FINCH output
finchOutput = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "finch", "output")
)

# Demo files
demoPath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "DemoFiles"
)
# Event-log output for normal (non-demo) processing
eventlogOutput = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "EventLog_Output")
)

# Event-log output for DemoFiles
eventlogOutputDemo = demoPath

# Temporary directory to cache results if necessary
tempDirectoryPath = os.path.abspath(
    os.path.join(currentDirectory, os.pardir, "temp")
)


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
def main():
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

        # Create the task-level log by mapping and merging adjacent segments.
        df_task = extractTaskLog(df_detailed)

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

        # Preprocess all input videos.
        input_videos = sorted(os.listdir(raw_video_input))

        for count, value in enumerate(input_videos):
            file_path = os.path.join(raw_video_input, value)
            video_name = os.path.splitext(os.path.basename(file_path))[0]

            caseID = CASEID
            CASEID += 1

            # As in the original RAVEE implementation, resize/downsample
            # videos before extracting Improved Dense Trajectories.
            reduce_video_size(
                file_path,
                processedVideoOutput,
                caseID,
                video_name,
            )

            print(f"Completed preprocessing: {value}")

            if (count + 1) % 10 == 0:
                print(f"{count + 1} files were completed.")

        # ---------------------------------------------------------------------
        # Improved Dense Trajectories (IDT)
        # ---------------------------------------------------------------------
        original_cwd = os.getcwd()
        try:
            os.chdir(targetIDT)
            subprocess.run("docker build -t idt .", shell=True, check=True)
            subprocess.run(
                "docker run -v "
                + targetIDT
                + seperator
                + "features:/densetrack/features idt",
                shell=True,
                check=True,
                )
        finally:
            os.chdir(original_cwd)

        # ---------------------------------------------------------------------
        # Feature extraction: compute framewise Fisher Vectors.
        # ---------------------------------------------------------------------
        compFV = {}

        for count, value in enumerate(sorted(os.listdir(idtOutput))):
            file_path = os.path.join(idtOutput, value)
            nameFV = os.path.splitext(os.path.basename(file_path))[0]

            print("NameFV", nameFV)

            caseNr = int(nameFV.split("_")[0])
            print("CaseNr", caseNr)

            videoNameList = nameFV.split("_")[1:-1]
            print("VideoName", videoNameList)

            videoName = "_".join(videoNameList)
            key = str(caseNr) + "-" + videoName
            print("key", key)

            features = extract_framewise_features(file_path, fvOutput)
            compFV[key] = features

            print(f"Completed feature extraction: {value}")

            if (count + 1) % 10 == 0:
                print(f"{count + 1} files were completed.")

        # Separate aggregate dataframes for detailed and task-level logs.
        eventlog_detailed = createDF()
        eventlog_task = pd.DataFrame()

        # ---------------------------------------------------------------------
        # FINCH clustering, manual labeling, and event-log extraction.
        # ---------------------------------------------------------------------
        for count, (key, value) in enumerate(compFV.items()):
            start = time.time()

            if required_cluster is not None:
                c, num_clust, req_c = FINCH(
                    value,
                    req_clust=required_cluster,
                    tw_finch=True,
                )
            else:
                c, num_clust, req_c = FINCH(
                    value,
                    req_clust=None,
                    tw_finch=True,
                )

            print(
                "Time Elapsed: {:2.2f} seconds".format(
                    time.time() - start
                )
            )

            # Save FINCH results.
            name = key.split("-")[1]
            keyNr = key.split("-")[0]

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

            # Find the preprocessed video corresponding to this case ID.
            video_path_name = None
            for processed_name in os.listdir(processedVideoOutput):
                key_video_name = processed_name.split("_")[0]
                if keyNr == key_video_name:
                    video_path_name = processed_name
                    break

            if video_path_name is None:
                raise FileNotFoundError(
                    f"Could not find processed video for case ID {keyNr}"
                )

            video_path = os.path.join(
                processedVideoOutput,
                video_path_name,
            )
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
            df_task = extractTaskLog(df_detailed)

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

        # ---------------------------------------------------------------------
        # Cleanup, preserving the original RAVEE behavior.
        # ---------------------------------------------------------------------
        for directory in (idtOutput, fvOutput, processedVideoOutput):
            if not os.path.isdir(directory):
                continue

            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as error:
                    print(
                        "Failed to delete %s. Reason: %s"
                        % (file_path, error)
                    )


if __name__ == "__main__":
    main()
