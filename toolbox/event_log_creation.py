import datetime
import os

import numpy as np
import pandas as pd
import pm4py
import toolbox.preprocessing

from custom_exceptions import processStepNumberException


# Low-level labels that describe body posture rather than a task-level
# process activity. They remain in the detailed event log but are omitted
# from the task-level event log.
NON_TASK_ACTIONS = {
    "look down",
    "bend down",
    "stand up",
}


# Mapping from the labels used in the current RAVEE demo to process tasks.
# Update this dictionary when annotating a different video/domain.
TASK_MAPPING = {
    "crafting the left side": "Assemble left side",
    "crafting left side": "Assemble left side",
    "pick up right side": "Assemble right side",
    "crafting right side": "Assemble right side",
    "finishing right side then go back to a neutral position": "Assemble right side",
    "pick up the behind": "Assemble rear section",
    "crafting the behind": "Assemble rear section",
    "sliding the behind then pick up the roof": "Prepare roof assembly",
    "crafting the roof": "Assemble roof",
    "tightening the roof": "Secure roof",
    "repositioning the box": "Reposition box",
}


DETAILED_COLUMNS = (
    "videoID",
    "segment_id",
    "cluster_id",
    "activity",
    "frame_start",
    "frame_end",
    "timestamp_start",
    "timestamp_end",
    "duration_seconds",
)

TASK_COLUMNS = (
    "videoID",
    "activity",
    "frame_start",
    "frame_end",
    "timestamp_start",
    "timestamp_end",
    "duration_seconds",
    "source_segment_ids",
    "source_activities",
)


def _get_segments(frame_cluster):
    """Return contiguous segments from a 1-D framewise cluster array.

    Each segment has a 1-based segment_id, cluster_id, and inclusive
    frame_start/frame_end indices.
    """
    values = np.asarray(frame_cluster).reshape(-1)
    if values.size == 0:
        return []

    segments = []
    start = 0

    for position in range(1, len(values) + 1):
        at_end = position == len(values)
        changed = not at_end and values[position] != values[start]

        if at_end or changed:
            cluster_value = values[start].item() if hasattr(values[start], "item") else values[start]
            if isinstance(cluster_value, (int, float, np.integer, np.floating)):
                if float(cluster_value).is_integer():
                    cluster_value = int(cluster_value)

            segments.append({
                "segment_id": len(segments) + 1,
                "cluster_id": cluster_value,
                "frame_start": start,
                "frame_end": position - 1,
            })
            start = position

    return segments


def getProcessStepTime(input, index):
    """Return the duration, in rounded seconds, of the indexed segment."""
    fps = float(toolbox.preprocessing.FPS_RESIZE)
    if fps <= 0:
        raise ValueError("FPS_RESIZE must be greater than zero")

    segments = _get_segments(input)
    if index < 0 or index >= len(segments):
        raise IndexError(f"Process-step index out of range: {index}")

    frame_count = segments[index]["frame_end"] - segments[index]["frame_start"] + 1
    return round(frame_count / fps)


def createDF():
    """Create an empty dataframe for detailed event-log records."""
    return pd.DataFrame(columns=DETAILED_COLUMNS)


def createTaskDF():
    """Create an empty dataframe for task-level event-log records."""
    return pd.DataFrame(columns=TASK_COLUMNS)


def getProcessSteps(input, index):
    """Return the number of contiguous process steps in one cluster column."""
    values = np.asarray(input)
    if values.ndim == 1:
        selected = values
    else:
        selected = values[:, index]

    if selected.size == 0:
        return 0

    return 1 + int(np.count_nonzero(selected[1:] != selected[:-1]))


def exportCSV(dataframe, name, outputPath):
    """Export an event dataframe as a PM4Py-formatted CSV file."""
    event_log = pm4py.format_dataframe(
        dataframe,
        case_id="videoID",
        activity_key="activity",
        timestamp_key="timestamp_start",
    )
    output_file = os.path.join(outputPath, name + "_eventlog.csv")
    event_log.to_csv(output_file, index=False)
    print(f"CSV event log saved: {output_file}")


def exportXES(dataframe, name, outputPath):
    """Export an event dataframe as a PM4Py XES file."""
    event_log = pm4py.format_dataframe(
        dataframe,
        case_id="videoID",
        activity_key="activity",
        timestamp_key="timestamp_start",
    )
    output_file = os.path.join(outputPath, name + "_eventlog.xes")
    pm4py.write_xes(event_log, output_file)
    print(f"XES event log saved: {output_file}")


def extractLog(caseID, label, frameCluster, idx_selected):
    """Create the detailed event log, preserving every contiguous segment."""
    cluster_array = np.asarray(frameCluster)
    if cluster_array.ndim == 1:
        selected_clusters = cluster_array
    else:
        selected_clusters = cluster_array[:, idx_selected]

    segments = _get_segments(selected_clusters)

    label_array = np.asarray(label, dtype=object)
    if label_array.ndim != 2 or label_array.shape[1] < 2:
        raise processStepNumberException("Invalid label array; expected segment number and activity label")

    labels_by_id = {}
    for row in label_array:
        try:
            segment_id = int(row[0])
        except (TypeError, ValueError):
            raise processStepNumberException(f"Invalid segment number in label: {row[0]!r}")
        labels_by_id[segment_id] = str(row[1]).strip()

    expected_ids = set(range(1, len(segments) + 1))
    actual_ids = set(labels_by_id)

    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise processStepNumberException(
            f"Labeling incomplete or inconsistent. Missing segment labels: {missing}; unexpected labels: {extra}"
        )

    fps = float(toolbox.preprocessing.FPS_RESIZE)
    if fps <= 0:
        raise ValueError("FPS_RESIZE must be greater than zero")

    # Timestamps are relative to one common base time and derived directly
    # from frame indices. No per-segment rounding is performed.
    base_time = datetime.datetime.now().replace(microsecond=0)
    rows = []

    for segment in segments:
        start_frame = segment["frame_start"]
        end_frame = segment["frame_end"]
        frame_count = end_frame - start_frame + 1

        timestamp_start = base_time + datetime.timedelta(seconds=start_frame / fps)
        timestamp_end = base_time + datetime.timedelta(seconds=(end_frame + 1) / fps)

        rows.append({
            "videoID": int(caseID),
            "segment_id": int(segment["segment_id"]),
            "cluster_id": segment["cluster_id"],
            "activity": labels_by_id[segment["segment_id"]],
            "frame_start": int(start_frame),
            "frame_end": int(end_frame),
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "duration_seconds": frame_count / fps,
        })

    dataframe = pd.DataFrame(rows, columns=DETAILED_COLUMNS)
    print(f"Detailed event log: {len(dataframe)} segments")
    print(dataframe[["segment_id", "cluster_id", "activity", "frame_start", "frame_end"]])
    return dataframe


def extractTaskLog(detailed_df, task_mapping=None):
    """Aggregate adjacent detailed segments into task-level activities.

    Adjacent segments are merged only when they map to the same task and
    their frame ranges touch. Body-posture labels in NON_TASK_ACTIONS are
    retained in the detailed log but omitted from this task-level log.
    """
    if detailed_df.empty:
        return createTaskDF()

    mapping = TASK_MAPPING if task_mapping is None else task_mapping
    mapping = {str(key).strip().lower(): value for key, value in mapping.items()}
    non_task = {value.strip().lower() for value in NON_TASK_ACTIONS}

    task_rows = []

    for _, source in detailed_df.sort_values("segment_id").iterrows():
        source_activity = str(source["activity"]).strip()
        normalized_activity = source_activity.lower()

        if normalized_activity in non_task:
            continue

        task_activity = mapping.get(normalized_activity)
        if task_activity is None:
            # Preserve an unmapped label rather than silently dropping data.
            task_activity = source_activity
            print(
                f"WARNING: No task mapping for '{source_activity}'. "
                "Keeping its original label in the task-level log."
            )

        current = {
            "videoID": int(source["videoID"]),
            "activity": str(task_activity),
            "frame_start": int(source["frame_start"]),
            "frame_end": int(source["frame_end"]),
            "timestamp_start": source["timestamp_start"],
            "timestamp_end": source["timestamp_end"],
            "duration_seconds": float(source["duration_seconds"]),
            "source_segment_ids": str(int(source["segment_id"])),
            "source_activities": source_activity,
        }

        if task_rows:
            previous = task_rows[-1]
            can_merge = (
                    previous["videoID"] == current["videoID"]
                    and previous["activity"] == current["activity"]
                    and previous["frame_end"] + 1 == current["frame_start"]
            )
        else:
            can_merge = False

        if can_merge:
            previous["frame_end"] = current["frame_end"]
            previous["timestamp_end"] = current["timestamp_end"]
            previous["duration_seconds"] += current["duration_seconds"]
            previous["source_segment_ids"] += "," + current["source_segment_ids"]
            previous["source_activities"] += " | " + current["source_activities"]
        else:
            task_rows.append(current)

    task_df = pd.DataFrame(task_rows, columns=TASK_COLUMNS)
    print(f"Task-level event log: {len(task_df)} tasks")
    print(task_df[["activity", "frame_start", "frame_end", "source_segment_ids"]])
    return task_df


def showProcessModel(data_path):
    log = pm4py.read_xes(data_path)
    process_tree = pm4py.discover_process_tree_inductive(log)
    bpmn_model = pm4py.convert_to_bpmn(process_tree)
    pm4py.view_bpmn(bpmn_model)
