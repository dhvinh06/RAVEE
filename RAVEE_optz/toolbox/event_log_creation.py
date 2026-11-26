import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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


_INTEGER_EVENT_COLUMNS = ('videoID', 'segment_id', 'cluster_id', 'frame_start',
                          'frame_end', 'task_group_id', '@@index', '@@case_index')


def _numeric_event_fields(dataframe):
    """Type numeric attributes before PM4Py's broad object-to-date inference.

    Also recover the specific epoch-nanosecond encoding in legacy cached CSVs.
    Fractional durations lost by that encoding are reconstructed from real times.
    Never mutate the caller's annotations.
    """
    frame = dataframe.copy(deep=True)
    for column in (*_INTEGER_EVENT_COLUMNS, 'duration_seconds'):
        if column not in frame:
            continue
        values = frame[column]
        legacy = values.notna() & values.astype(str).str.match(
            r'^1970-01-01[ T]00:00:00(?:\.\d{1,9})?(?:\+00:00|Z)?$')
        if legacy.any():
            values = values.astype(object).copy()
            restored = pd.to_datetime(values.loc[legacy], utc=True, format='mixed')
            values.loc[legacy] = restored.dt.as_unit('ns').astype('int64')
            if column == 'duration_seconds':
                # Epoch serialization discards fractional nanoseconds: timestamps
                # retain the actual duration and are the recoverable source.
                begin = pd.to_datetime(frame['timestamp_start'], utc=True, format='mixed')
                finish = pd.to_datetime(frame['timestamp_end'], utc=True, format='mixed')
                values.loc[legacy] = (finish - begin).dt.total_seconds().loc[legacy]
        numeric = pd.to_numeric(values, errors='raise')
        if not np.isfinite(numeric.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite numeric event field: {column}')
        if column in _INTEGER_EVENT_COLUMNS:
            if (numeric.dropna() % 1 != 0).any():
                raise ValueError(f'Non-integer event field: {column}')
            frame[column] = numeric.astype('Int64' if numeric.isna().any() else 'int64')
        else:
            frame[column] = numeric.astype(float)
    return frame


def readDetailedCSV(path):
    """Load current or affected legacy detailed logs without changing the CSV."""
    return _numeric_event_fields(pd.read_csv(path))


def _format_event_log(dataframe):
    original_columns = list(dataframe.columns)
    formatted = pm4py.format_dataframe(
        _numeric_event_fields(dataframe),
        case_id="videoID",
        activity_key="activity",
        timestamp_key="timestamp_start",
    )
    # PM4Py recreates its aliases, moving them behind existing index columns.
    # Keep the established CSV schema/order when re-exporting a cached log.
    return formatted.loc[:, original_columns + [column for column in formatted if column not in original_columns]]


def exportCSV(dataframe, name, outputPath):
    """Export an event dataframe as a PM4Py-formatted CSV file."""
    event_log = _format_event_log(dataframe)
    output_file = os.path.join(outputPath, name + "_eventlog.csv")
    event_log.to_csv(output_file, index=False)
    print(f"CSV event log saved: {output_file}")


def exportXES(dataframe, name, outputPath):
    """Export an event dataframe as a PM4Py XES file."""
    event_log = _format_event_log(dataframe)
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


def extractTaskLog(detailed_df, task_mapping=None, *, confirmed_groups=None):
    """Aggregate adjacent detailed segments into task-level activities.

    Adjacent segments are merged only when they map to the same task and
    their frame ranges touch. Body-posture labels in NON_TASK_ACTIONS are
    retained in the detailed log but omitted from this task-level log.
    With confirmed_groups, use the exact reviewed partition instead of mapping,
    retaining every original segment and preserving explicit task boundaries.
    """
    if detailed_df.empty:
        return createTaskDF()

    if confirmed_groups is not None:
        from toolbox.semantic_aggregation import segment_records, validate_groups
        groups = validate_groups(confirmed_groups, segment_records(detailed_df))
        sources = {int(row['segment_id']): row for _, row in detailed_df.iterrows()}
        task_rows = []
        for number, group in enumerate(groups, 1):
            members = [sources[sid] for sid in group['source_segment_ids']]
            first, last = members[0], members[-1]
            task_rows.append({
                'videoID': int(first['videoID']), 'activity': group['suggested_task_label'],
                'frame_start': int(first['frame_start']), 'frame_end': int(last['frame_end']),
                'timestamp_start': pd.Timestamp(first['timestamp_start']),
                'timestamp_end': pd.Timestamp(last['timestamp_end']),
                'duration_seconds': sum(float(row['duration_seconds']) for row in members),
                'source_segment_ids': ','.join(str(sid) for sid in group['source_segment_ids']),
                'source_activities': ' | '.join(str(row['activity']) for row in members),
                'task_group_id': number, 'group_explanation': group['explanation'],
            })
        return pd.DataFrame(task_rows, columns=(*TASK_COLUMNS, 'task_group_id', 'group_explanation'))

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


def showProcessModel(data_path, output_dir=None, timeout=30):
    """Save PNG/SVG without opening a blocking Matplotlib or Windows viewer.

    Rendering is bounded by ``timeout`` seconds per format. A failed diagram
    must not undo successful CSV/XES exports or stop the remaining pipeline.
    Return the paths actually saved; report discovery/render errors on stderr.
    """
    saved = {}
    try:
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('Render timeout must be a positive finite number')
        from pm4py.visualization.bpmn import visualizer

        source = Path(data_path)
        directory = Path(output_dir) if output_dir is not None else source.parent
        directory.mkdir(parents=True, exist_ok=True)
        log = pm4py.read_xes(str(source))
        process_tree = pm4py.discover_process_tree_inductive(log)
        bpmn_model = pm4py.convert_to_bpmn(process_tree)
        graph = visualizer.apply(bpmn_model)
    except Exception as error:
        print(f'ERROR: Cannot create process model from {data_path}: '
              f'{type(error).__name__}: {error}. CSV/XES files are preserved.', file=sys.stderr)
        return saved

    for image_format in ('png', 'svg'):
        destination = directory / f'{source.stem}_process_model.{image_format}'
        try:
            # Use dot directly so a stalled renderer has a timeout. Argument
            # lists preserve Windows paths containing spaces, apostrophes or &.
            with tempfile.TemporaryDirectory(prefix='ravee_model_', dir=directory) as work:
                temporary = Path(work) / f'model.{image_format}'
                subprocess.run(
                    ['dot', f'-T{image_format}', '-o', str(temporary)],
                    input=graph.source, text=True, encoding='utf-8',
                    capture_output=True, check=True, timeout=timeout,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0,
                )
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError('Graphviz did not produce a nonempty output file')
                os.replace(temporary, destination)
            saved[image_format] = str(destination)
            print(f'Process model saved: {destination}')
        except Exception as error:
            detail = (error.stderr or '').strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
            print(f'ERROR: Process model {image_format.upper()} render failed: '
                  f'{type(error).__name__}: {detail}. '
                  f'Check Graphviz dot on PATH and output permissions: {destination}. '
                  'CSV/XES files are preserved; processing can continue.', file=sys.stderr)
    return saved
