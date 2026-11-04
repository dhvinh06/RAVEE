"""Prepare video segments from FINCH clustering and launch the labeling GUI."""

from collections import Counter

import numpy as np

from gui import gui_elements
from custom_exceptions import idxSelectionError


def run_snippets(video_path, c, num_clust, req_c=None):
    """Select a clustering granularity, create contiguous segments, and label them.

    The clustering logic and returned ``idx_selected`` are kept compatible with
    the existing RAVEE main.py workflow. Each contiguous run of frames is a
    separate segment, even when the same cluster ID occurs more than once.
    """
    ges_cluster = np.array([0])
    idx_selected = "Fehler"

    if req_c is None:
        # Let the user choose a prepared clustering granularity.
        gui_elements.setOptions(num_clust)
        gui_elements.inputboxList()
        num_cluster = gui_elements.NUM_CLUSTER
        print("DEBUG - Selected cluster count:", repr(num_cluster))
        print("DEBUG - Available options:", gui_elements.OPTIONS)

        if num_cluster == 0 or num_cluster == "No options available":
            raise idxSelectionError("No clustering granularity was selected.")

        for i in range(c.shape[1]):
            templist = np.array([c[j][i] for j in range(c.shape[0])])
            num_diff_clust = len(Counter(templist).keys())
            if num_diff_clust == int(float(num_cluster)):
                idx_selected = i
                ges_cluster = templist
                break
    else:
        # Use the clustering result passed by the caller.
        ges_cluster = np.asarray(req_c)
        req_c_sorted = np.sort(ges_cluster, axis=-1)
        idx_selected = req_c_sorted[-1] + 1

    if idx_selected == "Fehler":
        raise idxSelectionError(
            "Could not find a clustering result matching the selected cluster count."
        )

    if ges_cluster.size == 0:
        raise idxSelectionError("The selected clustering result contains no frames.")

    # Build a list of contiguous segments. This preserves the original logic:
    # a new segment ends whenever the cluster ID changes.
    cluster_assignment = []
    for i in range(ges_cluster.shape[0] - 1):
        cluster = ges_cluster[i]
        if ges_cluster[i] != ges_cluster[i + 1]:
            cluster_assignment.append([cluster, i])

        if i == int(ges_cluster.shape[0] - 2):
            last_cluster = ges_cluster[i + 1]
            cluster_assignment.append([last_cluster, i + 1])

    # A one-frame input has no loop iterations, so add its sole segment.
    if ges_cluster.shape[0] == 1:
        cluster_assignment.append([ges_cluster[0], 0])

    print(cluster_assignment)

    segments = []
    first_frame = 0
    for segment_index, (cluster_id, last_frame) in enumerate(cluster_assignment):
        last_frame = int(last_frame)
        segment = {
            "segment_index": segment_index,
            "cluster_id": cluster_id,
            "start_frame": first_frame,
            "end_frame": last_frame,
        }
        segments.append(segment)
        print(
            "Cluster Nr:", cluster_id,
            "Segment:", segment_index + 1,
            "1.Frame:", first_frame,
            "letzter Frame:", last_frame,
                        )
        first_frame = last_frame + 1

    gui_elements.label_video_clusters(video_path, segments)
    return idx_selected
