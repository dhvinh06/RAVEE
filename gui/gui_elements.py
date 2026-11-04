"""GUI helpers for RAVEE.

The original cluster-selection dialog is retained. Video labeling is presented
in one Tkinter window with an embedded OpenCV video preview.
"""

import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk


# Different cluster numbers offered by FINCH
OPTIONS = ["No options available"]

# Number of clusters selected by the user
NUM_CLUSTER = 0

# Labels consumed by toolbox/main.py. Keys are 1-based segment indices.
LABEL = {}
counter = 1


def popupmsg(msg, title):
    """Show a simple modal information window."""
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    root.transient()

    label = ttk.Label(root, text=msg, padding=16, wraplength=420)
    label.pack(fill="x")
    button = ttk.Button(root, text="OK", command=root.destroy)
    button.pack(pady=(0, 12))

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    root.mainloop()


def inputboxList():
    """Let the user choose one of the available cluster granularities."""
    global NUM_CLUSTER

    root = tk.Tk()
    root.title("RAVEE - Select granularity")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text="Select the number of clusters to inspect:",
    ).pack(anchor="w", pady=(0, 8))

    display = tk.StringVar(root, value=str(OPTIONS[0]))

    dropdown = ttk.OptionMenu(
        frame,
        display,
        str(OPTIONS[0]),
        *[str(x) for x in OPTIONS]
    )
    dropdown.pack(fill="x", pady=(0, 12))

    def confirm():
        global NUM_CLUSTER  # THÊM DÒNG NÀY

        NUM_CLUSTER = display.get()

        print("Selected cluster count:", NUM_CLUSTER)

        root.destroy()

    ttk.Button(
        frame,
        text="Continue",
        command=confirm
    ).pack(fill="x")

    root.update_idletasks()

    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2

    root.geometry(f"+{x}+{y}")
    root.mainloop()

def setOptions(options):
    """Set the cluster-count options displayed in inputboxList."""
    global OPTIONS
    OPTIONS = list(options) if options is not None else ["No options available"]
    if not OPTIONS:
        OPTIONS = ["No options available"]


def inputboxLine():
    """Legacy single-label dialog, retained for compatibility."""
    global LABEL, counter

    root = tk.Tk()
    root.title("RAVEE - Activity label")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Activity label:").pack(anchor="w")
    text_box = tk.Text(frame, height=3, width=34, wrap="word")
    text_box.pack(fill="both", expand=True, pady=8)

    def commit():
        value = text_box.get("1.0", "end-1c").strip()
        if not value:
            messagebox.showwarning("Missing label", "Enter an activity label first.", parent=root)
            return
        LABEL[counter] = value
        counter += 1
        text_box.delete("1.0", "end")

    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="Commit", command=commit).pack(side="left", expand=True, fill="x")
    ttk.Button(buttons, text="Exit", command=root.destroy).pack(side="left", expand=True, fill="x", padx=(8, 0))
    root.mainloop()


def label_video_clusters(video_path, segments):
    """Display and label all video segments in one Tkinter window.

    Parameters
    ----------
    video_path : str
        Path to the video being labeled.
    segments : list[dict]
        Each item must contain ``cluster_id``, ``start_frame`` and ``end_frame``.

    Labels are stored in the global LABEL dictionary with 1-based keys so the
    existing main.py / extractLog workflow can consume them unchanged.
    """
    global LABEL, counter

    if not segments:
        raise ValueError("No video segments were supplied for labeling.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    playback_delay = max(15, int(1000 / fps))

    # Start a fresh labeling session. Labels are keyed by segment order,
    # not by the cluster ID, because the same cluster may occur more than once.
    LABEL.clear()
    counter = 1

    root = tk.Tk()
    root.title("RAVEE - Video activity labeling")
    root.geometry("1180x760")
    root.minsize(900, 600)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    header = ttk.Frame(root, padding=(14, 12, 14, 8))
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    ttk.Label(
        header,
        text="RAVEE  |  Video Activity Labeling",
        font=("Segoe UI", 17, "bold"),
    ).grid(row=0, column=0, sticky="w")
    progress_text = tk.StringVar(value=f"0 / {len(segments)} labeled")
    ttk.Label(header, textvariable=progress_text).grid(row=0, column=1, sticky="e")

    body = ttk.Frame(root, padding=(14, 4, 14, 12))
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=4)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    # Left side: embedded video preview and playback controls.
    video_panel = ttk.LabelFrame(body, text="Video preview", padding=10)
    video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    video_panel.columnconfigure(0, weight=1)
    video_panel.rowconfigure(0, weight=1)

    video_label = tk.Label(
        video_panel,
        text="Loading video...",
        bg="#17191d",
        fg="white",
        font=("Segoe UI", 12),
        anchor="center",
    )
    video_label.grid(row=0, column=0, sticky="nsew")

    video_status = tk.StringVar(value="Frame: -")
    ttk.Label(video_panel, textvariable=video_status).grid(row=1, column=0, sticky="w", pady=(8, 2))

    playback_controls = ttk.Frame(video_panel)
    playback_controls.grid(row=2, column=0, pady=(4, 0))

    # Right side: cluster details, label editor and segment overview.
    side = ttk.Frame(body)
    side.grid(row=0, column=1, sticky="nsew")
    side.columnconfigure(0, weight=1)
    side.rowconfigure(4, weight=1)

    cluster_title = tk.StringVar()
    cluster_range = tk.StringVar()
    ttk.Label(side, text="Current segment", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Label(side, textvariable=cluster_title, font=("Segoe UI", 22, "bold")).grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Label(side, textvariable=cluster_range).grid(row=2, column=0, sticky="w", pady=(0, 12))

    ttk.Label(side, text="Activity label", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w")
    label_entry = ttk.Entry(side)
    label_entry.grid(row=4, column=0, sticky="new", pady=(5, 8))

    label_status = tk.StringVar(value="Not saved")
    ttk.Label(side, textvariable=label_status).grid(row=5, column=0, sticky="w", pady=(0, 8))

    action_button = ttk.Button(side, text="Commit & Next")
    action_button.grid(row=6, column=0, sticky="ew", pady=(0, 6))

    navigation = ttk.Frame(side)
    navigation.grid(row=7, column=0, sticky="ew", pady=(0, 12))
    navigation.columnconfigure(0, weight=1)
    navigation.columnconfigure(1, weight=1)
    previous_button = ttk.Button(navigation, text="Previous")
    previous_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    play_button = ttk.Button(navigation, text="Play")
    play_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    ttk.Label(side, text="Segment overview", font=("Segoe UI", 10, "bold")).grid(row=8, column=0, sticky="w", pady=(0, 5))
    tree = ttk.Treeview(
        side,
        columns=("cluster", "frames", "status"),
        show="headings",
        height=11,
        selectmode="browse",
    )
    tree.heading("cluster", text="Cluster")
    tree.heading("frames", text="Frames")
    tree.heading("status", text="Status")
    tree.column("cluster", width=65, anchor="center")
    tree.column("frames", width=90, anchor="center")
    tree.column("status", width=105, anchor="center")
    tree.grid(row=9, column=0, sticky="nsew")
    side.rowconfigure(9, weight=1)

    tree_scroll = ttk.Scrollbar(side, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.grid(row=9, column=1, sticky="ns")

    footer = ttk.Frame(root, padding=(14, 0, 14, 12))
    footer.grid(row=2, column=0, sticky="ew")
    footer.columnconfigure(0, weight=1)
    progress_bar = ttk.Progressbar(footer, maximum=len(segments), mode="determinate")
    progress_bar.grid(row=0, column=0, sticky="ew")

    current_index = 0
    current_frame = int(segments[0]["start_frame"])
    playing = False
    after_id = None

    def segment_at(index):
        return segments[index]

    def label_is_saved(index):
        value = LABEL.get(index + 1, "")
        return bool(value.strip()) and value == drafts.get(index, value)

    drafts = {i: LABEL.get(i + 1, "") for i in range(len(segments))}

    def update_progress_and_tree():
        saved_count = sum(
            1 for i in range(len(segments))
            if bool(LABEL.get(i + 1, "").strip()) and LABEL.get(i + 1) == drafts.get(i, LABEL.get(i + 1, ""))
        )
        progress_text.set(f"{saved_count} / {len(segments)} labeled")
        progress_bar.configure(value=saved_count)

        for i, seg in enumerate(segments):
            saved = bool(LABEL.get(i + 1, "").strip()) and LABEL.get(i + 1) == drafts.get(i, LABEL.get(i + 1, ""))
            status = "Saved" if saved else ("Editing" if i == current_index else "Pending")
            tree.item(
                str(i),
                values=(
                    str(seg["cluster_id"]),
                    f'{seg["start_frame"]}-{seg["end_frame"]}',
                    status,
                ),
            )

        current_saved = label_is_saved(current_index)
        label_status.set("Saved" if current_saved else "Not saved")

    def render_frame(frame, frame_number):
        frame = frame.copy()
        cv2.putText(
            frame,
            f"Frame {frame_number}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Fit the frame inside the preview without changing its aspect ratio.
        max_width = max(video_label.winfo_width(), 800)
        max_height = max(video_label.winfo_height(), 440)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        video_label.configure(image=photo, text="")
        video_label.image = photo
        video_status.set(f"Frame: {frame_number}")

    def stop_playback():
        nonlocal playing, after_id
        playing = False
        if after_id is not None:
            try:
                root.after_cancel(after_id)
            except tk.TclError:
                pass
            after_id = None
        try:
            play_button.configure(text="Play")
        except tk.TclError:
            # The window may already have been destroyed when mainloop exits.
            pass

    def show_segment_first_frame():
        nonlocal current_frame
        seg = segment_at(current_index)
        start_frame = int(seg["start_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            video_label.configure(image="", text=f"Could not read frame {start_frame}")
            video_label.image = None
            video_status.set(f"Frame: {start_frame} (unavailable)")
            current_frame = start_frame
            return
        current_frame = start_frame
        render_frame(frame, current_frame)

    def play_next_frame():
        nonlocal after_id, current_frame, playing
        if not playing:
            return

        seg = segment_at(current_index)
        next_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if next_frame_number > int(seg["end_frame"]):
            stop_playback()
            return

        ret, frame = cap.read()
        if not ret:
            stop_playback()
            return

        current_frame = next_frame_number
        render_frame(frame, current_frame)
        after_id = root.after(playback_delay, play_next_frame)

    def toggle_playback():
        nonlocal playing
        if playing:
            stop_playback()
            return

        seg = segment_at(current_index)
        if current_frame >= int(seg["end_frame"]):
            show_segment_first_frame()

        playing = True
        play_button.configure(text="Pause")
        play_next_frame()

    def save_current_draft():
        drafts[current_index] = label_entry.get().strip()

    def go_to_segment(index):
        nonlocal current_index
        if index < 0 or index >= len(segments):
            return

        save_current_draft()
        stop_playback()
        current_index = index
        seg = segment_at(current_index)

        cluster_title.set(f'Cluster {seg["cluster_id"]}  ·  Segment {current_index + 1}/{len(segments)}')
        cluster_range.set(f'Frames {seg["start_frame"]} – {seg["end_frame"]}')

        label_entry.delete(0, "end")
        label_entry.insert(0, drafts.get(current_index, LABEL.get(current_index + 1, "")))
        previous_button.configure(state="normal" if current_index > 0 else "disabled")
        action_button.configure(text="Commit & Finish" if current_index == len(segments) - 1 else "Commit & Next")

        tree.selection_set(str(current_index))
        tree.see(str(current_index))
        show_segment_first_frame()
        update_progress_and_tree()
        label_entry.focus_set()

    def commit_and_next():
        global counter
        value = label_entry.get().strip()
        if not value:
            messagebox.showwarning(
                "Missing activity label",
                "Enter an activity label before continuing.",
                parent=root,
            )
            label_entry.focus_set()
            return

        drafts[current_index] = value
        LABEL[current_index + 1] = value
        counter = len(LABEL) + 1
        update_progress_and_tree()

        if current_index < len(segments) - 1:
            go_to_segment(current_index + 1)
            return

        missing = [i for i in range(len(segments)) if not LABEL.get(i + 1, "").strip()]
        if missing:
            messagebox.showinfo(
                "Labels remaining",
                "The last segment is saved. The GUI will move to the first unlabeled segment.",
                parent=root,
            )
            go_to_segment(missing[0])
            return

        messagebox.showinfo(
            "Labeling complete",
            f"All {len(segments)} video segments have been labeled.",
            parent=root,
        )
        root.destroy()

    def go_previous():
        if current_index > 0:
            go_to_segment(current_index - 1)

    def on_tree_double_click(_event=None):
        selected = tree.selection()
        if selected:
            try:
                go_to_segment(int(selected[0]))
            except (ValueError, IndexError):
                pass

    def on_entry_change(_event=None):
        save_current_draft()
        update_progress_and_tree()

    def on_space(event):
        if root.focus_get() == label_entry:
            return None
        toggle_playback()
        return "break"

    def on_left(_event=None):
        if root.focus_get() == label_entry:
            return None
        go_previous()
        return "break"

    def on_right(_event=None):
        if root.focus_get() == label_entry:
            return None
        # Arrow-right previews the next segment; it does not commit a label.
        if current_index < len(segments) - 1:
            go_to_segment(current_index + 1)
        return "break"

    def on_enter(_event=None):
        commit_and_next()
        return "break"

    def on_close():
        save_current_draft()
        missing = [
            i for i in range(len(segments))
            if not LABEL.get(i + 1, "").strip()
               or LABEL.get(i + 1) != drafts.get(i, LABEL.get(i + 1, ""))
        ]
        if missing:
            messagebox.showwarning(
                "Labeling incomplete",
                f"There are {len(missing)} unlabeled segments. Use Commit & Next to finish labeling before closing.",
                parent=root,
            )
            return
        root.destroy()

    previous_button.configure(command=go_previous)
    play_button.configure(command=toggle_playback)
    action_button.configure(command=commit_and_next)
    tree.bind("<Double-1>", on_tree_double_click)
    label_entry.bind("<KeyRelease>", on_entry_change)
    root.bind("<space>", on_space)
    root.bind("<Left>", on_left)
    root.bind("<Right>", on_right)
    root.bind("<Return>", on_enter)
    root.protocol("WM_DELETE_WINDOW", on_close)

    # Populate the segment overview before entering the event loop.
    for i, seg in enumerate(segments):
        tree.insert(
            "",
            "end",
            iid=str(i),
            values=(str(seg["cluster_id"]), f'{seg["start_frame"]}-{seg["end_frame"]}', "Pending"),
        )

    root.update_idletasks()
    go_to_segment(0)
    root.mainloop()

    stop_playback()
    cap.release()

    # The existing main.py expects a label for every segment.
    missing = [i + 1 for i in range(len(segments)) if not LABEL.get(i + 1, "").strip()]
    if missing:
        raise RuntimeError(
            "Labeling window closed before all segments were labeled. "
            f"Missing segment indices: {missing}"
        )

    counter = len(LABEL) + 1
