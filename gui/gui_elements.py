"""GUI helpers for RAVEE.

The original cluster-selection dialog is retained. Video labeling is presented
in one Tkinter window with an embedded OpenCV video preview.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue, Empty
from threading import Thread
from label_suggester import suggest_activity  # RAVEE_LABEL_SUGGESTER_INTEGRATED
from batch_label_suggester import suggest_all_activities
from gemini_runtime import redact_text

import cv2
from PIL import Image, ImageTk
from gui.theme import COLORS, apply_theme, fit_window, style_text, workspace_header
from gui.quota_recovery import QuotaRecovery


# Different cluster numbers offered by FINCH
OPTIONS = ["No options available"]

# Number of clusters selected by the user
NUM_CLUSTER = 0

# Labels consumed by toolbox/main.py. Keys are 1-based segment indices.
LABEL = {}
counter = 1


def uncommitted_segments(labels, drafts, count):
    """Zero-based segments whose current draft still needs explicit Commit."""
    return [i for i in range(count)
            if not labels.get(i + 1, '').strip()
            or labels.get(i + 1) != drafts.get(i, labels.get(i + 1, ''))]


def popupmsg(msg, title):
    """Show a simple modal information window."""
    root = tk.Tk()
    apply_theme(root)
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
    apply_theme(root)
    root.title("RAVEE - Select granularity")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="RAVEE   /   Setup", style="Brand.TLabel").pack(anchor="w")
    ttk.Label(frame, text="Choose granularity", style="Title.TLabel").pack(anchor="w", pady=(10, 12))

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
        style="Primary.TButton",
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
    apply_theme(root)
    root.title("RAVEE - Activity label")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text="Activity label:").pack(anchor="w")
    text_box = tk.Text(frame, height=3, width=34, wrap="word")
    style_text(text_box)
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
    ttk.Button(buttons, text="Commit", style="Primary.TButton", command=commit).pack(side="left", expand=True, fill="x")
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
    apply_theme(root)
    root.title("RAVEE - Segment Activity Labeling")
    fit_window(root, 1240, 820, 980, 640)

    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    header = workspace_header(root, "Label video segments",
        "Preview each segment, name the activity, then save your label.", "01  /  Segment labels")
    progress_text = tk.StringVar(value=f"0 / {len(segments)} labeled")
    ttk.Label(header, textvariable=progress_text, style="Muted.TLabel").grid(row=1, column=1, sticky="e")

    body = ttk.Frame(root, padding=(24, 0, 24, 16))
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=3, minsize=380)
    body.columnconfigure(1, weight=2, minsize=370)
    body.rowconfigure(0, weight=1)

    # Left side: embedded video preview and playback controls.
    video_panel = ttk.Frame(body, style="Card.TFrame", padding=16)
    video_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    video_panel.columnconfigure(0, weight=1)
    video_panel.rowconfigure(1, weight=1)
    ttk.Label(video_panel, text="Video preview", style="Section.TLabel").grid(
        row=0, column=0, sticky="w", pady=(0, 12))

    video_label = tk.Label(
        video_panel,
        text="Loading video...",
        bg=COLORS["preview"],
        fg=COLORS["surface"],
        width=1, height=1,
        font=("Segoe UI", 12),
        anchor="center",
    )
    video_label.grid(row=1, column=0, sticky="nsew")

    video_status = tk.StringVar(value="Frame: -")
    ttk.Label(video_panel, textvariable=video_status, style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(8, 2))

    playback_controls = ttk.Frame(video_panel, style="Card.TFrame")
    playback_controls.grid(row=3, column=0, sticky="ew", pady=(8, 0))
    ttk.Label(playback_controls, text="Space  Play / pause     |     Left / right  Browse",
              style="CardMuted.TLabel").pack(anchor="w")

    # Right side: cluster details, label editor and segment overview.
    side = ttk.Frame(body, style="Card.TFrame", padding=16)
    side.grid(row=0, column=1, sticky="nsew")
    side.columnconfigure(0, weight=1)

    cluster_title = tk.StringVar()
    cluster_range = tk.StringVar()
    ttk.Label(side, text="Current segment", style="Section.TLabel").grid(row=0, column=0, sticky="w")
    ttk.Label(side, textvariable=cluster_title, style="Card.TLabel", font=("Segoe UI", 16, "bold")).grid(row=1, column=0, sticky="w", pady=(4, 0))
    ttk.Label(side, textvariable=cluster_range, style="CardMuted.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 12))

    ttk.Label(side, text="Activity label", style="Section.TLabel").grid(row=3, column=0, sticky="w")
    use_batch_button = ttk.Button(side, text="Use batch suggestion", state="disabled")
    use_batch_button.grid(row=3, column=0, sticky="e", pady=(0, 4))
    label_entry = ttk.Entry(side)
    label_entry.grid(row=4, column=0, sticky="new", pady=(5, 8))

    label_status = tk.StringVar(value="Not saved")
    ttk.Label(side, textvariable=label_status, style="CardMuted.TLabel").grid(row=5, column=0, sticky="w", pady=(0, 8))

    action_button = ttk.Button(side, text="Commit & Next", style="Primary.TButton")
    action_button.grid(row=6, column=0, sticky="ew", pady=(0, 6))

    navigation = ttk.Frame(side, style="Card.TFrame")
    navigation.grid(row=7, column=0, sticky="ew", pady=(0, 12))
    navigation.columnconfigure(0, weight=1)
    navigation.columnconfigure(1, weight=1)
    previous_button = ttk.Button(navigation, text="Previous")
    previous_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    play_button = ttk.Button(navigation, text="Play")
    play_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
    suggestion_controls = ttk.Frame(navigation, style="Card.TFrame")
    suggestion_controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    suggestion_controls.columnconfigure(0, weight=1)
    suggestion_controls.columnconfigure(1, weight=1)
    suggest_button = ttk.Button(suggestion_controls, text="Suggest label")
    suggest_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    suggest_all_button = ttk.Button(suggestion_controls, text="Suggest All")
    suggest_all_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    ttk.Label(side, text="Segment overview", style="Section.TLabel").grid(row=8, column=0, sticky="w", pady=(0, 5))
    tree = ttk.Treeview(
        side,
        columns=("cluster", "frames", "status", "batch"),
        show="headings",
        height=11,
        selectmode="browse",
    )
    tree.heading("cluster", text="Cluster")
    tree.heading("frames", text="Frames")
    tree.heading("status", text="Status")
    tree.heading("batch", text="Batch suggestion")
    tree.column("cluster", width=65, anchor="center")
    tree.column("frames", width=90, anchor="center")
    tree.column("status", width=75, anchor="center")
    tree.column("batch", width=155, minwidth=100, anchor="w")
    tree.tag_configure("alternate", background=COLORS['stripe'])
    tree.grid(row=9, column=0, sticky="nsew")
    side.rowconfigure(9, weight=1)

    tree_scroll = ttk.Scrollbar(side, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.grid(row=9, column=1, sticky="ns")

    suggestion_text = tk.StringVar(value="Click Suggest label for an AI suggestion.")
    suggestion_status = ttk.Label(side, textvariable=suggestion_text,
                                  style="CardMuted.TLabel", wraplength=340)
    suggestion_status.grid(row=10, column=0, sticky="w", pady=(6, 0))
    batch_text = tk.StringVar(value="Suggest All proposes labels for every segment in one request.")

    footer = ttk.Frame(root, padding=(24, 0, 24, 16))
    footer.grid(row=2, column=0, sticky="ew")
    footer.columnconfigure(0, weight=1)
    progress_bar = ttk.Progressbar(footer, maximum=len(segments), mode="determinate")
    progress_bar.grid(row=0, column=0, sticky="ew")

    current_index = 0
    current_frame = int(segments[0]["start_frame"])
    playing = False
    after_id = None
    suggestions = {}  # segment index -> proposed text (not committed)
    batch_suggestions = {}  # Separate from individual V3; never written to LABEL.
    batch_pending = False
    suggestion_queue = Queue()
    pending_suggestion = None
    preview_image = None

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
                    batch_suggestions.get(i, ""),
                ),
            )

        current_saved = label_is_saved(current_index)
        label_status.set("Saved" if current_saved else "Not saved")

    def resize_preview(_event=None):
        if preview_image is None:
            return
        image = preview_image.copy()
        # Fit the frame inside the preview without changing its aspect ratio.
        max_width = max(video_label.winfo_width(), 1)
        max_height = max(video_label.winfo_height(), 1)
        image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        video_label.configure(image=photo, text="")
        video_label.image = photo

    def render_frame(frame, frame_number):
        nonlocal preview_image
        preview_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resize_preview()
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
        nonlocal current_frame, preview_image
        seg = segment_at(current_index)
        start_frame = int(seg["start_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            preview_image = None
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

    def update_suggestion_ui():
        busy = pending_suggestion is not None or batch_pending
        suggest_all_button.configure(state="disabled" if busy or recovery.disabled else "normal")
        use_batch_button.configure(state="normal" if current_index in batch_suggestions else "disabled")
        # Share the status area to leave room for the scrollable segment list.
        suggestion_status.configure(textvariable=batch_text if batch_pending or
            (current_index in batch_suggestions and pending_suggestion is None) else suggestion_text)
        if current_index in batch_suggestions:
            batch_text.set(f"Batch suggestion: {batch_suggestions[current_index]} (review before committing)")
        if batch_pending:
            suggest_button.configure(text="Suggest label", state="disabled")
        elif pending_suggestion is not None:
            suggest_button.configure(text="Suggesting...", state="disabled")
            suggestion_text.set(
                "Analyzing this segment..." if pending_suggestion == current_index
                else "AI is analyzing another segment..."
            )
        elif current_index in suggestions:
            suggest_button.configure(text="Use suggestion", state="normal")
            suggestion_text.set(f"AI suggestion: {suggestions[current_index]}")
        elif recovery.disabled:
            suggest_button.configure(text="AI disabled for this review", state="disabled")
            suggestion_text.set("Manual labeling is available. Completed suggestions are preserved.")
        else:
            suggest_button.configure(text="Suggest label", state="normal")
            suggestion_text.set("Click Suggest label to generate a suggestion.")

    def request_suggestion():
        nonlocal pending_suggestion
        if pending_suggestion is not None or batch_pending:
            return
        index = current_index
        if index in suggestions:
            # The user explicitly chooses to put the suggestion in the entry;
            # Commit & Next remains the only action that stores a LABEL.
            label_entry.delete(0, "end")
            label_entry.insert(0, suggestions[index])
            save_current_draft()
            update_progress_and_tree()
            label_entry.focus_set()
            suggestion_text.set("Suggestion inserted. Review and click Commit & Next.")
            suggestion_status.configure(textvariable=suggestion_text)
            return
        if recovery.disabled:
            return

        pending_suggestion = index
        update_suggestion_ui()
        segment = segment_at(index)

        def worker():
            try:
                label = suggest_activity(
                    video_path, segment["start_frame"], segment["end_frame"],
                    on_daily_quota=recovery.handle_quota,
                )
                suggestion_queue.put((index, label, None))
            except Exception as exc:
                suggestion_queue.put((index, None, str(exc)))

        Thread(target=worker, daemon=True).start()

    def request_all_suggestions():
        nonlocal batch_pending
        if batch_pending or pending_suggestion is not None or recovery.disabled:
            return
        save_current_draft()
        batch_pending = True
        update_suggestion_ui()
        batch_text.set(f"Preparing suggestions for all {len(segments)} segments...")
        # Freeze IDs/boundaries on the UI thread; workers never read widgets.
        boundaries = [dict(segment_id=i + 1, start_frame=int(segment['start_frame']),
                           end_frame=int(segment['end_frame'])) for i, segment in enumerate(segments)]

        def worker():
            try:
                labels = suggest_all_activities(video_path, boundaries,
                    on_daily_quota=recovery.handle_quota,
                    progress=lambda message: suggestion_queue.put(('batch_progress', message, None)))
                suggestion_queue.put(('batch', labels, None))
            except Exception as error:
                suggestion_queue.put(('batch', None, redact_text(error, 300)))
        Thread(target=worker, daemon=True).start()

    def use_batch_suggestion():
        if current_index not in batch_suggestions:
            return
        label_entry.delete(0, 'end')
        label_entry.insert(0, batch_suggestions[current_index])
        save_current_draft()
        update_progress_and_tree()
        label_entry.focus_set()
        batch_text.set('Batch suggestion inserted. Review and click Commit & Next.')
        suggestion_status.configure(textvariable=batch_text)

    def poll_suggestions():
        nonlocal pending_suggestion, batch_pending
        try:
            while True:
                index, label, error = suggestion_queue.get_nowait()
                if index == 'batch_progress':
                    batch_text.set(label)
                    continue
                if index == 'batch':
                    batch_pending = False
                    if label is not None:
                        batch_suggestions.update({identifier - 1: value for identifier, value in label.items()})
                        update_progress_and_tree()
                    update_suggestion_ui()
                    suggestion_status.configure(textvariable=batch_text)
                    batch_text.set('Suggest All unavailable; your labels and drafts are preserved. ' + error
                                   if error else f'{len(batch_suggestions)} batch suggestions ready. '
                                   'Select a segment and use its batch suggestion when ready; nothing was committed.')
                    continue
                pending_suggestion = None
                if label:
                    suggestions[index] = label
                update_suggestion_ui()
                if error and index == current_index:
                    suggestion_text.set(
                        "AI unavailable; manual labeling still works. " + error[:160]
                    )
        except Empty:
            pass
        root.after(150, poll_suggestions)

    def go_to_segment(index):
        nonlocal current_index
        if index < 0 or index >= len(segments):
            return

        save_current_draft()
        stop_playback()
        current_index = index
        seg = segment_at(current_index)

        cluster_title.set(f'Segment {current_index + 1} of {len(segments)}')
        cluster_range.set(f'Cluster {seg["cluster_id"]}   /   Frames {seg["start_frame"]} – {seg["end_frame"]}')

        label_entry.delete(0, "end")
        label_entry.insert(0, drafts.get(current_index, LABEL.get(current_index + 1, "")))
        previous_button.configure(state="normal" if current_index > 0 else "disabled")
        action_button.configure(text="Commit & Finish" if current_index == len(segments) - 1 else "Commit & Next")

        tree.selection_set(str(current_index))
        tree.see(str(current_index))
        show_segment_first_frame()
        update_progress_and_tree()
        update_suggestion_ui()
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

        missing = uncommitted_segments(LABEL, drafts, len(segments))
        if missing:
            messagebox.showinfo(
                "Labels remaining",
                "The last segment is saved. Review and commit the remaining label edits.",
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
        missing = uncommitted_segments(LABEL, drafts, len(segments))
        if missing:
            messagebox.showwarning(
                "Labeling incomplete",
                f"There are {len(missing)} unlabeled segments. Use Commit & Next to finish labeling before closing.",
                parent=root,
            )
            return
        root.destroy()

    def show_quota_status(message):
        (batch_text if batch_pending else suggestion_text).set(message)

    recovery = QuotaRecovery(root,
        on_pause=lambda: show_quota_status('Daily quota exhausted. Request paused; manual labeling is available.'),
        on_decision=lambda action: show_quota_status({
            'manual': 'Further AI requests disabled. Continue labeling manually.',
            'cancel': 'Pending AI request cancelled. Your labels and drafts are preserved.',
            'select': 'Resuming the pending AI request with the selected project.'}[action]))
    previous_button.configure(command=go_previous)
    play_button.configure(command=toggle_playback)
    suggest_button.configure(command=request_suggestion)
    suggest_all_button.configure(command=request_all_suggestions)
    use_batch_button.configure(command=use_batch_suggestion)
    action_button.configure(command=commit_and_next)
    video_label.bind("<Configure>", resize_preview)
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
            tags=("alternate",) if i % 2 else (),
            values=(str(seg["cluster_id"]), f'{seg["start_frame"]}-{seg["end_frame"]}', "Pending", ""),
        )

    root.update_idletasks()
    go_to_segment(0)
    root.after(150, poll_suggestions)
    try:
        root.mainloop()
    finally:
        recovery.close()
        stop_playback()
        cap.release()

    # The existing main.py expects a label for every segment.
    missing = [i + 1 for i in uncommitted_segments(LABEL, drafts, len(segments))]
    if missing:
        raise RuntimeError(
            "Labeling window closed before all segments were labeled. "
            f"Missing segment indices: {missing}"
        )

    counter = len(LABEL) + 1
