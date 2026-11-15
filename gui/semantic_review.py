"""Tkinter task review beside the existing, unchanged segment-labeling GUI."""

import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue, Empty
from threading import Thread

import cv2
from PIL import Image, ImageTk

from toolbox.semantic_aggregation import GroupEditor, ReviewCancelled, segment_records, propose_task_groups


def review_task_groups(video_path, detailed_df, *, initial_groups=None, task_mapping=None, offline=False):
    records = segment_records(detailed_df)
    editor = GroupEditor(records, initial_groups, task_mapping)
    by_id = {row['segment_id']: row for row in records}
    root = tk.Tk()
    root.title('RAVEE - Semantic Event Aggregation')
    root.geometry('1220x850')
    root.minsize(960, 700)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    header = ttk.Frame(root, padding=12)
    header.grid(row=0, column=0, sticky='ew')
    ttk.Label(header, text='Review semantic task groups', font=('Segoe UI', 17, 'bold')).pack(anchor='w')
    ttk.Label(header, text='Original FINCH segments and labels stay unchanged. '
                          'AI proposals require your confirmation.').pack(anchor='w')
    body = ttk.Frame(root, padding=(12, 0, 12, 8))
    body.grid(row=1, column=0, sticky='nsew')
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2)
    body.rowconfigure(0, weight=1)
    tree_frame = ttk.Frame(body)
    tree_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(0, weight=1)
    tree = ttk.Treeview(tree_frame, columns=('ids', 'frames', 'label'), show='headings', selectmode='extended')
    for name, title, width in (('ids', 'Source segments', 130), ('frames', 'Frames', 100),
                                ('label', 'Task label', 260)):
        tree.heading(name, text=title)
        tree.column(name, width=width)
    tree.grid(row=0, column=0, sticky='nsew')
    scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
    scrollbar.grid(row=0, column=1, sticky='ns')
    tree.configure(yscrollcommand=scrollbar.set)
    side = ttk.Frame(body)
    side.grid(row=0, column=1, sticky='nsew')
    side.columnconfigure(0, weight=1)
    preview = tk.Label(side, bg='#17191d', fg='white', text='Select a task to preview its video')
    preview.grid(row=0, column=0, sticky='nsew')
    side.rowconfigure(0, weight=1)
    play_button = ttk.Button(side, text='Play selected task')
    play_button.grid(row=1, column=0, sticky='ew', pady=4)
    details = tk.Text(side, height=11, wrap='word', state='disabled')
    details.grid(row=2, column=0, sticky='ew')
    form = ttk.Frame(root, padding=(12, 0, 12, 6))
    form.grid(row=2, column=0, sticky='ew')
    form.columnconfigure(1, weight=1)
    ttk.Label(form, text='Task label').grid(row=0, column=0, sticky='w', padx=(0, 8))
    label = ttk.Entry(form)
    label.grid(row=0, column=1, sticky='ew')
    ttk.Label(form, text='Grouping explanation').grid(row=1, column=0, sticky='w', padx=(0, 8))
    reason = ttk.Entry(form)
    reason.grid(row=1, column=1, sticky='ew', pady=5)
    controls = ttk.Frame(root, padding=(12, 0, 12, 6))
    controls.grid(row=3, column=0, sticky='ew')
    apply_button = ttk.Button(controls, text='Apply label / explanation')
    apply_button.pack(side='left')
    merge_button = ttk.Button(controls, text='Merge selected adjacent groups')
    merge_button.pack(side='left', padx=6)
    ttk.Label(controls, text='Split after source ID:').pack(side='left')
    split_at = ttk.Combobox(controls, state='readonly', width=7)
    split_at.pack(side='left', padx=4)
    split_button = ttk.Button(controls, text='Split')
    split_button.pack(side='left')
    undo_button = ttk.Button(controls, text='Undo')
    undo_button.pack(side='left', padx=6)
    reset_button = ttk.Button(controls, text='Keep original segments')
    reset_button.pack(side='left')
    footer = ttk.Frame(root, padding=(12, 0, 12, 12))
    footer.grid(row=4, column=0, sticky='ew')
    footer.columnconfigure(0, weight=1)
    status = tk.StringVar(value='Review groups manually, or explicitly request Gemini proposals.')
    ttk.Label(footer, textvariable=status, wraplength=700).grid(row=0, column=0, sticky='w')
    suggest_button = ttk.Button(footer, text='Suggest task groups (Gemini)')
    suggest_button.grid(row=1, column=0, sticky='w', pady=(6, 0))
    confirm_button = ttk.Button(footer, text='Confirm task groups')
    confirm_button.grid(row=1, column=1, padx=6)
    cancel_button = ttk.Button(footer, text='Cancel review')
    cancel_button.grid(row=1, column=2)
    if offline:
        suggest_button.configure(state='disabled')
        status.set('Offline review: Gemini is disabled. Edit, merge and split groups, then confirm.')

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    delay = max(15, round(1000 / fps)) if fps and fps > 0 else 66
    result, playback, polling = None, None, None
    queue = Queue()
    pending = False
    mutation_buttons = (apply_button, merge_button, split_button, undo_button, reset_button, confirm_button)

    def selection():
        return sorted(int(item) for item in tree.selection())

    def stop():
        nonlocal playback
        if playback is not None:
            root.after_cancel(playback)
            playback = None
        play_button.configure(text='Play selected task')

    def paint(frame, number):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max(preview.winfo_width(), 400), 280), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        preview.configure(image=photo, text='')
        preview.image = photo

    def show_selection(_event=None):
        stop()
        selected = selection()
        if not selected:
            return
        group = editor.groups[selected[0]]
        label.delete(0, 'end')
        label.insert(0, group['suggested_task_label'])
        reason.delete(0, 'end')
        reason.insert(0, group['explanation'])
        choices = group['source_segment_ids'][:-1]
        split_at.configure(values=choices)
        split_at.set(str(choices[len(choices) // 2]) if choices else '')
        lines = [group['explanation'], '']
        for sid in group['source_segment_ids']:
            row = by_id[sid]
            lines.append(f'Segment {sid}: {row["activity"]}\nFrames {row["start_frame"]}-{row["end_frame"]}\n'
                         f'{row["timestamp_start"]} — {row["timestamp_end"]}\n')
        details.configure(state='normal')
        details.delete('1.0', 'end')
        details.insert('1.0', '\n'.join(lines))
        details.configure(state='disabled')
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, group['start_frame'])
            ok, frame = cap.read()
            if ok:
                paint(frame, group['start_frame'])
                return
        preview.configure(image='', text='Video preview unavailable; original labels/timestamps remain reviewable')
        preview.image = None

    def refresh(index=0):
        stop()
        tree.delete(*tree.get_children())
        for i, group in enumerate(editor.groups):
            tree.insert('', 'end', iid=str(i), values=(
                ','.join(map(str, group['source_segment_ids'])),
                f'{group["start_frame"]}-{group["end_frame"]}', group['suggested_task_label']))
        index = min(index, len(editor.groups) - 1)
        tree.selection_set(str(index))
        tree.see(str(index))
        show_selection()

    def act(operation):
        if pending:
            return
        try:
            operation()
            refresh(selection()[0] if selection() else 0)
            status.set(f'{len(editor.groups)} task groups. Review names before confirming.')
        except (ValueError, IndexError) as error:
            messagebox.showwarning('Review task groups', str(error), parent=root)

    def apply_edit():
        indices = selection()
        if len(indices) != 1:
            raise ValueError('Select one group to edit its label and explanation')
        editor.rename(indices[0], label.get(), reason.get())

    def play_next(end):
        nonlocal playback
        number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = cap.read() if number <= end else (False, None)
        if not ok:
            playback = None
            play_button.configure(text='Play selected task')
            return
        paint(frame, number)
        playback = root.after(delay, lambda: play_next(end))

    def play():
        if playback is not None:
            stop()
            return
        indices = selection()
        if not indices or not cap.isOpened():
            return
        group = editor.groups[indices[0]]
        cap.set(cv2.CAP_PROP_POS_FRAMES, group['start_frame'])
        play_button.configure(text='Stop preview')
        play_next(group['end_frame'])

    def lock(value):
        nonlocal pending
        pending = value
        for button in mutation_buttons:
            button.configure(state='disabled' if value else 'normal')
        label.configure(state='disabled' if value else 'normal')
        reason.configure(state='disabled' if value else 'normal')
        suggest_button.configure(state='disabled' if value or offline else 'normal')

    def suggest():
        if offline or pending:
            return
        if not messagebox.askyesno('Gemini task proposals',
                'Send existing labels, timestamps and sampled video context to the configured Gemini model? '
                'API charges may apply. Your current edits will be replaced by proposals; Undo can restore them.',
                parent=root):
            return
        lock(True)
        status.set('Gemini is proposing adjacent task groups. You must still review and confirm.')

        def worker():
            try:
                queue.put((propose_task_groups(video_path, detailed_df.copy(deep=True)), None))
            except Exception as error:
                queue.put((None, str(error)))
        Thread(target=worker, daemon=True).start()

    def poll():
        nonlocal polling
        try:
            groups, error = queue.get_nowait()
            lock(False)
            if error:
                status.set('AI unavailable; current groups preserved. Manual review still works. ' + error[:220])
            else:
                editor.replace(groups)
                refresh()
                status.set('AI proposals loaded, not confirmed. Review shared objectives, retries and unclear actions.')
        except Empty:
            pass
        polling = root.after(150, poll)

    def confirm():
        nonlocal result
        if pending:
            return
        try:
            if len(selection()) == 1:
                apply_edit()
            groups = editor.confirm()
        except ValueError as error:
            messagebox.showwarning('Invalid task group', str(error), parent=root)
            return
        if messagebox.askyesno('Confirm task groups',
                f'Use these {len(groups)} reviewed activities for the task-level log and process model? '
                'The detailed log remains unchanged.', parent=root):
            result = groups
            root.quit()

    def cancel():
        if messagebox.askyesno('Cancel review', 'Cancel task review? Detailed segment annotations are already saved.', parent=root):
            root.quit()

    tree.bind('<<TreeviewSelect>>', show_selection)
    play_button.configure(command=play)
    apply_button.configure(command=lambda: act(apply_edit))
    merge_button.configure(command=lambda: act(lambda: editor.merge(selection())))
    split_button.configure(command=lambda: act(lambda: editor.split(selection()[0], int(split_at.get()))))
    undo_button.configure(command=lambda: act(editor.undo))
    reset_button.configure(command=lambda: act(editor.reset))
    suggest_button.configure(command=suggest)
    confirm_button.configure(command=confirm)
    cancel_button.configure(command=cancel)
    root.protocol('WM_DELETE_WINDOW', cancel)
    try:
        root.update_idletasks()
        refresh()
        polling = root.after(150, poll)
        root.mainloop()
    finally:
        stop()
        if polling is not None:
            root.after_cancel(polling)
        cap.release()
        root.destroy()
    if result is None:
        raise ReviewCancelled('Task review cancelled. Detailed annotations are saved; no new task groups were confirmed.')
    return result
