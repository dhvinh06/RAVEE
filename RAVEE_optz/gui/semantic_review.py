"""Human review of semantic task groups above the original FINCH segments."""

import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue, Empty
from threading import Thread

import cv2
from PIL import Image, ImageTk
from gui.theme import COLORS, apply_theme, fit_window, style_text, workspace_header
from gui.video_preview import resize_preview_image
from gui.quota_recovery import QuotaRecovery
from gemini_runtime import redact_text
from ai_providers import provider_status
from gui.ai_upload import UploadConsent
from gui.lifecycle import WindowLifetime, WorkerInbox, AIJob, run_ai_job, new_window

from toolbox.semantic_aggregation import GroupEditor, ReviewCancelled, segment_records, propose_task_groups


def review_task_groups(video_path, detailed_df, *, initial_groups=None, task_mapping=None, offline=False,
                       refresh_suggestions=False, run_metadata=None):
    records = segment_records(detailed_df)
    editor = GroupEditor(records, initial_groups, task_mapping)
    by_id = {row['segment_id']: row for row in records}
    root = new_window()
    lifetime = WindowLifetime(root)
    apply_theme(root)
    root.title('RAVEE - Semantic Task Aggregation')
    fit_window(root, 1400, 960, 1020, 700)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    header = workspace_header(root, 'Review task groups',
        'Group adjacent actions by their objective. Review each task before confirming.',
        '02  /  Semantic tasks')
    group_count = lifetime.variable(tk.StringVar)
    ttk.Label(header, textvariable=group_count, style='Muted.TLabel').grid(row=1, column=1, sticky='e')
    body = ttk.Frame(root, padding=(24, 0, 24, 12))
    body.grid(row=1, column=0, sticky='nsew')
    body.columnconfigure(0, weight=1)
    body.rowconfigure(0, weight=1)
    panes = tk.PanedWindow(body, orient=tk.HORIZONTAL, background=COLORS['background'],
                           sashwidth=10, sashrelief='flat', borderwidth=0, opaqueresize=True)
    panes.grid(row=0, column=0, sticky='nsew')
    tree_frame = ttk.Frame(panes, style="Card.TFrame", padding=12)
    panes.add(tree_frame, minsize=300, stretch='always')
    tree_frame.columnconfigure(0, weight=1)
    tree_frame.rowconfigure(1, weight=1)
    ttk.Label(tree_frame, text='Task groups', style='Section.TLabel').grid(
        row=0, column=0, sticky='w', pady=(0, 12))
    tree = ttk.Treeview(tree_frame, columns=('ids', 'frames', 'label'), show='headings', selectmode='extended')
    for name, title, width in (('ids', 'Segments', 95), ('frames', 'Frames', 100),
                                ('label', 'Task label', 260)):
        tree.heading(name, text=title)
        tree.column(name, width=width, minwidth=65, stretch=(name == 'label'))
    tree.tag_configure('alternate', background=COLORS['stripe'])
    tree.grid(row=1, column=0, sticky='nsew')
    scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
    scrollbar.grid(row=1, column=1, sticky='ns')
    tree.configure(yscrollcommand=scrollbar.set)
    ttk.Label(tree_frame, text='Source segment details', style='Section.TLabel').grid(
        row=2, column=0, sticky='w', pady=(12, 6))
    details = tk.Text(tree_frame, height=6, width=1, wrap='word', state='disabled')
    style_text(details, readonly=True)
    details.grid(row=3, column=0, sticky='ew')
    details_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=details.yview)
    details_scroll.grid(row=3, column=1, sticky='ns')
    details.configure(yscrollcommand=details_scroll.set)
    side = ttk.Frame(panes, style="Card.TFrame", padding=12)
    panes.add(side, minsize=420, stretch='always')
    side.columnconfigure(0, weight=1)
    ttk.Label(side, text='Task preview', style='Section.TLabel').grid(
        row=0, column=0, sticky='w', pady=(0, 12))
    preview = tk.Label(side, bg=COLORS['preview'], fg=COLORS['surface'], width=1, height=1,
                       borderwidth=0, highlightthickness=0, padx=0, pady=0,
                       wraplength=350, text='Select a task to preview its video')
    preview.grid(row=1, column=0, sticky='nsew')
    side.rowconfigure(1, weight=1)
    play_button = ttk.Button(side, text='Play selected task')
    play_button.grid(row=2, column=0, sticky='ew', pady=(8, 8))
    form = ttk.Frame(side, style='Card.TFrame')
    form.grid(row=3, column=0, sticky='ew')
    form.columnconfigure(1, weight=1)
    ttk.Label(form, text='Task label', style='Card.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 8))
    label = ttk.Entry(form)
    label.grid(row=0, column=1, sticky='ew')
    ttk.Label(form, text='Grouping explanation', style='Card.TLabel').grid(
        row=1, column=0, columnspan=2, sticky='w', pady=(8, 4))
    reason = tk.Text(form, height=2, width=1, wrap='word')
    style_text(reason)
    reason.grid(row=2, column=0, columnspan=2, sticky='ew')
    controls = ttk.Frame(root, padding=(24, 0, 24, 12))
    controls.grid(row=2, column=0, sticky='ew')
    edit_actions = ttk.Frame(controls)
    edit_actions.pack(side='left')
    apply_button = ttk.Button(edit_actions, text='Apply edits')
    apply_button.pack(side='left')
    merge_button = ttk.Button(edit_actions, text='Merge adjacent groups')
    merge_button.pack(side='left', padx=(8, 0))
    split_actions = ttk.Frame(controls)
    split_actions.pack(side='left', padx=(16, 0))
    ttk.Label(split_actions, text='Split after segment').pack(side='left')
    split_at = ttk.Combobox(split_actions, state='readonly', width=5)
    split_at.pack(side='left', padx=6)
    split_button = ttk.Button(split_actions, text='Split group')
    split_button.pack(side='left')
    undo_button = ttk.Button(controls, text='Undo')
    undo_button.pack(side='left', padx=(8, 0))
    reset_button = ttk.Button(controls, text='Reset groups')
    reset_button.pack(side='right')
    footer = ttk.Frame(root, padding=(24, 0, 24, 16))
    footer.grid(row=3, column=0, sticky='ew')
    footer.columnconfigure(0, weight=1)
    status = lifetime.variable(tk.StringVar, value='Review groups manually, or request AI suggestions.')
    ai_status = ttk.Label(footer, text=provider_status('aggregation'), style='Muted.TLabel')
    ai_status.grid(row=2, column=0, columnspan=2, sticky='w')
    upload_consent = UploadConsent(root, timers=lifetime)
    status_label = ttk.Label(footer, textvariable=status, style='Muted.TLabel', wraplength=950)
    status_label.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 12))
    footer.bind('<Configure>', lambda event: status_label.configure(wraplength=max(200, event.width - 48)))
    ai_actions = ttk.Frame(footer)
    ai_actions.grid(row=1, column=0, sticky='w')
    suggest_button = ttk.Button(ai_actions, text='Suggest task groups')
    suggest_button.pack(side='left')
    refresh_button = ttk.Button(ai_actions, text='Refresh AI suggestions')
    refresh_button.pack(side='left', padx=(8, 0))
    review_actions = ttk.Frame(footer)
    review_actions.grid(row=1, column=1, sticky='e')
    cancel_button = ttk.Button(review_actions, text='Cancel review')
    cancel_button.pack(side='left', padx=(0, 8))
    confirm_button = ttk.Button(review_actions, text='Confirm task groups', style='Primary.TButton')
    confirm_button.pack(side='left')
    if offline:
        suggest_button.configure(state='disabled')
        refresh_button.configure(state='disabled')
        status.set('Offline review: AI requests are disabled. Edit, merge and split groups, then confirm.')

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    delay = max(15, round(1000 / fps)) if fps and fps > 0 else 66
    result, playback, polling = None, None, None
    preview_image = None
    resize_timer = None
    restore_timer = None
    queue = WorkerInbox()
    pending = False
    boundary_sources = []
    if refresh_suggestions and not offline:
        status.set('Refresh selected: the next AI request rechecks every boundary. No API call until you request it.')
    mutation_buttons = (apply_button, merge_button, split_button, undo_button, reset_button, confirm_button)

    def selection():
        return sorted(int(item) for item in tree.selection())

    def stop():
        nonlocal playback
        if playback is not None:
            lifetime.cancel(playback)
            playback = None
        if not lifetime.destroyed:
            play_button.configure(text='Play selected task')

    def resize_preview(_event=None):
        nonlocal resize_timer
        if lifetime.closed:
            return
        if resize_timer is not None:
            lifetime.cancel(resize_timer)
            resize_timer = None
        if preview_image is None:
            return
        image = resize_preview_image(preview_image, preview.winfo_width(), preview.winfo_height())
        photo = lifetime.image(ImageTk.PhotoImage(image, master=root))
        preview.configure(image=photo, text='')
        preview.image = photo

    def request_resize(_event=None):
        nonlocal resize_timer
        if lifetime.closed:
            return
        if resize_timer is not None:
            lifetime.cancel(resize_timer)
        resize_timer = lifetime.after(40, resize_preview)

    def paint(frame, number):
        nonlocal preview_image
        preview_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        resize_preview()

    def show_selection(_event=None):
        nonlocal preview_image
        stop()
        selected = selection()
        if not selected:
            return
        group = editor.groups[selected[0]]
        label.delete(0, 'end')
        label.insert(0, group['suggested_task_label'])
        reason.delete('1.0', 'end')
        reason.insert('1.0', group['explanation'])
        choices = group['source_segment_ids'][:-1]
        split_at.configure(values=choices)
        split_at.set(str(choices[len(choices) // 2]) if choices else '')
        lines = []
        for evaluation in boundary_sources:
            if (evaluation['left_segment_id'] in group['source_segment_ids']
                    or evaluation['right_segment_id'] in group['source_segment_ids']):
                source = evaluation['source']
                source = 'OLD cached fallback — NOT fresh' if source == 'fallback' else source
                lines.append(f'Boundary {evaluation["left_segment_id"]} -> '
                             f'{evaluation["right_segment_id"]}: {source}')
                decision = evaluation.get('decision') or {}
                for field in ('objective_before', 'objective_after', 'objective_transition',
                              'objective_scope', 'task_boundary', 'operation_relation',
                              'transition_evidence', 'evidence'):
                    if field in decision:
                        lines.append(f'  {field}: {redact_text(decision[field], 2000)}')
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
        preview_image = None
        preview.configure(image='', text='Video preview unavailable; original labels/timestamps remain reviewable')
        preview.image = None

    def refresh(index=0):
        stop()
        tree.delete(*tree.get_children())
        group_count.set(f'{len(records)} segments  /  {len(editor.groups)} tasks')
        for i, group in enumerate(editor.groups):
            tree.insert('', 'end', iid=str(i), tags=('alternate',) if i % 2 else (), values=(
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
        editor.rename(indices[0], label.get(), reason.get('1.0', 'end-1c'))

    def play_next(end):
        nonlocal playback
        if lifetime.closed:
            return
        number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = cap.read() if number <= end else (False, None)
        if not ok:
            playback = None
            play_button.configure(text='Play selected task')
            return
        paint(frame, number)
        playback = lifetime.after(delay, lambda: play_next(end))

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
        ai_disabled = value or offline or request_inflight or recovery.disabled
        suggest_button.configure(state='disabled' if ai_disabled else 'normal')
        refresh_button.configure(state='disabled' if ai_disabled else 'normal')

    def paused():
        lock(False)  # Manual review stays usable while the worker waits.
        status.set('Daily quota exhausted. Request paused; labels, edits and selection are preserved.')

    request_inflight = False
    recovered_proposals = None
    request_interruptions = 0
    recovery = QuotaRecovery(root, timers=lifetime, on_pause=paused, on_resume=lambda: lock(True),
        on_decision=lambda action: status.set({
            'manual': 'Further AI requests disabled for this review. Continue editing manually.',
            'cancel': 'Pending AI request cancelled. Current review is preserved.',
            'select': 'Resuming only the pending request with the selected project.'}[action]))

    def load_recovered():
        nonlocal recovered_proposals, boundary_sources
        if recovered_proposals is None or not messagebox.askyesno('Load recovered proposals',
                'Replace the current task groups with the recovered proposals? '
                'Apply any pending text edits first. Undo restores the previous groups.', parent=root):
            return
        payload, detail = recovered_proposals
        # Save pending form content too; Undo restores this exact review state.
        checkpoint = (selection(), label.get(), reason.get('1.0', 'end-1c'), boundary_sources,
                      dict(run_metadata) if run_metadata is not None else None)
        editor.replace(payload)
        recovered_checkpoints[len(editor.history)] = checkpoint
        boundary_sources = detail['boundary_sources']
        record_metadata(detail)
        recovered_proposals = None
        suggest_button.configure(text='Suggest task groups')
        refresh()
        status.set('Recovered proposals loaded, not confirmed. Undo restores your previous review.')

    recovered_checkpoints = {}

    def undo_review():
        nonlocal boundary_sources, restore_timer
        if pending:
            return
        checkpoint = recovered_checkpoints.pop(len(editor.history), None)
        editor.undo()
        if checkpoint:
            boundary_sources = checkpoint[3]
            if run_metadata is not None:
                run_metadata.clear()
                run_metadata.update(checkpoint[4])
        refresh(checkpoint[0][0] if checkpoint and checkpoint[0] else 0)
        if checkpoint:
            indices, text, explanation = checkpoint[:3]
            tree.selection_set(tuple(str(index) for index in indices))
            def restore_form():
                nonlocal restore_timer
                restore_timer = None
                if selection() == indices:
                    label.delete(0, 'end')
                    label.insert(0, text)
                    reason.delete('1.0', 'end')
                    reason.insert('1.0', explanation)
            if restore_timer is not None:
                lifetime.cancel(restore_timer)
            restore_form()
            # Tk queues <<TreeviewSelect>>; restore after that event too, so
            # selection notifications cannot discard the restored pending text.
            restore_timer = lifetime.after_idle(restore_form)

    def record_metadata(detail):
        if run_metadata is not None:
            run_metadata['ai_provenance'] = detail.get('provenance', {})
            run_metadata['api_calls'] = detail.get('api_calls', 0)

    def suggest(force_refresh=False):
        nonlocal refresh_suggestions, request_inflight, request_interruptions
        if lifetime.closed or offline or pending or request_inflight or recovery.disabled:
            return
        if recovered_proposals is not None and not force_refresh:
            load_recovered()
            return
        refresh_requested = force_refresh or refresh_suggestions
        if not messagebox.askyesno(provider_status('aggregation') + ' task proposals',
                ('Re-evaluate all boundaries, bypassing this case\'s AI boundary cache? '
                 'Valid old results will be labeled as fallbacks if refresh fails. ' if refresh_requested else '') +
                'Send existing labels, timestamps and video context to ' + provider_status('aggregation') + '? '
                'API charges may apply. Your current edits will be replaced by proposals; Undo can restore them.',
                parent=root):
            return
        refresh_suggestions = False  # CLI arms one explicit request; Refresh always forces a new one.
        request_inflight = True
        request_interruptions = recovery.interruptions
        lock(True)
        ai_status.configure(text=provider_status('aggregation'))
        status.set(provider_status('aggregation') + ' is proposing adjacent task groups. Review and confirm intermediate objectives.')

        # Freeze all request inputs/options on the UI thread. The thread target
        # is module-level and receives no widget, Variable or UI controller.
        job = AIJob(propose_task_groups, (video_path, detailed_df.copy(deep=True)),
                    dict(refresh_suggestions=refresh_requested,
                         on_daily_quota=recovery.handle_quota, on_upload=upload_consent.authorize),
                    'result', 'error', progress_tag='progress', with_stats=True)
        Thread(target=run_ai_job, args=(job, queue), daemon=True).start()

    def poll():
        nonlocal polling, boundary_sources, request_inflight, recovered_proposals
        if lifetime.closed:
            return
        try:
            while True:
                kind, payload, detail = queue.get_nowait()
                if kind == 'progress':
                    status.set(payload)
                    continue
                request_inflight = False
                lock(False)
                if kind == 'error':
                    status.set('AI unavailable; current groups preserved. Manual review still works. ' + redact_text(detail, 220))
                else:
                    if detail.get('error_kind') in ('daily_quota', 'provider_quota'):
                        # Even partial proposals must not replace human review
                        # after Cancel/manual. Fresh results remain cached.
                        quota_name = 'Daily' if detail.get('error_kind') == 'daily_quota' else 'TwelveLabs'
                        status.set(quota_name + ' quota request stopped; current review preserved. ' +
                                   ('Further AI requests disabled.' if recovery.disabled else
                                    'A later explicit request can reuse completed cached work.'))
                        continue
                    if detail['total'] > 0 and detail['unresolved'] == detail['total']:
                        status.set('AI unavailable; current groups preserved. All boundaries remain '
                                   f"unresolved for manual review ({detail.get('error_message') or detail['error']}).")
                        continue
                    if recovery.interruptions > request_interruptions:
                        recovered_proposals = (payload, detail)
                        suggest_button.configure(text='Load recovered proposals')
                        status.set('Pending AI work recovered. Current review is preserved; '
                                   'use Load recovered proposals when ready.')
                        continue
                    boundary_sources = detail['boundary_sources']
                    editor.replace(payload)
                    record_metadata(detail)
                    refresh()
                    status.set(f"Proposals ready, not confirmed: {detail['api_calls']} API calls; "
                               f"{detail['fresh']} fresh; {detail['cache_hits']} reused "
                               f"({detail['fallback']} OLD fallbacks); {detail['unresolved']} unresolved. "
                               f"{(detail.get('error_message') or detail['error']) + '; ' if detail['error'] else ''}"
                               'Review, merge or split manually before confirming.')
                    if detail.get('warnings'):
                        status.set(status.get() + ' Review warning: ' + ' '.join(detail['warnings']))
        except Empty:
            pass
        polling = lifetime.after(150, poll)

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
            lifetime.request_close()

    def cancel():
        if messagebox.askyesno('Cancel review', 'Cancel task review? Detailed segment annotations are already saved.', parent=root):
            lifetime.request_close()

    tree.bind('<<TreeviewSelect>>', show_selection)
    preview.bind('<Configure>', request_resize)
    play_button.configure(command=play)
    apply_button.configure(command=lambda: act(apply_edit))
    merge_button.configure(command=lambda: act(lambda: editor.merge(selection())))
    split_button.configure(command=lambda: act(lambda: editor.split(selection()[0], int(split_at.get()))))
    undo_button.configure(command=undo_review)
    reset_button.configure(command=lambda: act(editor.reset))
    suggest_button.configure(command=suggest)
    refresh_button.configure(command=lambda: suggest(force_refresh=True))
    confirm_button.configure(command=confirm)
    cancel_button.configure(command=cancel)
    root.protocol('WM_DELETE_WINDOW', cancel)
    try:
        root.update_idletasks()
        # Give the video most of the initial width. Subsequent sash/window
        # changes are controlled by the user and the paned widget itself.
        panes.sash_place(0, round(panes.winfo_width() * 0.40), 0)
        root.update_idletasks()
        refresh()
        polling = lifetime.after(150, poll)
        root.mainloop()
    finally:
        lifetime.close()
        queue.close()
        recovery.close()
        upload_consent.close()
        stop()
        if resize_timer is not None:
            lifetime.cancel(resize_timer)
        if restore_timer is not None:
            lifetime.cancel(restore_timer)
        if polling is not None:
            lifetime.cancel(polling)
        cap.release()
        lifetime.destroy()
    if result is None:
        raise ReviewCancelled('Task review cancelled. Detailed annotations are saved; no new task groups were confirmed.')
    return result
