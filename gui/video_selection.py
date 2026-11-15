"""Startup selection and background stage preparation; all Tk calls stay on UI thread."""

from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gui.theme import COLORS, apply_theme, fit_window, workspace_header
from toolbox.pipeline_cache import VIDEO_SUFFIXES, discover_videos


class VideoSelectionController:
    """Small testable queue boundary between selection and the existing pipeline."""

    def __init__(self, inspect, process):
        self.inspect = inspect
        self.process = process
        self.selected = None
        self.busy = False
        self.revision = 0
        self.closed = False
        self.jobs, self.events = Queue(), Queue()
        self.worker = Thread(target=self._work, daemon=True, name='ravee-video-preparation')
        self.worker.start()

    def select(self, path):
        if self.busy:
            raise ValueError('Wait for the selected video to finish processing')
        source = Path(path).resolve()
        if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f'Not a supported video: {source}')
        self.selected = source
        self.revision += 1
        self.jobs.put(('inspect', self.revision, source))

    def clear(self):
        if not self.busy:
            self.selected = None
            self.revision += 1

    def start(self):
        if self.closed or self.busy:
            return False
        if self.selected is None:
            raise ValueError('Select exactly one video first')
        # This is the only route that invokes process(). select() is read-only.
        self.busy = True
        self.jobs.put(('process', self.revision, self.selected))
        return True

    def _work(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            kind, revision, source = job
            if self.closed or revision != self.revision:
                continue
            try:
                if kind == 'inspect':
                    result = self.inspect(source)
                    self.events.put(('cache', revision, result))
                else:
                    result = self.process(source, lambda message: self.events.put(
                        ('progress', revision, message)))
                    self.events.put(('success', revision, (source, result)))
            except Exception as error:
                self.events.put(('error' if kind == 'process' else 'inspection_error',
                                 revision, str(error)))

    def poll(self):
        events = []
        while True:
            try:
                kind, revision, payload = self.events.get_nowait()
            except Empty:
                return events
            if revision != self.revision or self.closed:
                continue
            if kind in ('success', 'error'):
                self.busy = False
            events.append((kind, payload))

    def close(self):
        self.closed = True
        self.jobs.put(None)


def select_video_to_process(input_directory, inspect, process):
    """Return (selected_path, prepared_run), or None if closed before processing.

    Preparation finishes before this root is destroyed. Existing labeling and
    semantic review roots are then opened by main.py on the same main thread.
    """
    root = tk.Tk()
    apply_theme(root)
    root.title('RAVEE - Select Video to Process')
    fit_window(root, 1120, 760, 900, 620)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    workspace_header(root, 'Select Video to Process',
        'Choose one video. Verified stages are reused; missing stages run when you process it.',
        'Start  /  Video selection')

    body = ttk.Frame(root, padding=(24, 0, 24, 16))
    body.grid(row=1, column=0, sticky='nsew')
    body.columnconfigure(0, weight=1)
    body.rowconfigure(1, weight=1)
    tools = ttk.Frame(body)
    tools.grid(row=0, column=0, sticky='ew', pady=(0, 12))
    ttk.Label(tools, text='Input videos', style='Brand.TLabel').pack(side='left')
    refresh_button = ttk.Button(tools, text='Refresh List')
    refresh_button.pack(side='right')
    browse_button = ttk.Button(tools, text='Browse')
    browse_button.pack(side='right', padx=(0, 8))
    listing = ttk.Frame(body, style='Card.TFrame', padding=12)
    listing.grid(row=1, column=0, sticky='nsew')
    listing.columnconfigure(0, weight=1)
    listing.rowconfigure(0, weight=1)
    tree = ttk.Treeview(listing, columns=('name', 'folder'), show='headings', selectmode='browse', height=8)
    tree.heading('name', text='Video')
    tree.heading('folder', text='Folder')
    tree.column('name', width=400, minwidth=180)
    tree.column('folder', width=420, minwidth=180)
    tree.grid(row=0, column=0, sticky='nsew')
    scrollbar = ttk.Scrollbar(listing, orient='vertical', command=tree.yview)
    scrollbar.grid(row=0, column=1, sticky='ns')
    tree.configure(yscrollcommand=scrollbar.set)
    tree.tag_configure('alternate', background=COLORS['stripe'])

    info = ttk.Frame(body, style='Card.TFrame', padding=16)
    info.grid(row=2, column=0, sticky='ew', pady=(12, 0))
    info.columnconfigure(0, weight=1)
    selected_text = tk.StringVar(value='Select one video from the list, or browse for a video.')
    cache_text = tk.StringVar(value='Cache status: select a video')
    stage_text = tk.StringVar(value='')
    ttk.Label(info, textvariable=selected_text, style='Section.TLabel', wraplength=820).grid(row=0, column=0, sticky='w')
    ttk.Label(info, textvariable=cache_text, style='Card.TLabel').grid(row=1, column=0, sticky='w', pady=(8, 4))
    ttk.Label(info, textvariable=stage_text, style='CardMuted.TLabel', wraplength=820).grid(row=2, column=0, sticky='w')

    footer = ttk.Frame(root, padding=(24, 0, 24, 20))
    footer.grid(row=2, column=0, sticky='ew')
    footer.columnconfigure(0, weight=1)
    progress = ttk.Progressbar(footer, mode='indeterminate')
    progress.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 12))
    status = tk.StringVar(value='Process Video authorizes missing preprocessing and Docker IDT extraction for your selected video.')
    status_label = ttk.Label(footer, textvariable=status, style='Muted.TLabel', wraplength=650)
    status_label.grid(row=1, column=0, sticky='w', padx=(0, 16))
    process_button = ttk.Button(footer, text='Process Video', style='Primary.TButton', state='disabled')
    process_button.grid(row=1, column=1, sticky='e')
    controller = VideoSelectionController(inspect, process)
    paths, browsed = {}, set()
    result, polling = None, None

    def set_busy(busy):
        state = 'disabled' if busy else 'normal'
        browse_button.configure(state=state)
        refresh_button.configure(state=state)
        process_button.configure(state='normal' if not busy and controller.selected is not None else 'disabled')
        if busy:
            tree.state(['disabled'])
            progress.start(12)
        else:
            tree.state(['!disabled'])
            progress.stop()

    def choose(_event=None):
        if controller.busy:
            return
        selection = tree.selection()
        if len(selection) != 1:
            controller.clear()
            process_button.configure(state='disabled')
            return
        source = paths[selection[0]]
        if source == controller.selected:
            return
        try:
            controller.select(source)
        except ValueError as error:
            controller.clear()
            process_button.configure(state='disabled')
            status.set(str(error))
            return
        selected_text.set(source.name)
        cache_text.set('Cache status: checking verified artifacts...')
        stage_text.set(str(source))
        process_button.configure(state='normal')

    def refresh(preferred=None):
        previous = preferred or controller.selected
        sources = sorted(set(path.resolve() for path in discover_videos(input_directory)) |
                         {path for path in browsed if path.is_file()}, key=lambda path: str(path).casefold())
        controller.clear()
        tree.delete(*tree.get_children())
        paths.clear()
        for i, source in enumerate(sources):
            key = str(i)
            paths[key] = source
            tree.insert('', 'end', iid=key, values=(source.name, str(source.parent)),
                        tags=('alternate',) if i % 2 else ())
        if sources:
            source = previous if previous in sources else sources[0]
            key = next(key for key, path in paths.items() if path == source)
            tree.selection_set(key)
            tree.see(key)
            choose()
        else:
            selected_text.set('No input videos found. Use Browse to choose a video.')
            cache_text.set('Cache status: select a video')
            stage_text.set('')
            process_button.configure(state='disabled')

    def browse():
        filename = filedialog.askopenfilename(parent=root, title='Select Video to Process',
            initialdir=str(input_directory), filetypes=[('Video files', ' '.join('*' + suffix for suffix in sorted(VIDEO_SUFFIXES))),
                                                       ('All files', '*.*')])
        if filename:
            source = Path(filename).resolve()
            if source.suffix.lower() not in VIDEO_SUFFIXES:
                messagebox.showerror('Unsupported video', 'Choose a supported video file.', parent=root)
                return
            browsed.add(source)
            refresh(source)

    def start():
        try:
            if controller.start():
                set_busy(True)
                status.set('Preparing the selected video. Verified completed stages will be reused.')
        except ValueError as error:
            status.set(str(error))

    def poll():
        nonlocal result, polling
        for kind, payload in controller.poll():
            if kind == 'cache':
                cache_text.set('Cache status: ' + payload['status'])
                names = {'preprocessing': 'Preprocessing', 'idt': 'IDT', 'fisher_vectors': 'Fisher Vectors', 'finch': 'FINCH'}
                stage_text.set('   /   '.join(f'{names[name]}: {"verified" if ready else "missing / unverified"}'
                                             for name, ready in payload['stages'].items()))
            elif kind == 'inspection_error':
                cache_text.set('Cache status: verification unavailable')
                stage_text.set(payload)
            elif kind == 'progress':
                status.set(payload)
            elif kind == 'error':
                set_busy(False)
                status.set('Processing failed. Completed caches are preserved. ' + payload[:450])
                messagebox.showerror('Video processing failed', payload, parent=root)
                # Recheck any stages successfully cached before the failure.
                controller.select(controller.selected)
            elif kind == 'success':
                result = payload
                root.quit()
                return
        polling = root.after(100, poll)

    def close():
        if controller.busy:
            status.set('Processing is running. Wait for it to finish; completed stages are saved automatically.')
            return
        root.quit()

    browse_button.configure(command=browse)
    refresh_button.configure(command=refresh)
    process_button.configure(command=start)
    tree.bind('<<TreeviewSelect>>', choose)
    root.protocol('WM_DELETE_WINDOW', close)
    try:
        refresh()
        polling = root.after(100, poll)
        root.mainloop()
    finally:
        controller.close()
        progress.stop()
        if polling is not None:
            root.after_cancel(polling)
        root.destroy()
    return result
