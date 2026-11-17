"""Main-thread ownership of screens, Tcl resources and scheduled callbacks.

Worker endpoints below contain only Python data/locks/queues. Never pass a
window/controller, a widget closure or a Tk Variable to an AI worker.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from queue import Queue, Empty
from threading import Event, Lock, get_ident, main_thread
import tkinter as tk

from gemini_runtime import redact_text


def _main_thread():
    if get_ident() != main_thread().ident:
        raise RuntimeError('RAVEE GUI lifecycle operations must run on the main thread')


def release_variable(variable):
    """Dispose a Python Tk variable on its owner thread, disarming later GC."""
    _main_thread()
    if isinstance(variable, tk.Variable) and variable._tk is not None:
        tk.Variable.__del__(variable)
        variable._tk = variable._root = None


class AfterCallbacks:
    def __init__(self, root):
        _main_thread()
        self.root, self.closed, self.ids = root, False, set()

    def _schedule(self, delay, callback, idle=False):
        _main_thread()
        if self.closed:
            return None
        token = []
        def invoke():
            self.ids.discard(token[0])
            if not self.closed:
                callback()
        identifier = self.root.after_idle(invoke) if idle else self.root.after(delay, invoke)
        token.append(identifier)
        self.ids.add(identifier)
        return identifier

    def after(self, delay, callback):
        return self._schedule(delay, callback)

    def after_idle(self, callback):
        return self._schedule(0, callback, idle=True)

    def cancel(self, identifier):
        _main_thread()
        if identifier in self.ids:
            self.ids.remove(identifier)
            self.root.after_cancel(identifier)

    def close(self):
        _main_thread()
        self.closed = True  # Guard callbacks already dispatched before cancellation.
        for identifier in tuple(self.ids):
            self.cancel(identifier)
        self.root = None


class WindowLifetime(AfterCallbacks):
    def __init__(self, root):
        super().__init__(root)
        self.window, self.destroyed = root, False
        self.owns_interpreter = isinstance(root, tk.Misc) and root.master is None
        self.variables, self.destroyed_widgets = [], []
        self.images = set()
        root.bind('<Destroy>', self._on_destroy, add='+')
        root.protocol('WM_DELETE_WINDOW', self.request_close)

    def variable(self, factory, **kwargs):
        _main_thread()
        value = factory(master=self.window, **kwargs)
        self.variables.append(value)
        return value

    def release_variables(self):
        _main_thread()
        for variable in self.variables:
            # Tk has no public Variable.dispose(). Run its normal cleanup while
            # Tcl is owned here; cyclic GC may run on ANY allocating thread.
            release_variable(variable)
        self.variables.clear()

    def image(self, image):
        _main_thread()
        # Each screen has one preview. Dispose the preceding frame on the GUI
        # thread before replacing it, even if a traceback/cycle retains it.
        # Strong ownership of only the current frame also prevents worker GC
        # from finalizing an abandoned image while the screen is still open.
        self.release_images()
        if isinstance(getattr(image, '_PhotoImage__photo', None), tk.Image):
            self.images.add(image)
        return image

    def release_images(self):
        _main_thread()
        for image in tuple(self.images):
            photo = image._PhotoImage__photo
            if photo.name and not (self.destroyed and self.owns_interpreter):
                photo.tk.call('image', 'delete', photo.name)
            # Tk/Pillow have no public disposal API. Disarm their destructors
            # and release BOTH native interpreter refs, even if Python retains
            # an old photo after the screen's widgets have been destroyed.
            photo.name = photo.tk = image.tk = None
        self.images.clear()

    def _on_destroy(self, event):
        self.destroyed_widgets.append(event.widget)
        if event.widget is self.window:
            self.destroyed = True
            self.close()

    def request_close(self):
        _main_thread()
        self.close()
        self.window.quit()  # Destroy only in the owning function's finally block.

    def destroy(self):
        _main_thread()
        if self.window is None:
            return
        self.close()
        window = self.window
        self.release_images()
        widgets = list(self.destroyed_widgets)
        if isinstance(window, tk.Misc) and not self.destroyed:
            def descendants(widget):
                widgets.append(widget)
                for child in widget.winfo_children():
                    descendants(child)
            descendants(window)
        for widget in widgets:
            # ImageTk owns a Tcl image handle; release it here, not in a worker GC.
            if getattr(widget, 'image', None) is not None:
                widget.image = None
        if not self.destroyed:
            window.destroy()
            self.destroyed = True
        self.release_variables()
        # Destroyed widgets have no further valid Tk operations. Detach their
        # interpreter refs so recursive Python callback cycles cannot keep a
        # Tkapp alive until its last DECREF occurs on a background thread.
        for widget in widgets + self.destroyed_widgets:
            if isinstance(widget, tk.Misc):
                widget.tk = None
                widget.master = None
        self.destroyed_widgets.clear()
        self.window = None


_application = None


@contextmanager
def gui_application():
    """One lazy hidden Tk owner for the actual main.py workflow; no nested mainloops."""
    global _application
    _main_thread()
    if _application is not None:
        yield
        return
    application = _application = []
    try:
        yield
    finally:
        try:
            if application:
                WindowLifetime(application[0]).destroy()
        finally:
            _application = None


def new_window():
    _main_thread()
    if _application is None:
        return tk.Tk()
    if not _application:
        owner = tk.Tk()
        owner.withdraw()
        _application.append(owner)
    return tk.Toplevel(_application[0])


class WorkerInbox:
    """Pure Python endpoint: late results/progress are dropped after screen close."""
    def __init__(self):
        self.queue, self.lock, self.stopped = Queue(), Lock(), Event()

    def put(self, value):
        with self.lock:
            if not self.stopped.is_set():
                self.queue.put(value)

    def progress(self, tag, message):
        self.put((tag, message, None))

    def get_nowait(self):
        return self.queue.get_nowait()

    def close(self):
        with self.lock:
            self.stopped.set()
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break


@dataclass
class AIJob:
    operation: object
    args: tuple
    options: dict
    result_tag: object
    error_tag: object
    progress_tag: object = None
    with_stats: bool = False


def run_ai_job(job, inbox):
    """Module-level thread target: neither closure cells nor arguments own Tk."""
    if inbox.stopped.is_set():
        return
    options, stats = dict(job.options), None
    if job.with_stats:
        stats = options['stats'] = {}
    if job.progress_tag is not None:
        options['progress'] = partial(inbox.progress, job.progress_tag)
    try:
        result = job.operation(*job.args, **options)
    except Exception as error:
        inbox.put((job.error_tag, None, redact_text(error, 400)))
    else:
        inbox.put((job.result_tag, result, stats))


class DecisionChannel:
    """Worker-facing consent/recovery endpoint; deliberately has no GUI references."""
    def __init__(self):
        self.lock, self.pending = Lock(), None
        self.inflight = None
        self.closed = self.disabled = False

    def request(self, payload):
        with self.lock:
            if self.closed or self.disabled or self.inflight is not None:
                return None
            request = dict(event=Event(), payload=payload, answer=None)
            self.pending = self.inflight = request
        request['event'].wait()
        return None if self.closed else request['answer']

    def authorize_upload(self, path, size):
        return bool(self.request((path, size)))

    def peek(self):
        with self.lock:
            return self.pending

    def take(self):
        with self.lock:
            request, self.pending = self.pending, None
            return request

    def resolve(self, request, answer):
        with self.lock:
            request['answer'] = None if self.closed else answer
            if self.inflight is request:
                self.inflight = None
        request['event'].set()

    def close(self):
        with self.lock:
            self.closed = True
            request = self.inflight
            self.inflight = self.pending = None
        if request is not None:
            self.resolve(request, None)
