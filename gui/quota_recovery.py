"""Nonmodal quota recovery. Workers wait on an Event, never on Tkinter.

The controller owns no annotation/review state. Cancel/manual leave the
pending API operation unsuccessful; completed results remain in their cache.
"""
from threading import Event, Lock
import tkinter as tk
from tkinter import ttk

from gemini_runtime import configured_credentials, select_credential
from gui.theme import apply_theme


class QuotaRecovery:
    _visible = None  # At most one quota dialog across review windows.

    def __init__(self, root, *, on_pause=None, on_resume=None, on_decision=None):
        self.root = root
        self.on_pause = on_pause or (lambda: None)
        self.on_resume = on_resume or (lambda: None)
        self.on_decision = on_decision or (lambda action: None)
        self.disabled = False
        self.closed = False
        self.interruptions = 0
        self.last_action = None
        self.dialog = None
        self._pending = None
        self._lock = Lock()
        self._timer = root.after(100, self._poll)

    @property
    def waiting(self):
        with self._lock:
            return self._pending is not None

    def handle_quota(self, credential):
        """Worker callback: no Tk calls, no automatic retries or key rotation."""
        with self._lock:
            if self.closed or self.disabled or self._pending is not None:
                return None
            request = {'event': Event(), 'credential': credential, 'replacement': None}
            self._pending = request
        request['event'].wait()
        with self._lock:
            return None if self.closed else request['replacement']

    def _poll(self):
        self._timer = None
        if self.closed:
            return
        with self._lock:
            pending = self._pending
        if pending is not None and self.dialog is None and QuotaRecovery._visible is None:
            self.interruptions += 1
            self.on_pause()
            self._show(pending['credential'])
        self._timer = self.root.after(100, self._poll)

    def _show(self, current):
        dialog = self.dialog = tk.Toplevel(self.root)
        QuotaRecovery._visible = self
        dialog.title('Gemini daily quota exhausted')
        dialog.transient(self.root)
        dialog.resizable(True, False)
        apply_theme(dialog)
        content = ttk.Frame(dialog, padding=20)
        content.pack(fill='both', expand=True)
        ttk.Label(content, text='Daily request quota exhausted (RPD)',
                  font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        ttk.Label(content, wraplength=520, justify='left', text=(
            'Your labels, edits and completed suggestions are preserved. '
            'Continue reviewing while this request is paused. No credential was switched.\n\n'
            'You may explicitly select another authorized project, subject to provider limits. '
            'Keys belonging to the same project share its quota. '
            'Only the pending request will resume; no video preprocessing or upload will run.')).pack(anchor='w', pady=12)
        alternatives = [(label, cred) for label, cred in configured_credentials() if cred.key != current.key]
        names = [label for label, _ in alternatives]
        selector = ttk.Combobox(content, values=names, state='readonly', width=26)
        selector.pack(anchor='w', pady=(0, 10))
        if names:
            selector.current(0)
        message = ttk.Label(content, text='' if names else 'No other configured credential is available.',
                            wraplength=520)
        message.pack(anchor='w')

        def choose():
            index = selector.current()
            if index < 0 or index >= len(alternatives):
                return
            try:
                replacement = select_credential(alternatives[index][1].name)
                if replacement.key == current.key:
                    message.configure(text='Choose a different configured credential.')
                    return
            except RuntimeError:
                message.configure(text='Credential configuration changed. Cancel and try again.')
                return
            self.decide('select', replacement)

        buttons = ttk.Frame(content)
        buttons.pack(fill='x', pady=(14, 0))
        ttk.Button(buttons, text='Continue manually', command=lambda: self.decide('manual')).pack(side='left')
        ttk.Button(buttons, text='Select credential', state='normal' if names else 'disabled',
                   command=choose).pack(side='left', padx=8)
        ttk.Button(buttons, text='Cancel', command=lambda: self.decide('cancel')).pack(side='left')
        dialog.protocol('WM_DELETE_WINDOW', lambda: self.decide('cancel'))
        # Intentionally no grab_set, wait_window, nested mainloop or UI-thread I/O.

    def decide(self, action, replacement=None):
        """UI-thread decision. A selection must come from the explicit chooser."""
        if action not in ('manual', 'cancel', 'select'):
            raise ValueError('Unknown quota recovery action')
        if action == 'select' and replacement is None:
            raise ValueError('Credential selection is required')
        with self._lock:
            request = self._pending
            if request is None:
                return
            self._pending = None
            request['replacement'] = replacement if action == 'select' else None
        self.disabled = self.disabled or action == 'manual'
        self.last_action = action
        try:
            self._dismiss()
            if action == 'select':
                self.on_resume()
            self.on_decision(action)
        finally:
            # Even a destroyed widget/error in a UI notification must not leave
            # a worker permanently waiting after its request has been detached.
            request['event'].set()

    def _dismiss(self):
        if self.dialog is not None:
            try:
                self.dialog.destroy()
            except tk.TclError:
                pass  # Parent may already have destroyed its child windows.
            self.dialog = None
        if QuotaRecovery._visible is self:
            QuotaRecovery._visible = None

    def close(self):
        self.closed = True
        if self._timer is not None:
            try:
                self.root.after_cancel(self._timer)
            except tk.TclError:
                pass
            self._timer = None
        with self._lock:
            request, self._pending = self._pending, None
        self._dismiss()
        if request is not None:
            request['event'].set()
