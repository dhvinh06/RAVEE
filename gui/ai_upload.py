"""Worker-to-UI consent for large optional-provider uploads, without widget I/O in workers."""

from pathlib import Path
from tkinter import messagebox
from gui.lifecycle import AfterCallbacks, DecisionChannel


class UploadConsent:
    def __init__(self, root, *, timers=None):
        self.root, self.closed = root, False
        self.channel = DecisionChannel()
        self.timers = timers if timers is not None else AfterCallbacks(root)
        self._owns_timers = timers is None
        self.timer = self.timers.after(100, self._poll)
        root.bind('<Destroy>', self._owner_destroyed, add='+')

    def _owner_destroyed(self, event):
        if event.widget is self.root:
            self.close()

    @property
    def pending(self):
        return self.channel.peek()

    @property
    def authorize(self):
        return self.channel.authorize_upload

    def _poll(self):
        self.timer = None
        if self.closed:
            return
        request = self.channel.take()
        if request is not None:
            approved = False
            path, size = request['payload']
            try:
                approved = messagebox.askyesno('Upload video to TwelveLabs?',
                    f'Upload the complete video {Path(path).name} '
                    f'({size / (1024 * 1024):.1f} MiB) to TwelveLabs? '
                    'It exceeds the 64 MiB automatic-upload threshold. API charges and provider storage apply. '
                    'Cancel keeps all labels, edits and review state.', parent=self.root)
            finally:
                self.channel.resolve(request, approved and not self.closed)
        if not self.closed:
            self.timer = self.timers.after(100, self._poll)

    def close(self):
        self.closed = True
        self.channel.close()
        if self.timer is not None:
            self.timers.cancel(self.timer)
            self.timer = None
        self.root = None
        if self._owns_timers:
            self.timers.close()
