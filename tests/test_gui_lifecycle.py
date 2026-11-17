"""Offline GUI lifetime races, including native Tcl finalizers in a child process."""

import contextlib
import gc
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Event, Thread, current_thread
import textwrap
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import gemini_runtime as runtime
from gui import gui_elements as labels, semantic_review as tasks
from gui.ai_upload import UploadConsent
from gui.quota_recovery import QuotaRecovery
from gui import lifecycle as life
from gui.video_selection import VideoSelectionController
from toolbox import main as app
from tests.test_quota_recovery import Desktop
from tests.test_semantic_aggregation import detailed


ROOT = Path(__file__).resolve().parents[1]


class StrictDesktop(Desktop):
    """Reject widget/variable access after destruction and on worker threads."""
    def __init__(self):
        self.dead = False
        super().__init__()
        self.root.destroy.side_effect = self.destroy

    def widget(self, kind, *args, **options):
        item = super().widget(kind, *args, **options)
        for name in ('configure', 'set', 'get', 'insert', 'delete', 'item', 'selection_set'):
            operation = getattr(item, name)
            original = operation.side_effect
            def checked(*args, _original=original, **kwargs):
                assert current_thread().name == 'MainThread', 'Widget/Variable touched by worker'
                assert not self.dead, 'Destroyed window touched'
                if _original is not None:
                    return _original(*args, **kwargs)
            operation.side_effect = checked
        return item

    def destroy(self):
        assert not self.dead, 'Window destroyed twice'
        self.root.bindings['<Destroy>'](type('DestroyEvent', (), {'widget': self.root})())
        self.dead = True


class CallbackTests(unittest.TestCase):
    def test_close_unblocks_a_decision_already_taken_by_the_ui(self):
        channel = life.DecisionChannel()
        ready, result = Event(), []
        def request():
            ready.set()
            result.append(channel.request('large video'))
        worker = Thread(target=request)
        worker.start()
        self.assertTrue(ready.wait(2))
        # Synchronize on the channel lock, without a Tk loop or I/O.
        for _ in range(1000):
            decision = channel.take()
            if decision is not None:
                break
            Event().wait(.001)
        self.assertIsNotNone(decision)
        channel.close()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [None])
        channel.resolve(decision, True)  # A late dialog cannot restore approval.
        self.assertIsNone(decision['answer'])

    def test_retained_photo_releases_native_refs_on_ui_thread(self):
        desktop = Desktop()
        owner = life.WindowLifetime(desktop.root)
        engine = Mock()
        calls = []
        engine.call.side_effect = lambda *args: calls.append((current_thread().name, args))
        photo = tk.PhotoImage.__new__(tk.PhotoImage)
        photo.name, photo.tk = 'old_preview', engine
        class RetainedPhoto:
            pass
        image = RetainedPhoto()
        image._PhotoImage__photo = photo
        image.tk = engine
        owner.image(image)
        replacement = RetainedPhoto()
        next_photo = tk.PhotoImage.__new__(tk.PhotoImage)
        next_photo.name, next_photo.tk = 'next_preview', engine
        replacement._PhotoImage__photo, replacement.tk = next_photo, engine
        owner.image(replacement)
        self.assertEqual(len(owner.images), 1)  # Playback cannot retain every frame.
        self.assertIsNone(photo.tk)  # A retained old frame is already disarmed.
        owner.release_images()
        self.assertEqual(calls, [('MainThread', ('image', 'delete', 'old_preview')),
                                 ('MainThread', ('image', 'delete', 'next_preview'))])
        self.assertIsNone(photo.tk)
        self.assertIsNone(photo.name)
        self.assertIsNone(image.tk)
        owner.destroy()

    def test_tk_variable_disposal_runs_on_ui_and_later_worker_gc_makes_no_tcl_calls(self):
        calls = []
        engine = Mock()
        engine.call.side_effect = lambda *args: calls.append((current_thread().name, args)) or True
        engine.getboolean.side_effect = bool
        engine.globalunsetvar.side_effect = lambda name: calls.append((current_thread().name, ('unset', name)))
        engine.deletecommand.side_effect = lambda name: calls.append((current_thread().name, ('delete', name)))
        variable = tk.StringVar.__new__(tk.StringVar)
        variable._tk, variable._root = engine, object()
        variable._name, variable._tclCommands = 'reviewed_label', ['trace']
        variable.cycle = variable
        life.release_variable(variable)
        self.assertIsNone(variable._tk)
        self.assertEqual([name for _, (name, *_) in calls], ['info', 'unset', 'delete'])
        self.assertTrue(all(thread == 'MainThread' for thread, _ in calls))
        queue = Queue()
        queue.put(variable)
        del variable
        before = list(calls)
        def collect():
            value = queue.get()
            del value
            gc.collect()
        worker = Thread(target=collect)
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(calls, before)

    def test_all_scheduled_callbacks_cancel_and_dispatched_callbacks_cannot_reschedule(self):
        desktop = StrictDesktop()
        owner = life.WindowLifetime(desktop.root)
        ran = []
        def repeat():
            ran.append(True)
            owner.after(1, repeat)
        owner.after(1, repeat)
        owner.after_idle(repeat)
        dispatched = [callback for _, callback in desktop.timers.values()]
        owner.request_close()
        self.assertFalse(desktop.timers)
        owner.destroy()
        for callback in dispatched:
            callback()
        self.assertEqual(ran, [])
        self.assertFalse(desktop.timers)
        self.assertIsNone(owner.after(1, repeat))
        desktop.root.destroy.assert_called_once()

    def test_callback_exceptions_are_not_suppressed(self):
        desktop = Desktop()
        owner = life.AfterCallbacks(desktop.root)
        owner.after(1, Mock(side_effect=ValueError('visible callback failure')))
        with self.assertRaisesRegex(ValueError, 'visible callback failure'):
            desktop.pump()
        self.assertEqual(owner.ids, set())
        owner.close()

    def test_gui_operations_from_worker_fail_explicitly(self):
        desktop = Desktop()
        owner = life.AfterCallbacks(desktop.root)
        errors = []
        def worker():
            try:
                owner.after(1, lambda: None)
            except RuntimeError as error:
                errors.append(str(error))
        thread = Thread(target=worker)
        thread.start()
        thread.join(2)
        self.assertIn('main thread', errors[0])
        self.assertFalse(desktop.timers)
        owner.close()

    def test_worker_endpoints_do_not_retain_ui_controllers_and_close_unblocks_waiters(self):
        desktop = StrictDesktop()
        quota = QuotaRecovery(desktop.root)
        upload = UploadConsent(desktop.root)
        self.assertIs(quota.handle_quota.__self__, quota.channel)
        self.assertIs(upload.authorize.__self__, upload.channel)
        self.assertFalse(hasattr(quota.channel, 'root'))
        self.assertFalse(hasattr(upload.channel, 'root'))
        result = []
        thread = Thread(target=lambda: result.append(quota.handle_quota(runtime.Credential('fake', 'fake'))))
        thread.start()
        desktop.until(lambda: quota.waiting)
        quota.close()
        upload.close()
        thread.join(2)
        self.assertEqual(result, [None])
        self.assertFalse(desktop.timers)
        self.assertIsNone(quota.root)
        self.assertIsNone(upload.root)

    def test_inbox_drops_completion_and_progress_after_close(self):
        inbox = life.WorkerInbox()
        started, release = Event(), Event()
        def request(**kwargs):
            started.set()
            release.wait(2)
            kwargs['progress']('completed late')
            return 'A label'
        job = life.AIJob(request, (), {}, 'result', 'error', progress_tag='progress')
        thread = Thread(target=life.run_ai_job, args=(job, inbox))
        thread.start()
        self.assertTrue(started.wait(2))
        inbox.close()
        release.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        with self.assertRaises(Empty):
            inbox.get_nowait()
        self.assertIsNone(life.run_ai_job.__closure__)


class TransitionTests(unittest.TestCase):
    def test_closing_with_active_individual_or_batch_then_opening_task_review(self):
        for provider in ('gemini', 'twelvelabs'):
            for batch in (False, True):
                with self.subTest(provider=provider, batch=batch):
                    self.transition(provider, batch, completed_before_close=False)

    def test_repeated_completed_suggest_all_to_task_review_transitions(self):
        for repetition in range(3):
            for provider in ('gemini', 'twelvelabs'):
                with self.subTest(repetition=repetition, provider=provider):
                    self.transition(provider, True, completed_before_close=True)

    def transition(self, provider, batch, completed_before_close):
        old, new = StrictDesktop(), StrictDesktop()
        started, release, finished = Event(), Event(), Event()
        stale = []
        config = dict(RAVEE_LABEL_PROVIDER=provider, RAVEE_AGGREGATION_PROVIDER=provider,
                      TWELVELABS_MODEL='pegasus1.6', GEMINI_MODEL='offline-gemini')
        records = [dict(cluster_id=i, start_frame=i * 15, end_frame=i * 15 + 14) for i in range(2)]
        frames = detailed(['Human first action', 'Human final action'])
        original = frames.copy(deep=True)
        def request(*args, **kwargs):
            started.set()
            release.wait(3)
            try:
                callback_owner = kwargs['on_daily_quota'].__self__
                self.assertIsInstance(callback_owner, life.DecisionChannel)
                self.assertFalse(hasattr(callback_owner, 'root'))
                if 'progress' in kwargs:
                    kwargs['progress']('Worker finished')
                return {1: 'AI first action', 2: 'AI final action'} if batch else 'AI first action'
            finally:
                finished.set()
        def label_screen():
            old.click('Suggest All' if batch else 'Suggest label')
            old.until(started.is_set)
            if completed_before_close:
                release.set()
                old.until(lambda: old.find('Button', 'Use batch suggestion').options['state'] == 'normal')
            stale.extend(callback for _, callback in old.timers.values())
            entry = old.find('Entry')
            entry.value = 'Human first action'
            old.click('Commit & Next')
            entry.value = 'Human final action'
            old.click('Commit & Finish')
            # Cleanup is deferred until this mainloop returns, even when a
            # request is still in flight. Every pending timer is already canceled.
            self.assertFalse(old.timers)
            self.assertFalse(old.dead)
        old.root.mainloop.side_effect = label_screen
        def task_screen():
            self.assertTrue(old.dead)
            self.assertFalse(old.timers)
            for callback in stale:
                callback()  # Simulate a Tcl callback dispatched just before close.
            release.set()
            self.assertTrue(finished.wait(2))
            old.pump()
            new.pump()
            new.click('Confirm task groups')
        new.root.mainloop.side_effect = task_screen
        try:
            with patch.object(runtime, 'configuration', return_value=config), old.patch(labels), \
                 patch.object(labels, 'suggest_all_activities' if batch else 'suggest_activity', side_effect=request):
                labels.label_video_clusters('synthetic.avi', records)
                self.assertEqual(labels.LABEL, {1: 'Human first action', 2: 'Human final action'})
                self.assertTrue(old.dead)
                self.assertFalse(old.timers)
                with new.patch(tasks):
                    result = tasks.review_task_groups('synthetic.avi', frames, offline=True)
                self.assertEqual(len(result), 2)
                self.assertFalse(new.timers)
                self.assertTrue(new.dead)
        finally:
            release.set()
        self.assertTrue(frames.equals(original))

    def test_video_selector_drops_late_inspection_after_close(self):
        started, release = Event(), Event()
        def inspect(path):
            started.set()
            release.wait(2)
            return {'status': 'Cached'}
        controller = VideoSelectionController(inspect, Mock())
        # A synthetic queue job avoids touching any video file.
        controller.jobs.put(('inspect', controller.revision, Path('synthetic.avi')))
        self.assertTrue(started.wait(2))
        controller.close()
        release.set()
        controller.worker.join(2)
        self.assertFalse(controller.worker.is_alive())
        self.assertEqual(controller.poll(), [])

    def test_actual_entrypoint_shares_one_lazy_tk_owner_across_review_screens(self):
        owner, first, second = Desktop().root, Desktop().root, Desktop().root
        def run(**kwargs):
            self.assertIs(life.new_window(), first)
            self.assertIs(life.new_window(), second)
        with patch.object(life.tk, 'Tk', return_value=owner) as factory, \
             patch.object(life.tk, 'Toplevel', side_effect=[first, second]) as screens, \
             patch.object(app, 'launch_interactive', side_effect=run):
            app.entrypoint([])
        factory.assert_called_once()
        self.assertEqual([call.args for call in screens.call_args_list], [(owner,), (owner,)])
        owner.withdraw.assert_called_once()
        owner.destroy.assert_called_once()


class NativeTclTests(unittest.TestCase):
    def test_native_tcl_variables_and_retained_cycles_are_safe_under_worker_gc(self):
        script = textwrap.dedent('''
            import tests  # Offline credentials/configuration isolation in the child, too.
            import gc, sys, tkinter as tk
            from queue import Queue
            from threading import Thread
            from gui.lifecycle import release_variable
            errors = []
            sys.unraisablehook = lambda error: errors.append(str(error.exc_value))
            try:
                interpreter = tk.Tcl()  # No desktop/window, no external work.
            except tk.TclError as error:
                if 'usable init.tcl' in str(error):
                    print('TCL_RUNTIME_UNAVAILABLE')
                    sys.exit(0)
                raise
            variable = tk.StringVar(master=interpreter, value='Human reviewed label')
            name = str(variable)
            release_variable(variable)
            assert not interpreter.getboolean(interpreter.call('info', 'exists', name))
            variable.cycle = variable
            queue = Queue()
            queue.put(variable)
            del variable
            def collect():
                value = queue.get()
                assert value._tk is None and value._root is None
                del value
                gc.collect()
            worker = Thread(target=collect)
            worker.start()
            worker.join(5)
            assert not worker.is_alive()
            assert not errors, errors
            print('Tcl variables safely finalized')
        ''')
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if 'TCL_RUNTIME_UNAVAILABLE' in result.stdout:
            self.skipTest('Execution environment cannot load native Tcl init.tcl; deterministic disposal/GC tests still run')
        self.assertIn('Tcl variables safely finalized', result.stdout)
        self.assertEqual(result.stderr, '')

    def test_native_tk_repeated_hidden_screens_leave_no_old_callbacks_or_interpreter_refs(self):
        script = textwrap.dedent('''
            import tests  # Offline credentials/configuration isolation in the child, too.
            import gc, sys, tkinter as tk
            from threading import Thread
            from gui import lifecycle as life
            errors = []
            sys.unraisablehook = lambda error: errors.append(str(error.exc_value))
            retained = []
            try:
                with life.gui_application():
                    owner = None
                    for index in range(8):
                        window = life.new_window()
                        window.withdraw()
                        scope = life.WindowLifetime(window)
                        variable = scope.variable(tk.StringVar, value='Human label')
                        child = tk.Label(window, textvariable=variable)
                        if owner is None:
                            owner = window.master
                        assert owner is window.master
                        scope.after(0, scope.request_close)
                        window.mainloop()
                        scope.destroy()
                        assert variable._tk is None
                        assert window.tk is None and child.tk is None
                        assert not owner.tk.call('after', 'info')
                        cycle = [window, variable, child]
                        cycle.append(cycle)
                        retained.append(cycle)
                    owner.tk.call('update')
                assert owner.tk is None
                def collect():
                    retained.clear()
                    gc.collect()
                worker = Thread(target=collect)
                worker.start()
                worker.join(5)
                assert not worker.is_alive() and not errors, errors
                print('Native Tk transitions safely completed')
            except tk.TclError as error:
                if 'usable init.tcl' in str(error) or 'usable tk.tcl' in str(error):
                    print('TCL_RUNTIME_UNAVAILABLE')
                elif 'no display' in str(error).lower() or "couldn't connect to display" in str(error).lower():
                    print('DISPLAY_UNAVAILABLE')
                else:
                    raise
        ''')
        result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if 'TCL_RUNTIME_UNAVAILABLE' in result.stdout:
            self.skipTest('Execution environment cannot load native Tcl/Tk init files; mocked screen transitions still run')
        if 'DISPLAY_UNAVAILABLE' in result.stdout:
            self.skipTest('Native Tk display unavailable; deterministic GUI and Tcl tests still run')
        self.assertIn('Native Tk transitions safely completed', result.stdout)
        self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
