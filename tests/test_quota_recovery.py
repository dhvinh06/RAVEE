"""Offline quota recovery through real request loops and mocked Tk widgets.

All provider clients, media capture and encoding are synthetic. Real worker
threads exercise the Event/queue bridge without requiring a desktop/Tcl runtime.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Event, Thread, current_thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from google import genai
import gemini_runtime as runtime
import label_suggester as labels
from gui import gui_elements, semantic_review, quota_recovery
from toolbox import semantic_aggregation as semantic
from tests.test_gemini_configuration_and_tasks import Failure
from tests import test_process_model_and_labels_v3 as label_fixtures
from tests import test_semantic_aggregation as fixtures


class Desktop:
    """Small deterministic Tk stand-in; all widget calls stay on main thread."""
    def __init__(self):
        self.widgets, self.timers, self.serial = [], {}, 0
        self.root = self.widget('Root')
        self.root.after.side_effect = self.after
        self.root.after_idle.side_effect = lambda callback: self.after(0, callback)
        self.root.after_cancel.side_effect = lambda key: self.timers.pop(key, None)

    def widget(self, kind, *args, **options):
        assert current_thread().name == 'MainThread', 'Tk called from worker'
        item = Mock()
        item.kind, item.options, item.bindings = kind, options.copy(), {}
        item.value, item.children, item.chosen, item.index = options.get('value', ''), [], (), -1
        item.winfo_width.return_value, item.winfo_height.return_value = 800, 450
        item.configure.side_effect = lambda **kw: item.options.update(kw)
        item.bind.side_effect = lambda event, callback: item.bindings.update({event: callback})
        item.get.side_effect = lambda *args: item.value
        item.set.side_effect = lambda value: setattr(item, 'value', value)
        item.get_children.side_effect = lambda: tuple(item.children)
        item.selection.side_effect = lambda: item.chosen
        def selection_set(*values):
            item.chosen = tuple(values[0]) if len(values) == 1 and not isinstance(values[0], str) else values
        item.selection_set.side_effect = selection_set
        def insert(*args, **kwargs):
            if kind == 'Treeview':
                item.children.append(kwargs['iid'])
            else:
                item.value = args[1]
        def delete(*args):
            item.value = ''
            if kind == 'Treeview':
                item.children.clear()
                item.chosen = ()
        def current(index=None):
            if index is not None:
                item.index = index
            return item.index
        item.insert.side_effect, item.delete.side_effect = insert, delete
        item.current.side_effect = current
        self.widgets.append(item)
        return item

    def after(self, delay, callback):
        assert current_thread().name == 'MainThread', 'Tk.after called from worker'
        self.serial += 1
        self.timers[self.serial] = (delay, callback)
        return self.serial

    def pump(self):
        for key in list(self.timers):
            item = self.timers.pop(key, None)
            if item:
                item[1]()

    def until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError('Mock UI/worker did not reach expected state')
            self.pump()
            Event().wait(.001)

    def find(self, kind, text=None):
        return next(item for item in reversed(self.widgets)
                    if item.kind == kind and (text is None or item.options.get('text') == text))

    def click(self, text):
        self.find('Button', text).options['command']()

    @contextlib.contextmanager
    def patch(self, module):
        with contextlib.ExitStack() as stack:
            threads = []
            def thread_factory(*args, **kwargs):
                worker = Thread(*args, **kwargs)
                threads.append(worker)
                return worker
            stack.enter_context(patch.object(module, 'Thread', side_effect=thread_factory))
            stack.enter_context(patch.object(module.tk, 'Tk', return_value=self.root))
            for kind in ('PanedWindow', 'Label', 'Text', 'StringVar', 'Toplevel'):
                stack.enter_context(patch.object(module.tk, kind,
                    side_effect=lambda *a, _kind=kind, **kw: self.widget(_kind, *a, **kw)))
            for kind in ('Frame', 'Label', 'Button', 'Entry', 'Combobox', 'Treeview', 'Scrollbar', 'Progressbar'):
                stack.enter_context(patch.object(module.ttk, kind,
                    side_effect=lambda *a, _kind=kind, **kw: self.widget(_kind, *a, **kw)))
            for name in ('apply_theme', 'fit_window', 'workspace_header', 'style_text'):
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name))
            stack.enter_context(patch.object(quota_recovery, 'apply_theme'))
            cap = Mock()
            cap.isOpened.return_value = True
            cap.get.side_effect = lambda prop: 15 if prop == module.cv2.CAP_PROP_FPS else 0
            cap.read.return_value = (False, None)
            stack.enter_context(patch.object(module.cv2, 'VideoCapture', return_value=cap))
            for name in ('askyesno', 'showwarning', 'showinfo'):
                stack.enter_context(patch.object(module.messagebox, name, return_value=True))
            try:
                yield
            finally:
                for worker in threads:
                    worker.join(2)
                    assert not worker.is_alive(), 'Worker outlived its mocked provider scope'


def owner(raw):
    result = MagicMock()
    result.__enter__.return_value = raw
    return result


class ConfiguredTest(unittest.TestCase):
    def setUp(self):
        guard = patch.object(genai, 'Client', side_effect=AssertionError('Unexpected provider client: offline tests only'))
        guard.start()
        self.addCleanup(guard.stop)
        self.config = dict(GEMINI_API_KEY='synthetic-alpha', GEMINI_API_KEY_2='synthetic-beta',
                           GEMINI_MODEL='unchanged-model')
        patcher = patch.object(runtime, 'configuration', side_effect=lambda: self.config.copy())
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(runtime, '_SESSION_CREDENTIAL_NAME', None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(setattr, quota_recovery.QuotaRecovery, '_visible', None)
        self.a = runtime.Credential('GEMINI_API_KEY', 'synthetic-alpha')
        self.b = runtime.Credential('GEMINI_API_KEY_2', 'synthetic-beta')


class ClassificationTests(ConfiguredTest):
    def test_safe_profile_names_match_numeric_named_and_legacy_configuration(self):
        self.config.update(GEMINI_API_KEY_10='synthetic-tenth', GEMINI_API_KEY_RESEARCH='synthetic-named',
                           GOOGLE_API_KEY='synthetic-google', GEMINI_API_KEY_COPY=self.a.key)
        profiles = dict((credential.name, label) for label, credential in runtime.configured_credentials())
        self.assertEqual(profiles['GEMINI_API_KEY'], 'Project 1')
        self.assertEqual(profiles['GEMINI_API_KEY_2'], 'Project 2')
        self.assertEqual(profiles['GEMINI_API_KEY_10'], 'Project 10')
        self.assertEqual(profiles['GEMINI_API_KEY_RESEARCH'], 'Project RESEARCH')
        self.assertEqual(profiles['GOOGLE_API_KEY'], 'Google default')
        self.assertNotIn('GEMINI_API_KEY_COPY', profiles)

    def test_only_explicit_daily_evidence_can_open_recovery(self):
        cases = [(Failure(429, 'GenerateContentRequestsPerDay'), 'daily_quota'),
                 (Failure(429, 'GenerateContentRequestsPerMinute'), 'rate_limit'),
                 (Failure(429, 'GenerateContentTokensPerMinute'), 'rate_limit'),
                 (Failure(429), 'quota_unknown'), (Failure(503), 'service_unavailable'),
                 (Failure(401), 'authentication'), (Failure(403), 'authentication'),
                 (runtime.InvalidResponseError('bad schema'), 'invalid_response')]
        for error, kind in cases:
            self.assertEqual(runtime.classify_error(error).kind, kind)
        error = Failure(429)
        error.args = ('RESOURCE_EXHAUSTED: quota exceeded',)
        self.assertEqual(runtime.classify_error(error).kind, 'quota_unknown')
        # A key containing a daily identifier cannot manufacture quota evidence.
        self.config['GEMINI_API_KEY_2'] = 'private-RequestsPerDay-value'
        error.args = ('quota exceeded private-RequestsPerDay-value',)
        self.assertEqual(runtime.classify_error(error).kind, 'quota_unknown')

    def test_ambiguous_transient_auth_and_invalid_errors_never_call_quota_handler(self):
        for error in (Failure(429), Failure(429, 'RequestsPerMinute'), Failure(503),
                      Failure(401), Failure(403), runtime.InvalidResponseError('bad output')):
            raw, callback = Mock(), Mock()
            raw.models.generate_content.side_effect = error
            with patch.object(runtime.time, 'sleep'), patch.object(runtime.random, 'uniform', return_value=0):
                with self.assertRaises(Exception), runtime.RequestSession(self.a, callback, factory=lambda _: owner(raw)) as session:
                    runtime.generate_with_retry(session, {'model': 'same-model'})
            callback.assert_not_called()

    def test_quota_guidance_and_other_daily_limits_are_not_rpd_exhaustion(self):
        for message in ('Check your daily quota and RPD limits.',
                        'Quota exceeded; inspect RPM, TPM and RPD.',
                        'See https://example.invalid/daily/RequestsPerDay',
                        'Check whether your daily request quota is exhausted.',
                        'Your daily request quota is not exhausted.',
                        'Daily token quota exhausted; RPD is also monitored.'):
            error = Failure(429)
            error.args = (message,)
            error.details = [{'@type': 'type.googleapis.com/google.rpc.Help',
                              'links': [{'description': 'Daily request quota exhausted',
                                         'url': 'https://example.invalid/daily'}]}]
            with self.subTest(message=message):
                self.assertNotEqual(runtime.classify_error(error).kind, 'daily_quota')
        error = Failure(429)
        error.args = ('Check whether your daily request quota is exhausted.',)
        self.assertNotEqual(runtime.classify_error(error).kind, 'daily_quota')
        for identifier in ('GenerateContentInputTokensPerModelPerDay-FreeTier',
                           'TokensPerDay', 'DailySpendLimit'):
            self.assertEqual(runtime.classify_error(Failure(429, identifier)).kind, 'quota_unknown')

    def test_real_sdk_nested_quota_failure_and_explicit_message(self):
        from google.genai.errors import ClientError
        for field in ('quotaId', 'quotaMetric', 'quota_id', 'quota_limit'):
            error = ClientError(429, {'error': {'message': 'Quota exceeded', 'details': [
                {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [
                    {field: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'},
                    {'quotaId': 'GenerateContentRequestsPerMinute'}]},
                {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '2s'}]}})
            self.assertEqual(runtime.classify_error(error).kind, 'daily_quota')
        for message in ('Daily request quota exhausted.', 'Your RPD limit has been exceeded.',
                        'You exceeded your requests per day quota.'):
            error = ClientError(429, {'error': {'message': message}})
            self.assertEqual(runtime.classify_error(error).kind, 'daily_quota')

    def test_minute_violation_is_authoritative_over_message_guidance(self):
        from google.genai.errors import ClientError
        error = ClientError(429, {'error': {
            'message': 'Quota exceeded. Check whether your daily request quota is exhausted.',
            'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure',
                         'violations': [{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel'}]}]}})
        self.assertEqual(runtime.classify_error(error).kind, 'rate_limit')


class SessionTests(ConfiguredTest):
    def test_same_credential_cannot_trigger_an_unauthorized_retry(self):
        raw, factory = Mock(), Mock()
        raw.models.generate_content.side_effect = Failure(429, 'RequestsPerDay')
        factory.return_value = owner(raw)
        with self.assertRaisesRegex(RuntimeError, 'different configured'), \
             runtime.RequestSession(self.a, lambda _: self.a, factory=factory) as session:
            runtime.generate_with_retry(session, {'model': 'fixed'})
        factory.assert_called_once_with(self.a)
        raw.models.generate_content.assert_called_once()

    def test_cancel_and_manual_without_replacement_never_retry_or_rotate(self):
        for callback in (None, Mock(return_value=None)):
            raw, factory = Mock(), Mock()
            raw.models.generate_content.side_effect = Failure(429, 'RequestsPerDay')
            factory.return_value = owner(raw)
            with self.assertRaises(Failure), runtime.RequestSession(self.a, callback, factory=factory) as session:
                runtime.generate_with_retry(session, {'model': 'fixed'})
            factory.assert_called_once_with(self.a)
            raw.models.generate_content.assert_called_once()
            self.assertEqual(runtime.resolve_credential(), self.a)

    def test_explicit_selection_recreates_client_replays_only_pending_kwargs_and_keeps_model(self):
        first, second = Mock(), Mock()
        first.models.generate_content.side_effect = Failure(429, 'RequestsPerDay')
        second.models.generate_content.return_value = 'completed'
        owners = [owner(first), owner(second)]
        factory = Mock(side_effect=owners)
        callback = Mock(side_effect=lambda _: runtime.select_credential('GEMINI_API_KEY_2'))
        kwargs = {'model': 'fixed-model', 'contents': {'parts': [{'text': 'same pending target'}]}}
        stats = {'api_calls': 0}
        with runtime.RequestSession(self.a, callback, factory=factory) as session:
            self.assertEqual(runtime.generate_with_retry(session, kwargs, stats=stats), 'completed')
        self.assertEqual([call.args[0] for call in factory.call_args_list], [self.a, self.b])
        self.assertEqual(first.models.generate_content.call_args, second.models.generate_content.call_args)
        self.assertEqual(stats['api_calls'], 2)
        self.assertEqual(runtime.resolve_credential(), self.b)  # Both stages share explicit session choice.
        self.assertNotIn('GEMINI_ACTIVE_KEY', self.config)  # No .env writes.
        for item in owners:
            item.__exit__.assert_called_once()

    def test_remote_media_reauthorized_for_new_project_and_never_auto_uploaded(self):
        for compatible in (True, False):
            first, second = Mock(), Mock()
            first.models.generate_content.side_effect = Failure(429, 'RequestsPerDay')
            second.files.get.return_value = SimpleNamespace(name='files/example')
            if not compatible:
                second.files.get.side_effect = Failure(403)
            factory = Mock(side_effect=[owner(first), owner(second)])
            kwargs = {'contents': {'parts': [{'file_data': {'file_uri': 'https://generativelanguage.googleapis.com/v1beta/files/example'}}]}}
            with runtime.RequestSession(self.a, lambda _: self.b, factory=factory) as session:
                if compatible:
                    runtime.generate_with_retry(session, kwargs)
                    second.models.generate_content.assert_called_once()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'unavailable.*No upload'):
                        runtime.generate_with_retry(session, kwargs)
                    second.models.generate_content.assert_not_called()
            second.files.get.assert_called_once_with(name='files/example')
            first.files.upload.assert_not_called()
            second.files.upload.assert_not_called()

    def test_sdk_file_objects_and_camel_case_media_cannot_skip_project_check(self):
        from google.genai import types
        uri = 'https://generativelanguage.googleapis.com/v1beta/files/foreign'
        forms = [types.File(name='files/foreign', uri=uri, mime_type='video/mp4'),
                 types.Part.from_uri(file_uri=uri, mime_type='video/mp4'),
                 {'parts': [{'fileData': {'fileUri': uri, 'mimeType': 'video/mp4'}}]},
                 {'file_uri': uri, 'mime_type': 'video/mp4'},
                 {'fileUri': uri, 'mimeType': 'video/mp4'}]
        for contents in forms:
            with self.subTest(content_type=type(contents).__name__):
                first, second = Mock(), Mock()
                first.models.generate_content.side_effect = Failure(429, 'RequestsPerDay')
                second.files.get.side_effect = Failure(403)
                factory = Mock(side_effect=[owner(first), owner(second)])
                with self.assertRaisesRegex(RuntimeError, 'selected project'), \
                     runtime.RequestSession(self.a, lambda _: self.b, factory=factory) as session:
                    runtime.generate_with_retry(session, {'model': 'fixed', 'contents': [contents]})
                second.files.get.assert_called_once_with(name='files/foreign')
                second.models.generate_content.assert_not_called()
                second.files.upload.assert_not_called()

    def test_aggregation_resumes_current_batch_without_repeating_completed_boundaries(self):
        df = fixtures.detailed(['Position housing'] * 6)
        ids, raw_a, raw_b = [], Mock(), Mock()
        def respond(**kwargs):
            records = json.loads(kwargs['contents'].parts[1].text)['segments']
            ids.append([row['segment_id'] for row in records])
            if len(ids) == 2:
                raise Failure(429, 'RequestsPerDay')
            return SimpleNamespace(text=json.dumps(fixtures.proposal(records)))
        raw_a.models.generate_content.side_effect = respond
        raw_b.models.generate_content.side_effect = respond
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(genai, 'Client', side_effect=[owner(raw_a), owner(raw_b)]) as factory, \
             patch.object(semantic, '_media_parts', return_value=[]):
            video = Path(directory) / 'existing.bin'
            video.write_bytes(b'synthetic input, never processed')
            stats = {}
            semantic.propose_task_groups(video, df, batch_segments=3, cache_dir=Path(directory)/'cache',
                stats=stats, refresh_suggestions=True,
                on_daily_quota=lambda _: runtime.select_credential('GEMINI_API_KEY_2'))
            self.assertEqual(ids, [[1, 2, 3], [3, 4, 5], [3, 4, 5], [5, 6]])
            self.assertEqual(stats['fresh'], 5)
            self.assertEqual(stats['api_calls'], 4)
            self.assertEqual(stats['unresolved'], 0)
            saved = {path: path.read_bytes() for path in (Path(directory)/'cache').glob('*.json')}
            factory.reset_mock()
            semantic.propose_task_groups(video, df, batch_segments=3, cache_dir=Path(directory)/'cache', stats=stats)
            factory.assert_not_called()
            self.assertEqual(stats['cache_hits'], 5)
            self.assertEqual(stats['api_calls'], 0)
            self.assertTrue(all(path.read_bytes() == value for path, value in saved.items()))
            self.assertEqual(video.read_bytes(), b'synthetic input, never processed')

    def test_recovery_changes_preserve_previous_cache_identity(self):
        records = semantic.segment_records(fixtures.detailed(['Position housing', 'Inspect fit']))
        provenance = semantic.aggregation_provenance(records, 12)
        with patch.dict(os.environ, {'RAVEE_SUGGEST_MODE': 'video', 'RAVEE_SUGGEST_CONTEXT_SECONDS': '3'}):
            identity = {'sha256': 'verified content', 'path': 'same video'}
            # Exact payload used immediately before this recovery-only change.
            payload = dict(version=semantic._BOUNDARY_CACHE_VERSION, video=identity, records=records,
                frames=[records[0]['start_frame'], records[1]['end_frame']], model=provenance['model'],
                prompt=semantic._PROMPT, generation=provenance['generation'], batch_segments=12,
                mode='video', context='3',
                implementation='5e63de13e5008d8f04311050297cb34b4e1a479c4f3c8f6f55f2811247a2d283',
                encoder='bc25e92f096f26b8af236943c0652ec432e73a524d3ac8cfac7849ffce927d2c')
            expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            self.assertEqual(semantic._boundary_cache_key(identity, records, 0, 12), expected)

    def test_cancelled_partial_aggregation_keeps_cache_and_next_explicit_run_only_requests_missing_work(self):
        df = fixtures.detailed(['Position housing'] * 6)
        ids, raw = [], Mock()
        def respond(**kwargs):
            records = json.loads(kwargs['contents'].parts[1].text)['segments']
            ids.append([r['segment_id'] for r in records])
            if len(ids) == 2:
                raise Failure(429, 'RequestsPerDay')
            return SimpleNamespace(text=json.dumps(fixtures.proposal(records)))
        raw.models.generate_content.side_effect = respond
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(genai, 'Client', side_effect=lambda **_: owner(raw)), \
             patch.object(semantic, '_media_parts', return_value=[]):
            video = Path(directory)/'video.bin'
            video.write_bytes(b'unchanged local content')
            cache = Path(directory)/'cache'
            stats, decisions = {}, Mock(return_value=None)
            semantic.propose_task_groups(video, df, batch_segments=3, cache_dir=cache,
                                         stats=stats, on_daily_quota=decisions)
            self.assertEqual(stats['fresh'], 2)
            self.assertEqual(stats['unresolved'], 3)
            self.assertEqual(stats['error_kind'], 'daily_quota')
            self.assertEqual(ids, [[1, 2, 3], [3, 4, 5]])
            decisions.assert_called_once_with(self.a)
            saved = {p: p.read_bytes() for p in cache.glob('*.json')}
            self.assertEqual(len(saved), 2)
            runtime.select_credential('GEMINI_API_KEY_2')  # Explicit later user choice.
            semantic.propose_task_groups(video, df, batch_segments=3, cache_dir=cache, stats=stats)
            self.assertEqual(ids, [[1, 2, 3], [3, 4, 5], [3, 4, 5], [5, 6]])
            self.assertEqual(stats['cache_hits'], 2)
            self.assertEqual(stats['fresh'], 3)
            self.assertEqual(stats['error_kind'], '')
            self.assertTrue(all(p.read_bytes() == value for p, value in saved.items()))


class DialogTests(ConfiguredTest):
    def test_ui_notification_error_cannot_leave_worker_waiting(self):
        desktop = Desktop()
        bridge = quota_recovery.QuotaRecovery(desktop.root,
            on_decision=Mock(side_effect=RuntimeError('synthetic notification failure')))
        result = []
        worker = Thread(target=lambda: result.append(bridge.handle_quota(self.a)), daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 2
            while not bridge.waiting and time.monotonic() < deadline:
                Event().wait(.001)
            self.assertTrue(bridge.waiting)
            with self.assertRaisesRegex(RuntimeError, 'notification failure'):
                bridge.decide('cancel')
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, [None])
        finally:
            bridge.close()
            worker.join(2)

    def test_no_alternative_key_disables_selection_and_window_close_is_cancel(self):
        self.config.pop('GEMINI_API_KEY_2')
        desktop = Desktop()
        with desktop.patch(gui_elements):
            bridge = quota_recovery.QuotaRecovery(desktop.root)
            result = []
            worker = Thread(target=lambda: result.append(bridge.handle_quota(self.a)), daemon=True)
            worker.start()
            try:
                desktop.until(lambda: bridge.dialog is not None)
                self.assertEqual(desktop.find('Button', 'Select credential').options['state'], 'disabled')
                close = bridge.dialog.protocol.call_args.args[1]
                close()
                worker.join(2)
                self.assertEqual(result, [None])
                self.assertFalse(bridge.disabled)
                self.assertEqual(bridge.last_action, 'cancel')
            finally:
                bridge.close()
                worker.join(2)

    def test_two_controllers_cannot_show_simultaneous_dialogs(self):
        desktop = Desktop()
        with desktop.patch(gui_elements):
            first, second = quota_recovery.QuotaRecovery(desktop.root), quota_recovery.QuotaRecovery(desktop.root)
            workers = [Thread(target=bridge.handle_quota, args=(self.a,), daemon=True) for bridge in (first, second)]
            for worker in workers:
                worker.start()
            try:
                desktop.until(lambda: first.dialog is not None or second.dialog is not None)
                self.assertEqual(sum(b.dialog is not None for b in (first, second)), 1)
                visible = first if first.dialog is not None else second
                remaining = second if visible is first else first
                visible.decide('cancel')
                desktop.until(lambda: remaining.dialog is not None)
                self.assertIsNone(visible.dialog)
                remaining.decide('manual')
            finally:
                first.close()
                second.close()
                for worker in workers:
                    worker.join(2)
                    self.assertFalse(worker.is_alive())

    def test_manual_cancel_and_explicit_chooser_are_nonmodal_singleton_and_release_worker(self):
        for action in ('manual', 'cancel', 'select'):
            with self.subTest(action=action):
                runtime._SESSION_CREDENTIAL_NAME = None
                desktop = Desktop()
                # Use the same Tk mock machinery; no desktop or provider calls.
                with desktop.patch(gui_elements):
                    paused, resumed, decisions = Mock(), Mock(), Mock()
                    bridge = quota_recovery.QuotaRecovery(desktop.root, on_pause=paused,
                        on_resume=resumed, on_decision=decisions)
                    result = []
                    worker = Thread(target=lambda: result.append(bridge.handle_quota(self.a)), daemon=True)
                    worker.start()
                    desktop.until(lambda: bridge.dialog is not None)
                    dialog = bridge.dialog
                    self.assertTrue(worker.is_alive())
                    # Duplicate failures cannot open another dialog or queue another request.
                    self.assertIsNone(bridge.handle_quota(self.a))
                    desktop.pump()
                    self.assertIs(bridge.dialog, dialog)
                    self.assertEqual(bridge.interruptions, 1)
                    dialog.grab_set.assert_not_called()
                    dialog.wait_window.assert_not_called()
                    dialog.mainloop.assert_not_called()
                    if action == 'select':
                        desktop.click('Select credential')
                    else:
                        desktop.click('Continue manually' if action == 'manual' else 'Cancel')
                    worker.join(2)
                    self.assertFalse(worker.is_alive())
                    self.assertEqual(result, [self.b if action == 'select' else None])
                    self.assertEqual(bridge.disabled, action == 'manual')
                    self.assertEqual(runtime.resolve_credential(), self.b if action == 'select' else self.a)
                    self.assertEqual(resumed.call_count, int(action == 'select'))
                    paused.assert_called_once()
                    decisions.assert_called_once_with(action)
                    display = repr([widget.options for widget in desktop.widgets])
                    for key in ('synthetic-alpha', 'synthetic-beta'):
                        self.assertNotIn(key, display)
                    self.assertIn('Project 2', display)
                    bridge.close()
                    self.assertFalse(desktop.timers)

    def test_closing_parent_releases_paused_worker_without_requests_or_rotation(self):
        desktop = Desktop()
        bridge = quota_recovery.QuotaRecovery(desktop.root)
        result = []
        worker = Thread(target=lambda: result.append(bridge.handle_quota(self.a)), daemon=True)
        worker.start()
        deadline = time.monotonic() + 2
        while not bridge.waiting and time.monotonic() < deadline:
            Event().wait(.001)
        self.assertTrue(bridge.waiting)
        bridge.close()
        worker.join(2)
        self.assertEqual(result, [None])
        self.assertIsNone(bridge.handle_quota(self.a))
        self.assertFalse(desktop.timers)


class ReviewWorkflowTests(ConfiguredTest):
    def test_label_review_preserves_commits_drafts_selection_and_cached_suggestions_for_all_choices(self):
        for action in ('manual', 'cancel', 'select'):
            with self.subTest(action=action):
                runtime._SESSION_CREDENTIAL_NAME = None
                desktop, raw_a, raw_b = Desktop(), Mock(), Mock()
                raw_a.models.generate_content.side_effect = [label_fixtures.response(label='Position housing'),
                    Failure(429, 'RequestsPerDay'), label_fixtures.response(label='Inspect housing')]
                raw_b.models.generate_content.return_value = label_fixtures.response(label='Inspect housing')
                def factory(**kw):
                    self.assertNotEqual(current_thread().name, 'MainThread')
                    return owner(raw_b if kw['api_key'] == self.b.key else raw_a)
                def exercise():
                    entry, tree = desktop.find('Entry'), desktop.find('Treeview')
                    ai = desktop.find('Button', 'Suggest label')
                    desktop.click('Suggest label')
                    desktop.until(lambda: any(w.options.get('text') == 'Use suggestion' for w in desktop.widgets))
                    entry.value = 'Human first'
                    desktop.click('Commit & Next')
                    entry.value = 'Pending second'
                    desktop.click('Suggest label')
                    desktop.until(lambda: any(w.kind == 'Toplevel' for w in desktop.widgets))
                    desktop.click('Previous')
                    entry.value = 'Edited first, uncommitted'
                    entry.bindings['<KeyRelease>'](None)
                    self.assertEqual(gui_elements.LABEL, {1: 'Human first'})
                    self.assertEqual(tree.selection(), ('0',))
                    # The pending-worker guard also works if invoked programmatically.
                    desktop.find('Button', 'Suggesting...').options['command']()
                    self.assertEqual(raw_a.models.generate_content.call_count, 2)
                    desktop.click({'manual': 'Continue manually', 'cancel': 'Cancel', 'select': 'Select credential'}[action])
                    desktop.until(lambda: ai.options.get('text') == 'Use suggestion' and ai.options.get('state') == 'normal')
                    self.assertEqual(entry.value, 'Edited first, uncommitted')
                    self.assertEqual(tree.selection(), ('0',))
                    self.assertEqual(gui_elements.LABEL, {1: 'Human first'})
                    # Cached first suggestion is still available without another request.
                    desktop.click('Commit & Next')
                    self.assertEqual(entry.value, 'Pending second')
                    if action == 'manual':
                        disabled = desktop.find('Button', 'AI disabled for this review')
                        self.assertEqual(disabled.options['state'], 'disabled')
                        disabled.options['command']()
                    elif action == 'cancel':
                        desktop.click('Suggest label')  # Explicit new request; first segment is not repeated.
                        desktop.until(lambda: ai.options.get('text') == 'Use suggestion')
                    else:
                        desktop.until(lambda: ai.options.get('text') == 'Use suggestion')
                    self.assertEqual(entry.value, 'Pending second')  # Async result never inserts itself.
                    desktop.click('Commit & Finish')
                desktop.root.mainloop.side_effect = exercise
                with desktop.patch(gui_elements), patch.object(genai, 'Client', side_effect=factory), \
                     patch.object(labels, '_target_context_windows', side_effect=lambda path, start, end:
                                  (15, [('TARGET SEGMENT', start, end)])), \
                     patch.object(labels, '_make_clip', return_value=b'synthetic media'):
                    gui_elements.label_video_clusters('synthetic.avi', [
                        dict(start_frame=0, end_frame=14, cluster_id=0),
                        dict(start_frame=15, end_frame=29, cluster_id=1)])
                self.assertEqual(gui_elements.LABEL, {1: 'Edited first, uncommitted', 2: 'Pending second'})
                self.assertEqual(raw_a.models.generate_content.call_count + raw_b.models.generate_content.call_count,
                                 2 if action == 'manual' else 3)

    def test_semantic_review_preserves_history_pending_form_and_selection_on_all_choices(self):
        for action in ('manual', 'cancel', 'select'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                runtime._SESSION_CREDENTIAL_NAME = None
                desktop, raw_a, raw_b = Desktop(), Mock(), Mock()
                raw_a.models.generate_content.side_effect = Failure(429, 'RequestsPerDay')
                def respond(**kw):
                    records = json.loads(kw['contents'].parts[1].text)['segments']
                    return SimpleNamespace(text=json.dumps(fixtures.proposal(records)))
                raw_b.models.generate_content.side_effect = respond
                def factory(**kw):
                    self.assertNotEqual(current_thread().name, 'MainThread')
                    return owner(raw_b if kw['api_key'] == self.b.key else raw_a)
                video = Path(directory)/'existing.bin'
                video.write_bytes(b'never processed')
                df = fixtures.detailed(['Position housing', 'Adjust housing', 'Inspect housing'])
                snapshot = df.copy(deep=True)
                metadata = {'existing_review': 'preserved', 'ai_provenance': {'model': 'old-confirmed-model'}}
                original_metadata = dict(metadata)
                original_propose = semantic.propose_task_groups
                def propose(*args, **kwargs):
                    return original_propose(*args, cache_dir=Path(directory)/'cache', **kwargs)
                def exercise():
                    entry, tree = desktop.find('Entry'), desktop.find('Treeview')
                    reason = desktop.find('Text')
                    status = next(w for w in desktop.widgets if w.kind == 'StringVar' and
                                  w.options.get('value', '').startswith('Review groups manually'))
                    entry.value = 'Committed task edit'
                    desktop.click('Apply edits')
                    tree.selection_set('1')
                    tree.bindings['<<TreeviewSelect>>'](None)
                    entry.value, reason.value = 'Pending task edit', 'Pending explanation'
                    desktop.click('Suggest task groups')
                    desktop.until(lambda: any(w.kind == 'Toplevel' for w in desktop.widgets))
                    self.assertEqual(entry.options['state'], 'normal')
                    self.assertEqual(tree.selection(), ('1',))
                    self.assertEqual(entry.value, 'Pending task edit')
                    self.assertEqual(reason.value, 'Pending explanation')
                    # Confirm/merge/split/apply remain available during the pause.
                    self.assertEqual(desktop.find('Button', 'Confirm task groups').options['state'], 'normal')
                    desktop.click('Suggest task groups')
                    self.assertEqual(raw_a.models.generate_content.call_count, 1)
                    desktop.click({'manual': 'Continue manually', 'cancel': 'Cancel', 'select': 'Select credential'}[action])
                    if action == 'select':
                        desktop.until(lambda: any(w.options.get('text') == 'Load recovered proposals' for w in desktop.widgets))
                    else:
                        desktop.until(lambda: status.value.startswith('Daily quota request stopped'))
                    self.assertEqual(entry.value, 'Pending task edit')
                    self.assertEqual(reason.value, 'Pending explanation')
                    self.assertEqual(tree.selection(), ('1',))
                    self.assertEqual(len(tree.children), 3)
                    self.assertEqual(metadata, original_metadata)  # Unloaded proposals are not adopted provenance.
                    if action == 'select':
                        desktop.click('Load recovered proposals')
                        self.assertEqual(len(tree.children), 1)
                        desktop.click('Undo')
                        # Reproduce deferred Tk selection notifications before idle.
                        tree.bindings['<<TreeviewSelect>>'](None)
                        desktop.pump()
                        self.assertEqual(len(tree.children), 3)
                        self.assertEqual(entry.value, 'Pending task edit')
                        self.assertEqual(reason.value, 'Pending explanation')
                        self.assertEqual(tree.selection(), ('1',))
                        self.assertEqual(metadata, original_metadata)
                    else:
                        self.assertEqual(desktop.find('Button', 'Suggest task groups').options['state'],
                                         'disabled' if action == 'manual' else 'normal')
                    desktop.click('Confirm task groups')
                desktop.root.mainloop.side_effect = exercise
                with desktop.patch(semantic_review), patch.object(genai, 'Client', side_effect=factory), \
                     patch.object(semantic, '_media_parts', return_value=[]), \
                     patch.object(semantic_review, 'propose_task_groups', side_effect=propose):
                    groups = semantic_review.review_task_groups(video, df, run_metadata=metadata)
                self.assertEqual(groups[0]['suggested_task_label'], 'Committed task edit')
                self.assertEqual(groups[1]['suggested_task_label'], 'Pending task edit')
                self.assertEqual(groups[1]['explanation'], 'Pending explanation')
                self.assertEqual(raw_a.models.generate_content.call_count, 1)
                self.assertEqual(raw_b.models.generate_content.call_count, int(action == 'select'))
                self.assertTrue(df.equals(snapshot))
                self.assertFalse(desktop.timers)


if __name__ == '__main__':
    unittest.main()
