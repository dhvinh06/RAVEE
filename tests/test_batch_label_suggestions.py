"""Offline Suggest All: synthetic video bytes, mocked metadata/provider/Tk only."""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from threading import Event, current_thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import batch_label_suggester as batch
import gemini_runtime as runtime
from gui import gui_elements as gui
from toolbox import main as app
from tests.test_quota_recovery import ConfiguredTest, Desktop, Failure, owner


def boundaries(count=20):
    return [dict(segment_id=i + 1, cluster_id=i % 3, start_frame=i * 15, end_frame=i * 15 + 14)
            for i in range(count)]


def payload(records):
    return {'segments': [dict(segment_id=r['segment_id'], label=f"Move object {r['segment_id']}",
        evidence='The object moves inside this exact segment.', uncertainty='low',
        target_visible=True, bow_action='not_applicable') for r in records]}


def response(records):
    return SimpleNamespace(text=json.dumps(payload(records)))


class BatchFixture(ConfiguredTest):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="RAVEE batch ' & ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / 'synthetic.avi'
        self.video.write_bytes(b'not a real video; never decoded or processed')
        self.records = boundaries()
        self.cache = self.root / 'batch_cache'
        self.raw = Mock()
        self.raw.models.generate_content.return_value = response(self.records)
        for name, value in (('_video_info', Mock(return_value=(15., 300))),
                            ('create_client', Mock(side_effect=lambda credential: owner(self.raw)))):
            patcher = patch.object(batch, name, value)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        # Metadata, GUI preview and provider construction must all be explicitly mocked.
        guard = patch.object(batch.cv2, 'VideoCapture', side_effect=AssertionError('Real video access forbidden'))
        guard.start()
        self.addCleanup(guard.stop)

    def suggest(self, **kwargs):
        return batch.suggest_all_activities(self.video, self.records, cache_dir=self.cache, **kwargs)


class RequestTests(BatchFixture):
    def test_twenty_segments_use_one_generation_and_one_complete_video(self):
        stats = {}
        result = self.suggest(stats=stats)
        self.assertEqual(result, {i: f'Move object {i}' for i in range(1, 21)})
        self.assertEqual(stats['api_calls'], 1)
        self.raw.models.generate_content.assert_called_once()
        self.create_client.assert_called_once_with(self.a)
        request = self.raw.models.generate_content.call_args.kwargs
        self.assertEqual(request['model'], 'unchanged-model')
        parts = request['contents'].parts
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[2].inline_data.data, self.video.read_bytes())
        self.assertEqual(parts[2].inline_data.mime_type, 'video/avi')
        self.assertEqual(parts[2].video_metadata.fps, 3.)
        timeline = json.loads(parts[1].text)
        self.assertEqual(len(timeline['segments']), 20)
        self.assertEqual(timeline['segments'][0], dict(segment_id=1, start_frame=0,
            end_frame=14, start_seconds=0., end_seconds_exclusive=1.))
        self.assertEqual(timeline['segments'][-1]['end_seconds_exclusive'], 20.)
        schema = request['config'].response_json_schema['properties']['segments']
        self.assertEqual(schema['minItems'], 20)
        self.assertEqual(schema['maxItems'], 20)
        self.assertGreaterEqual(request['config'].max_output_tokens, 6144)
        self.raw.files.upload.assert_not_called()

    def test_repeated_clusters_are_separate_and_explicit_ids_do_not_depend_on_response_order(self):
        items = payload(self.records)
        items['segments'].reverse()
        self.raw.models.generate_content.return_value.text = json.dumps(items)
        self.assertEqual(self.suggest()[20], 'Move object 20')
        legacy = [dict(start_frame=i * 2, end_frame=i * 2 + 1, cluster_id=0) for i in range(3)]
        self.assertEqual([r['segment_id'] for r in batch.segment_boundaries(legacy)], [1, 2, 3])

    def test_invalid_boundaries_fail_before_any_request(self):
        for records in ([], [dict(segment_id=1, start_frame=-1, end_frame=5)],
                        [dict(segment_id=True, start_frame=0, end_frame=5)],
                        [dict(segment_id=1, start_frame=1.5, end_frame=5)],
                        [dict(segment_id=1, start_frame=0, end_frame=5),
                         dict(segment_id=1, start_frame=6, end_frame=7)],
                        [dict(segment_id=1, start_frame=0, end_frame=5),
                         dict(segment_id=2, start_frame=5, end_frame=7)]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                batch.suggest_all_activities(self.video, records, cache_dir=self.cache)
        self.create_client.assert_not_called()
        self._video_info.assert_not_called()

    def test_oversized_unsupported_or_out_of_video_input_never_splits_or_sends(self):
        with patch.object(batch, '_MAX_VIDEO_BYTES', 3), self.assertRaisesRegex(ValueError, 'inline video limit'):
            self.suggest()
        unsupported = self.root / 'unsupported.bin'
        unsupported.write_bytes(self.video.read_bytes())
        with self.assertRaisesRegex(ValueError, 'container'):
            batch.suggest_all_activities(unsupported, self.records, cache_dir=self.cache)
        self._video_info.return_value = (15., 200)
        with self.assertRaisesRegex(ValueError, 'beyond'):
            self.suggest()
        self.create_client.assert_not_called()

    def test_api_error_is_redacted_and_does_not_cache_a_partial_batch(self):
        self.raw.models.generate_content.side_effect = RuntimeError('failure synthetic-alpha')
        with self.assertRaises(RuntimeError) as error:
            self.suggest()
        self.assertNotIn('synthetic-alpha', str(error.exception))
        self.assertFalse(self.cache.exists())


class ResponseTests(BatchFixture):
    def validate(self, value):
        return batch._validate_response(SimpleNamespace(text=json.dumps(value)), self.records, 'offline')[0]

    def test_missing_extra_duplicate_unknown_string_and_boolean_ids_are_rejected_atomically(self):
        for mutate in (lambda p: p['segments'].pop(),
                       lambda p: p['segments'].append(deepcopy(p['segments'][0])),
                       lambda p: p['segments'][1].update(segment_id=1),
                       lambda p: p['segments'][0].update(segment_id=999),
                       lambda p: p['segments'][0].update(segment_id='1'),
                       lambda p: p['segments'][0].update(segment_id=True),
                       lambda p: p['segments'][0].update(segment_id=1.),
                       lambda p: p.update(unrequested='field')):
            candidate = payload(self.records)
            mutate(candidate)
            with self.assertRaises(runtime.InvalidResponseError):
                self.validate(candidate)

    def test_invalid_labels_evidence_and_field_types_are_rejected(self):
        for changes in (dict(label='word'), dict(label='one two three four five six seven'),
                        dict(evidence=[]), dict(target_visible='true'), dict(uncertainty='maybe'),
                        dict(bow_action='unknown'), dict(label='Expose synthetic-alpha'), dict(extra='field')):
            candidate = payload(self.records)
            candidate['segments'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(runtime.InvalidResponseError):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.validate(candidate)

    def test_malformed_empty_duplicate_fields_blocked_and_truncated_responses_are_rejected(self):
        for text in ('', 'null', '[]', '{invalid', '{"segments":[],"segments":[]}'):
            with self.assertRaises(runtime.InvalidResponseError):
                batch._validate_response(SimpleNamespace(text=text), self.records, 'offline')
        for reason in ('MAX_TOKENS', 'SAFETY', 'OTHER'):
            candidate = response(self.records)
            candidate.candidates = [SimpleNamespace(finish_reason=reason, finish_message=None)]
            with self.assertRaises(runtime.InvalidResponseError):
                batch._validate_response(candidate, self.records, 'offline')
        candidate = response(self.records)
        candidate.prompt_feedback = SimpleNamespace(block_reason='SAFETY')
        with self.assertRaises(runtime.InvalidResponseError):
            batch._validate_response(candidate, self.records, 'offline')

    def test_uncertain_targets_remain_present_as_unclear_activity(self):
        candidate = payload(self.records)
        candidate['segments'][0]['uncertainty'] = 'high'
        candidate['segments'][1]['target_visible'] = False
        candidate['segments'][2]['evidence'] = ''
        labels = self.validate(candidate)
        self.assertEqual(len(labels), 20)
        self.assertEqual([labels[i] for i in (1, 2, 3)], ['Unclear activity'] * 3)

    def test_invalid_model_output_is_not_retried_or_cached(self):
        candidate = payload(self.records)
        candidate['segments'].pop()
        self.raw.models.generate_content.return_value.text = json.dumps(candidate)
        with self.assertRaises(runtime.InvalidResponseError):
            self.suggest()
        self.raw.models.generate_content.assert_called_once()
        self.assertFalse(self.cache.exists())


class CacheTests(BatchFixture):
    def test_warm_cache_avoids_provider_media_and_credentials_and_remains_separate(self):
        other_cache = self.root / 'semantic_boundary_cache' / 'existing.json'
        other_cache.parent.mkdir()
        other_cache.write_bytes(b'preserve old cache exactly')
        before = self.video.read_bytes()
        first = self.suggest()
        self.create_client.reset_mock()
        self._video_info.reset_mock()
        self.config.pop('GEMINI_API_KEY')  # An already cached result needs no credential.
        self.config.pop('GEMINI_API_KEY_2')
        stats = {}
        self.assertEqual(first, self.suggest(stats=stats))
        self.assertTrue(stats['cache_hit'])
        self.assertEqual(stats['api_calls'], 0)
        self.create_client.assert_not_called()
        self._video_info.assert_not_called()
        self.assertEqual(before, self.video.read_bytes())
        self.assertEqual(other_cache.read_bytes(), b'preserve old cache exactly')

    def test_video_boundaries_model_prompt_and_generation_changes_have_distinct_cache_keys(self):
        self.suggest()
        initial = {p: p.read_bytes() for p in self.cache.glob('*.json')}
        self.video.write_bytes(b'changed video identity')
        self.suggest()
        self.records[0]['end_frame'] = 13
        self.suggest()
        self.config['GEMINI_LABEL_MODEL'] = 'another-label-model'
        self.suggest()
        with patch.object(batch, '_PROMPT', batch._PROMPT + ' Updated batch instructions.'):
            self.suggest()
        original = batch._generation_config
        def changed(*args):
            config = original(*args)
            config.temperature = .2
            return config
        with patch.object(batch, '_generation_config', side_effect=changed):
            self.suggest()
        self.assertEqual(self.raw.models.generate_content.call_count, 6)
        self.assertEqual(len(list(self.cache.glob('*.json'))), 6)
        self.assertTrue(all(path.read_bytes() == raw for path, raw in initial.items()))

    def test_switching_credential_reuses_the_same_complete_batch_cache(self):
        self.suggest()
        before = {p: p.read_bytes() for p in self.cache.glob('*.json')}
        runtime.select_credential('GEMINI_API_KEY_2')
        self.create_client.reset_mock()
        self.suggest()
        self.create_client.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.cache.glob('*.json')})

    def test_corrupt_incomplete_and_semantically_invalid_cache_entries_trigger_one_whole_request(self):
        self.suggest()
        cached = next(self.cache.glob('*.json'))
        valid = json.loads(cached.read_text())
        invalid_response = deepcopy(valid)
        invalid_response['response']['segments'].pop()
        raw = json.dumps(invalid_response['response'], sort_keys=True)
        invalid_response['sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        for candidate in ('{truncated', json.dumps(dict(valid, complete=False)),
                          json.dumps(dict(valid, sha256='corrupt')), json.dumps(invalid_response)):
            cached.write_text(candidate)
            self.raw.models.generate_content.reset_mock()
            self.assertEqual(len(self.suggest()), 20)
            self.raw.models.generate_content.assert_called_once()

    def test_cache_write_failure_keeps_valid_review_results_and_cleans_temporary_file(self):
        with patch.object(batch.os, 'replace', side_effect=PermissionError('offline read-only cache')):
            self.assertEqual(len(self.suggest()), 20)
        self.assertEqual(list(self.cache.iterdir()), [])


class GuiTests(BatchFixture):
    def run_gui(self, desktop, records=None):
        original = batch.suggest_all_activities
        with desktop.patch(gui), patch.object(gui, 'suggest_all_activities', side_effect=lambda *a, **kw:
                original(*a, cache_dir=self.cache, **kw)):
            gui.label_video_clusters(str(self.video), records or self.records)

    def finish_manually(self, desktop, start=1, count=20):
        entry = desktop.find('Entry')
        for identifier in range(start, count + 1):
            entry.value = f'Human segment {identifier}'
            desktop.find('Button', 'Commit & Finish' if identifier == count else 'Commit & Next').options['command']()

    def test_batch_is_responsive_preserves_commits_drafts_selection_and_needs_explicit_use_then_commit(self):
        desktop, started, release = Desktop(), Event(), Event()
        def generate(**kwargs):
            self.assertNotEqual(current_thread().name, 'MainThread')
            started.set()
            if not release.wait(2):
                raise AssertionError('GUI did not remain responsive')
            return response(self.records)
        self.raw.models.generate_content.side_effect = generate
        def exercise():
            entry, tree = desktop.find('Entry'), desktop.find('Treeview')
            individual, all_button = desktop.find('Button', 'Suggest label'), desktop.find('Button', 'Suggest All')
            self.assertEqual(individual.grid.call_args.kwargs['row'], all_button.grid.call_args.kwargs['row'])
            self.assertEqual(all_button.grid.call_args.kwargs['column'], individual.grid.call_args.kwargs['column'] + 1)
            entry.value = 'Reviewed first'
            desktop.click('Commit & Next')
            entry.value = 'Pending second draft'
            desktop.click('Suggest All')
            desktop.until(started.is_set)
            self.assertEqual(all_button.options['state'], 'disabled')
            self.assertEqual(individual.options['state'], 'disabled')
            all_button.options['command']()  # Even direct callback invocation is guarded.
            individual.options['command']()
            desktop.click('Previous')
            entry.value = 'Uncommitted first edit'
            release.set()
            desktop.until(lambda: desktop.find('Button', 'Use batch suggestion').options['state'] == 'normal')
            self.assertEqual(gui.LABEL, {1: 'Reviewed first'})
            self.assertEqual(entry.value, 'Uncommitted first edit')
            self.assertEqual(tree.selection(), ('0',))
            latest = {call.args[0]: call.kwargs['values'] for call in tree.item.call_args_list}
            self.assertEqual(latest['19'][3], 'Move object 20')
            self.raw.models.generate_content.assert_called_once()
            desktop.click('Use batch suggestion')
            self.assertEqual(entry.value, 'Move object 1')
            self.assertEqual(gui.LABEL, {1: 'Reviewed first'})
            desktop.click('Commit & Next')
            self.assertEqual(entry.value, 'Pending second draft')
            self.assertEqual(gui.LABEL, {1: 'Move object 1'})
            self.finish_manually(desktop, start=2)
        desktop.root.mainloop.side_effect = exercise
        try:
            self.run_gui(desktop)
        finally:
            release.set()

    def test_individual_v3_and_its_cached_suggestion_survive_batch_results(self):
        desktop = Desktop()
        def exercise():
            individual_button = desktop.find('Button', 'Suggest label')
            desktop.click('Suggest label')
            desktop.until(lambda: individual_button.options.get('text') == 'Use suggestion' and
                          individual_button.options['state'] == 'normal')
            desktop.click('Suggest All')
            desktop.until(lambda: desktop.find('Button', 'Use batch suggestion').options['state'] == 'normal')
            desktop.click('Use suggestion')
            self.assertEqual(desktop.find('Entry').value, 'Inspect housing')
            self.assertEqual(gui.LABEL, {})
            desktop.click('Use batch suggestion')
            self.assertEqual(desktop.find('Entry').value, 'Move object 1')
            self.assertEqual(gui.LABEL, {})
            self.finish_manually(desktop)
        desktop.root.mainloop.side_effect = exercise
        with patch.object(gui, 'suggest_activity', return_value='Inspect housing') as individual:
            self.run_gui(desktop)
        individual.assert_called_once()
        self.assertIn('on_daily_quota', individual.call_args.kwargs)

    def test_batch_cannot_start_while_individual_request_is_pending(self):
        desktop, started, release = Desktop(), Event(), Event()
        def individual(*args, **kwargs):
            started.set()
            release.wait(2)
            return 'Inspect housing'
        def exercise():
            desktop.click('Suggest label')
            desktop.until(started.is_set)
            all_button = desktop.find('Button', 'Suggest All')
            self.assertEqual(all_button.options['state'], 'disabled')
            all_button.options['command']()
            self.create_client.assert_not_called()
            release.set()
            desktop.until(lambda: all_button.options['state'] == 'normal')
            self.finish_manually(desktop)
        desktop.root.mainloop.side_effect = exercise
        with patch.object(gui, 'suggest_activity', side_effect=individual):
            self.run_gui(desktop)

    def test_batch_uses_existing_quota_dialog_for_manual_cancel_and_selected_credential(self):
        for action in ('manual', 'cancel', 'select'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                runtime._SESSION_CREDENTIAL_NAME = None
                self.cache = Path(directory) / 'cache'
                desktop, second = Desktop(), Mock()
                self.raw.models.generate_content.side_effect = Failure(429, 'RequestsPerDay')
                second.models.generate_content.return_value = response(self.records)
                self.create_client.side_effect = lambda credential: owner(second if credential == self.b else self.raw)
                def exercise():
                    entry, tree = desktop.find('Entry'), desktop.find('Treeview')
                    entry.value = 'Human draft'
                    desktop.click('Suggest All')
                    desktop.until(lambda: any(w.kind == 'Toplevel' for w in desktop.widgets))
                    self.assertEqual(entry.value, 'Human draft')
                    desktop.click({'manual': 'Continue manually', 'cancel': 'Cancel', 'select': 'Select credential'}[action])
                    if action == 'select':
                        desktop.until(lambda: desktop.find('Button', 'Use batch suggestion').options['state'] == 'normal')
                        self.assertEqual(second.models.generate_content.call_args.kwargs,
                                         self.raw.models.generate_content.call_args.kwargs)
                    else:
                        desktop.until(lambda: any(w.kind == 'StringVar' and
                            w.value.startswith('Suggest All unavailable') for w in desktop.widgets))
                        all_button = desktop.find('Button', 'Suggest All')
                        self.assertEqual(all_button.options['state'], 'disabled' if action == 'manual' else 'normal')
                        if action == 'manual':
                            all_button.options['command']()
                        second.models.generate_content.assert_not_called()
                    self.assertEqual(entry.value, 'Human draft')
                    self.assertEqual(tree.selection(), ('0',))
                    self.assertEqual(gui.LABEL, {})
                    self.finish_manually(desktop)
                desktop.root.mainloop.side_effect = exercise
                self.run_gui(desktop)

    def test_actual_main_entrypoint_opens_the_wired_button_in_demo_and_non_demo_flows(self):
        for demo in (True, False):
            with self.subTest(demo=demo):
                desktop = Desktop()
                clusters = np.repeat(np.arange(20) % 3, 15)
                c = clusters.reshape(-1, 1)
                seen = []
                def exercise():
                    button = desktop.find('Button', 'Suggest All')
                    self.assertTrue(callable(button.options['command']))
                    seen.append(True)
                    desktop.click('Suggest All')
                    desktop.until(lambda: desktop.find('Button', 'Use batch suggestion').options['state'] == 'normal')
                    self.finish_manually(desktop)
                desktop.root.mainloop.side_effect = exercise
                original = batch.suggest_all_activities
                prepared = (None, {'case': dict(video_name='synthetic', case_id=1, video_path=str(self.video))},
                            {'case': (c, [3], clusters)})
                with desktop.patch(gui), contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(gui, 'suggest_all_activities', side_effect=lambda *a, **kw:
                                  original(*a, cache_dir=self.cache, **kw)), \
                     patch.object(app, 'DEMO', demo), patch.object(app, 'demoPath', str(self.root)), \
                     patch.object(app.np, 'genfromtxt', side_effect=[c, np.array([3])]), \
                     patch.object(gui, 'inputboxList', side_effect=lambda: setattr(gui, 'NUM_CLUSTER', 3)), \
                     patch.object(app.np, 'savetxt'), patch.object(app.os, 'makedirs'), \
                     patch.object(app, 'extractLog', return_value=pd.DataFrame()), \
                     patch.object(app, 'review_semantic_task_log', return_value=pd.DataFrame()), \
                     patch.object(app, 'export_two_event_logs'), patch.object(app, 'showProcessModel'), \
                     patch.object(app, 'prepare_cases', side_effect=AssertionError('No pipeline allowed')):
                    # Both branches call the real run_snippets -> label_video_clusters.
                    if demo:
                        (self.root / 'scan_video_reduced.avi').write_bytes(self.video.read_bytes())
                    app.main(prepared=prepared)
                self.assertEqual(seen, [True])


if __name__ == '__main__':
    unittest.main()
