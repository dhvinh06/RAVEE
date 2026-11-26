"""Offline credentials, quota policy, project resources and intermediate tasks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from google import genai
from google.genai import types

import gemini_runtime as runtime
import label_suggester as labels
from tests import test_semantic_aggregation as fixtures
from toolbox import semantic_aggregation as semantic
from toolbox.pipeline_cache import StageCache


class Failure(Exception):
    def __init__(self, code, quota=None, retry_after=None):
        super().__init__(f'HTTP {code}')
        self.code = code
        self.details = [{'quotaId': quota}] if quota else []
        self.response = SimpleNamespace(headers={'Retry-After': str(retry_after)} if retry_after else {})


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(GEMINI_API_KEY='test-only-alpha', GEMINI_API_KEY_2='test-only-beta',
                           GEMINI_API_KEY_RESEARCH='test-only-research', GEMINI_MODEL='configured-default')
        patcher = patch.object(runtime, 'configuration', side_effect=lambda: self.config.copy())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_legacy_google_precedence_and_explicit_first_key(self):
        self.assertEqual(runtime.resolve_credential().key, 'test-only-alpha')
        self.config['GOOGLE_API_KEY'] = 'test-only-google'
        self.assertEqual(runtime.resolve_credential().key, 'test-only-google')
        self.config['GEMINI_ACTIVE_KEY'] = '1'
        self.assertEqual(runtime.resolve_credential().key, 'test-only-alpha')

    def test_numeric_named_selection_and_safe_repr(self):
        for selector, expected in [('2', 'test-only-beta'), ('research', 'test-only-research')]:
            self.config['GEMINI_ACTIVE_KEY'] = selector
            credential = runtime.resolve_credential()
            self.assertEqual(credential.key, expected)
            self.assertNotIn(expected, repr(credential))

    def test_bad_selection_never_falls_back_to_another_project(self):
        for selector in ('9', 'private/invalid'):
            self.config['GEMINI_ACTIVE_KEY'] = selector
            with self.assertRaises(RuntimeError) as error:
                runtime.resolve_credential()
            self.assertNotIn('test-only-alpha', str(error.exception))

    def test_both_stages_share_client_factory_and_independent_models(self):
        self.config.update(GEMINI_ACTIVE_KEY='2', GEMINI_LABEL_MODEL='label-model',
                           GEMINI_AGGREGATION_MODEL='aggregation-model')
        with patch.object(genai, 'Client') as factory:
            runtime.create_client()
        self.assertEqual(factory.call_args.kwargs['api_key'], 'test-only-beta')
        self.assertEqual(factory.call_args.kwargs['http_options'].retry_options.attempts, 1)
        self.assertIs(labels.create_client, semantic.create_client)
        self.assertEqual(runtime.get_model('label'), 'label-model')
        self.assertEqual(runtime.get_model('aggregation'), 'aggregation-model')
        del self.config['GEMINI_LABEL_MODEL']
        self.assertEqual(runtime.get_model('label'), 'configured-default')

    def test_all_keys_are_redacted_including_inactive_profile_and_api_exception(self):
        self.config['GEMINI_ACTIVE_KEY'] = '2'
        text = runtime.redact_text('test-only-alpha test-only-beta test-only-research')
        self.assertNotIn('test-only-', text)
        with self.assertLogs(labels.__name__, level='ERROR') as logs:
            labels._log_api_error('model', RuntimeError('Problem with test-only-research'))
        self.assertNotIn('test-only-research', '\n'.join(logs.output))

    def test_credential_switch_preserves_ai_keys_and_verified_local_cache(self):
        records = fixtures.detailed(['Position housing', 'Seat housing'])
        identity = {'path': 'same-local-video', 'sha256': 'same-content'}
        first = semantic._boundary_cache_key(identity, semantic.segment_records(records), 0, 12)
        with tempfile.TemporaryDirectory() as directory:
            store = StageCache(directory)
            path = store.file('example', {'source': 'content'}, 'artifact',
                              lambda p: p.write_bytes(b'completed'), lambda _: None)
            before = path.read_bytes()
            self.config['GEMINI_ACTIVE_KEY'] = '2'
            runtime.resolve_credential()
            second = semantic._boundary_cache_key(identity, semantic.segment_records(records), 0, 12)
            self.assertEqual(first, second)
            self.assertEqual(store.lookup('example', {'source': 'content'}, 'artifact'), path)
            self.assertEqual(path.read_bytes(), before)
        self.config['GEMINI_AGGREGATION_MODEL'] = 'another-aggregation-model'
        self.assertNotEqual(first, semantic._boundary_cache_key(identity, semantic.segment_records(records), 0, 12))
        key = semantic._boundary_cache_key(identity, semantic.segment_records(records), 0, 12)
        self.config['GEMINI_LABEL_MODEL'] = 'different-label-model'
        self.assertEqual(key, semantic._boundary_cache_key(identity, semantic.segment_records(records), 0, 12))

    def test_real_stage_entrypoints_use_same_active_profile_and_warm_cache_survives_switch(self):
        self.config.update(GEMINI_ACTIVE_KEY='2', GEMINI_LABEL_MODEL='label-model',
                           GEMINI_AGGREGATION_MODEL='aggregation-model')
        detailed = fixtures.detailed(['Position housing', 'Inspect seated fit'])
        records = semantic.segment_records(detailed)
        with tempfile.TemporaryDirectory() as directory, patch.object(genai, 'Client') as factory, \
             patch.object(labels, '_target_context_windows', return_value=(15, [('TARGET SEGMENT', 0, 14)])), \
             patch.object(labels, '_make_clip', return_value=b'synthetic media'), \
             patch.object(semantic, '_media_parts', return_value=[]):
            video = Path(directory) / 'existing.avi'
            video.write_bytes(b'source identity only')
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.side_effect = [SimpleNamespace(text=json.dumps(dict(
                label='Position housing', evidence='The housing moves into its socket.', uncertainty='low',
                target_visible=True, bow_action='not_applicable'))),
                SimpleNamespace(text=json.dumps(fixtures.proposal(records)))]
            self.assertEqual(labels.suggest_activity(video, 0, 14), 'Position housing')
            stats = {}
            semantic.propose_task_groups(video, detailed, cache_dir=Path(directory) / 'cache', stats=stats)
            self.assertEqual([call.kwargs['api_key'] for call in factory.call_args_list],
                             ['test-only-beta', 'test-only-beta'])
            self.assertEqual([call.kwargs['model'] for call in client.models.generate_content.call_args_list],
                             ['label-model', 'aggregation-model'])
            factory.reset_mock()
            self.config['GEMINI_ACTIVE_KEY'] = '1'
            semantic.propose_task_groups(video, detailed, cache_dir=Path(directory) / 'cache', stats=stats)
            factory.assert_not_called()
            self.assertEqual(stats['cache_hits'], 1)
            self.assertEqual(stats['api_calls'], 0)


class HotConfigurationTests(unittest.TestCase):
    def test_dotenv_changes_are_seen_without_reprocessing_and_os_env_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            with patch.object(runtime, '_ENV_PATH', path), patch.dict(os.environ, {}, clear=True):
                path.write_text('GEMINI_API_KEY=test-only-alpha\nGEMINI_API_KEY_2=test-only-beta\nGEMINI_ACTIVE_KEY=1\n')
                self.assertEqual(runtime.resolve_credential().key, 'test-only-alpha')
                path.write_text('GEMINI_API_KEY=test-only-alpha\nGEMINI_API_KEY_2=test-only-beta\nGEMINI_ACTIVE_KEY=2\n')
                self.assertEqual(runtime.resolve_credential().key, 'test-only-beta')
                os.environ['GEMINI_ACTIVE_KEY'] = '1'
                self.assertEqual(runtime.resolve_credential().key, 'test-only-alpha')


class QuotaTests(unittest.TestCase):
    def test_error_categories_and_retryability(self):
        for error, kind, retryable in [
            (Failure(503), 'service_unavailable', True),
            (Failure(429, 'RequestsPerMinute'), 'rate_limit', True),
            (Failure(429, 'TokensPerMinute'), 'rate_limit', True),
            (Failure(429, 'RequestsPerDay', retry_after=2), 'daily_quota', False),
            (Failure(429), 'quota_unknown', False),
            (Failure(401), 'authentication', False), (Failure(403), 'authentication', False),
            (runtime.InvalidResponseError('schema'), 'invalid_response', False)]:
            with self.subTest(kind=kind):
                actual = runtime.classify_error(error)
                self.assertEqual((actual.kind, actual.retryable), (kind, retryable))

    def test_transient_retry_is_bounded_and_does_not_rotate_keys(self):
        client = Mock()
        client.models.generate_content.side_effect = [Failure(503)] * 4
        with patch.object(runtime.time, 'sleep') as sleep, \
             patch.object(runtime.random, 'uniform', return_value=.1), \
             patch.object(runtime, 'resolve_credential') as select:
            with self.assertRaises(Failure):
                runtime.generate_with_retry(client, {})
        self.assertEqual(client.models.generate_content.call_count, 4)
        self.assertEqual(sleep.call_count, 3)
        select.assert_not_called()

    def test_daily_auth_invalid_and_unknown_quota_do_not_retry(self):
        for error in (Failure(429, 'RequestsPerDay'), Failure(401), Failure(403),
                      runtime.InvalidResponseError('bad JSON'), Failure(429)):
            client = Mock()
            client.models.generate_content.side_effect = error
            with patch.object(runtime.time, 'sleep') as sleep:
                with self.assertRaises(type(error)):
                    runtime.generate_with_retry(client, {})
            self.assertEqual(client.models.generate_content.call_count, 1)
            sleep.assert_not_called()


class ProjectResourceTests(unittest.TestCase):
    def test_remote_resource_checked_by_selected_project_before_reuse(self):
        first, second = Mock(), Mock()
        first.files.get.return_value = SimpleNamespace(name='files/example', state='ACTIVE')
        second.files.get.side_effect = Failure(403)
        self.assertEqual(runtime.get_project_file(first, 'files/example').name, 'files/example')
        with self.assertRaisesRegex(RuntimeError, 'selected project'):
            runtime.get_project_file(second, 'files/example')
        first.files.get.assert_called_once_with(name='files/example')
        second.files.upload.assert_not_called()
        second.models.generate_content.assert_not_called()

    def test_upload_needs_explicit_authorization_and_preserves_local_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'local.mp4'
            path.write_bytes(b'unchanged local input')
            client = Mock()
            with self.assertRaisesRegex(RuntimeError, '--allow-upload'):
                runtime.upload_project_file(client, path)
            client.files.upload.assert_not_called()
            runtime.upload_project_file(client, path, authorized=True)
            client.files.upload.assert_called_once_with(file=str(path))
            self.assertEqual(path.read_bytes(), b'unchanged local input')


def intermediate(records, *, transition='same_intermediate_result', scope='intermediate', relation='execution'):
    decisions = fixtures.task_boundaries(records, 'Seat housing', relation=relation)
    for decision in decisions:
        decision.update(objective_before='Seat the housing', objective_after='Complete its seated fit',
            objective_transition=transition, objective_scope=scope,
            transition_evidence='The housing remains in the same socket while its seated fit is completed.')
    return decisions


class IntermediateTaskTests(unittest.TestCase):
    def setUp(self):
        self.records = semantic.segment_records(fixtures.detailed(['Position housing', 'Adjust housing', 'Inspect fit']))

    def test_continuation_retry_correction_and_inspection_with_different_verbs_stay_one_task(self):
        for relation in ('continuation', 'retry', 'correction', 'inspection_pause', 'refinement', 'completion_check'):
            with self.subTest(relation=relation):
                groups = semantic._protect_proposals([fixtures.group(self.records)], self.records,
                                                     intermediate(self.records, relation=relation))
                self.assertEqual(len(groups), 1)

    def test_same_object_and_overall_process_do_not_override_changed_intermediate_result(self):
        decisions = intermediate(self.records)
        decisions[1].update(objective_transition='different_intermediate_result',
            objective_after='Seal the housing', transition_evidence='The seated fit is left and a sealing tool is applied.')
        result = semantic._protect_proposals([fixtures.group(self.records)], self.records, decisions)
        self.assertEqual([group['source_segment_ids'] for group in result], [[1, 2], [3]])
        self.assertTrue(semantic.grouping_diagnostics([fixtures.group(self.records)], decisions))
        process_only = intermediate(self.records, scope='process')
        self.assertEqual(len(semantic._protect_proposals([fixtures.group(self.records)], self.records, process_only)), 3)

    def test_identical_actions_can_have_different_intermediate_objectives(self):
        records = semantic.segment_records(fixtures.detailed(['Turn component', 'Turn component']))
        decisions = intermediate(records, transition='different_intermediate_result')
        self.assertEqual(len(semantic._protect_proposals([fixtures.group(records)], records, decisions)), 2)

    def test_legacy_boundary_contract_remains_readable_and_warning_is_nonblocking(self):
        decisions = fixtures.boundaries(self.records)
        result = semantic._protect_proposals([fixtures.group(self.records)], self.records, decisions)
        before = deepcopy(result)
        with patch.object(genai, 'Client') as client:
            self.assertTrue(semantic.grouping_diagnostics(result, decisions))
        self.assertEqual(result, before)
        self.assertEqual(len(result), 1)
        client.assert_not_called()

    def test_legitimate_full_sequence_task_is_not_split_based_on_count(self):
        records = semantic.segment_records(fixtures.detailed(['Work on component'] * 11))
        decisions = intermediate(records)
        result = semantic._protect_proposals([fixtures.group(records)], records, decisions)
        self.assertEqual(len(result), 1)
        self.assertEqual(semantic.grouping_diagnostics(result, decisions), [])

    def test_new_boundary_evidence_is_complete_and_typed(self):
        decisions = intermediate(self.records)
        del decisions[0]['objective_after']
        with self.assertRaisesRegex(ValueError, 'immediate-objective'):
            semantic._boundary_decisions(decisions, self.records)

    def test_all_ten_boundaries_evaluated_in_one_batched_request_with_model_provenance(self):
        detailed = fixtures.detailed(['Position component', 'Adjust fit', 'Inspect connection'] * 3 + ['Refine fit', 'Complete fit'])
        records = semantic.segment_records(detailed)
        def respond(**kwargs):
            supplied = json.loads(kwargs['contents'].parts[1].text)['segments']
            self.assertEqual(len(supplied), 11)
            return SimpleNamespace(text=json.dumps(fixtures.proposal(supplied,
                groups=[fixtures.group(supplied)], decisions=intermediate(supplied))))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runtime, 'configuration', return_value=dict(GEMINI_API_KEY='test-only-alpha',
                GEMINI_MODEL='legacy-model', GEMINI_AGGREGATION_MODEL='gemini-4-hydrogen')), \
             patch.object(genai, 'Client') as factory, patch.object(semantic, '_media_parts', return_value=[]):
            video = Path(directory) / 'existing.avi'
            video.write_bytes(b'existing identity; never decoded')
            client = factory.return_value.__enter__.return_value
            client.models.generate_content.side_effect = respond
            stats = {}
            semantic.propose_task_groups(video, detailed, cache_dir=Path(directory) / 'cache', stats=stats)
            self.assertEqual(stats['fresh'], 10)
            self.assertEqual(stats['api_calls'], 1)
            self.assertEqual(len(stats['boundary_sources']), 10)
            self.assertEqual(stats['provenance']['model'], 'gemini-4-hydrogen')
            self.assertEqual(client.models.generate_content.call_args.kwargs['model'], 'gemini-4-hydrogen')
            audit = next((Path(directory) / 'cache' / 'runs').glob('*.json'))
            saved = json.loads(audit.read_text())
            self.assertEqual(len(saved['boundaries']), 10)
            self.assertEqual(saved['provenance']['generation']['thinking_config']['thinking_level'], 'LOW')
            self.assertNotIn('test-only-alpha', audit.read_text())
            self.assertEqual(saved['records'], records)

    def test_confirmed_baseline_is_archived_without_schema_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'confirmed.json'
            groups = semantic.singleton_groups(self.records)
            semantic.save_confirmed_review(path, 'identity', groups)
            baseline = path.read_bytes()
            semantic.save_confirmed_review(path, 'identity', groups, provenance={'model': 'another-model'})
            archived = list((Path(directory) / 'semantic_history').glob('*/confirmed.json'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), baseline)
            self.assertEqual(semantic.load_confirmed_review(path, 'identity', self.records), groups)


if __name__ == '__main__':
    unittest.main()
