"""Offline Pegasus SDK/asset/label/aggregation contracts. Never calls a provider."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Event, Thread, current_thread
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ai_providers as providers
import batch_label_suggester as batch
import gemini_runtime as runtime
import label_suggester as labels
import twelvelabs_provider as tl
from gui import gui_elements as gui, semantic_review
from gui.ai_upload import UploadConsent
from toolbox import semantic_aggregation as semantic
from tests.offline_support import TWELVELABS_CLIENT_FACTORY
from tests.test_batch_label_suggestions import boundaries, payload
from tests.test_quota_recovery import Desktop
from tests.test_semantic_aggregation import detailed, proposal, task_boundaries


METADATA = dict(fps=15., frame_count=300, duration=20., width=640, height=480, start_time=0.)
REAL_CONFIGURATION = tl.configuration
REAL_VIDEO_METADATA = tl.video_metadata


def sdk_response(value):
    return SimpleNamespace(data=json.dumps(value), finish_reason='stop', error=None)


def activity():
    return dict(label='Move object', evidence='The hand moves the object inside this segment.',
                uncertainty='low', target_visible=True, bow_action='not_applicable')


def complete_proposal(records):
    decisions = task_boundaries(records, 'Attach decoration')
    for decision in decisions:
        decision.update(objective_before='Secure decoration', objective_after='Secure decoration',
            objective_transition='same_intermediate_result', objective_scope='intermediate',
            transition_evidence='The same attachment is still being adjusted until secure.')
    return proposal(records, decisions=decisions)


class Failure(Exception):
    def __init__(self, status, code=None):
        self.status_code = status
        self.body = {'code': code} if code else {}
        self.response = SimpleNamespace(headers={})
        super().__init__(f'HTTP {status}')


class PegasusFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="Pegasus ' & ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / 'test video (converter).avi'
        self.video.write_bytes(b'synthetic video bytes; never decoded')
        self.config = dict(TWELVELABS_API_KEY='offline-tl-project-secret', TWELVELABS_MODEL='pegasus1.6',
                           RAVEE_LABEL_PROVIDER='twelvelabs', RAVEE_AGGREGATION_PROVIDER='twelvelabs')
        self.client = Mock()
        self.assets = {}
        self.client.assets.create.side_effect = self.create_asset
        self.client.assets.retrieve.side_effect = lambda identifier, **_: self.assets[identifier]
        self.client.analyze.return_value = sdk_response(activity())
        self.factory = Mock(return_value=self.client)
        sdk = ModuleType('twelvelabs')
        sdk.TwelveLabs = Mock(return_value=self.client)
        types = ModuleType('twelvelabs.types')
        types.VideoContext_AssetId = lambda **kwargs: SimpleNamespace(**kwargs)
        types.SyncResponseFormat = lambda **kwargs: SimpleNamespace(**kwargs)
        self.sdk = sdk
        for target, value in ((runtime, ('configuration', Mock(return_value=self.config))),
                               (tl, ('configuration', Mock(return_value=self.config))),
                               (tl, ('create_client', self.factory)),
                               (tl, ('video_metadata', Mock(return_value=dict(METADATA)))),
                               (tl, ('_CACHE_DIR', self.root / 'tl-cache')),
                               (tl.time, ('sleep', Mock()))):
            owner, (name, replacement) = target, value
            patcher = patch.object(owner, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.dict(sys.modules, {'twelvelabs': sdk, 'twelvelabs.types': types})
        patcher.start()
        self.addCleanup(patcher.stop)

    def create_asset(self, **kwargs):
        self.assertEqual(kwargs['file'].read(), self.video.read_bytes())
        identifier = 'asset-' + str(len(self.assets) + 1)
        value = SimpleNamespace(id=identifier, status='ready', user_metadata=json.loads(kwargs['user_metadata']))
        self.assets[identifier] = value
        return value

    def adapter(self, **kwargs):
        return providers.create_adapter('label', **kwargs)

    def batch(self, records=None, **kwargs):
        records = records or boundaries()
        self.client.analyze.return_value = sdk_response(payload(records))
        return batch.suggest_all_activities(self.video, records, **kwargs)


class ConfigurationTests(PegasusFixture):
    def test_gemini_is_default_and_stages_are_independent(self):
        self.config.clear()
        self.assertEqual(providers.get_provider('label'), 'gemini')
        self.assertEqual(providers.get_provider('aggregation'), 'gemini')
        self.config.update(GEMINI_MODEL='unchanged', GEMINI_LABEL_MODEL='label-specific',
                           RAVEE_AGGREGATION_PROVIDER='twelvelabs')
        self.assertEqual(providers.get_model('label'), 'label-specific')
        self.assertEqual(providers.get_model('aggregation'), 'pegasus1.6')
        self.assertEqual(providers.provider_status('aggregation'), 'twelvelabs / pegasus1.6')
        self.factory.assert_not_called()

    def test_invalid_selection_and_missing_key_do_not_fallback(self):
        self.config['RAVEE_LABEL_PROVIDER'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'must be gemini or twelvelabs'):
            labels.suggest_activity(self.video, 0, 14)
        self.assertIn('configuration error', providers.provider_status('label'))
        self.config['RAVEE_LABEL_PROVIDER'] = 'twelvelabs'
        self.config['TWELVELABS_API_KEY'] = ''
        with self.assertRaisesRegex(RuntimeError, 'Missing TWELVELABS_API_KEY'):
            self.adapter()
        self.factory.assert_not_called()

    def test_sdk_initialization_is_lazy_uses_explicit_key_and_timeout(self):
        adapter = self.adapter()
        self.factory.assert_not_called()
        client = TWELVELABS_CLIENT_FACTORY(adapter.credential)
        self.assertIs(client, self.client)
        self.sdk.TwelveLabs.assert_called_once_with(api_key='offline-tl-project-secret', timeout=120)
        self.assertNotIn('offline-tl-project-secret', repr(adapter.credential))

    def test_missing_optional_sdk_is_actionable(self):
        with patch.dict(sys.modules, {'twelvelabs': None}), self.assertRaisesRegex(RuntimeError, 'Optional TwelveLabs SDK missing'):
            TWELVELABS_CLIENT_FACTORY(self.adapter().credential)

    def test_dotenv_provider_options_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / '.env'
            path.write_text('RAVEE_LABEL_PROVIDER=twelvelabs\nTWELVELABS_MODEL=pegasus1.6\n', encoding='utf-8')
            with patch.object(runtime, '_ENV_PATH', path), patch.object(runtime, '_INJECTED', {}), \
                 patch.object(runtime, 'configuration', REAL_CONFIGURATION):
                self.assertEqual(providers.get_provider('label'), 'twelvelabs')
                self.assertEqual(providers.get_model('label'), 'pegasus1.6')
                os.environ['RAVEE_LABEL_PROVIDER'] = 'gemini'
                os.environ['GEMINI_MODEL'] = 'explicit-env-model'
                self.assertEqual(providers.get_provider('label'), 'gemini')
                self.assertEqual(providers.get_model('label'), 'explicit-env-model')


class AssetTests(PegasusFixture):
    def test_creation_and_authenticated_reuse_across_stages(self):
        identity = tl.video_identity(self.video)
        first = self.adapter().asset(self.video, identity)
        second = providers.create_adapter('aggregation').asset(self.video, identity)
        self.assertEqual(first, second)
        self.client.assets.create.assert_called_once()
        self.client.assets.retrieve.assert_called_once_with(first, request_options=tl._REQUEST_OPTIONS)
        request = self.client.assets.create.call_args.kwargs
        self.assertEqual(request['method'], 'direct')
        self.assertEqual(request['filename'], self.video.name)
        self.assertEqual(request['request_options']['max_retries'], 0)

    def test_processing_asset_is_polled_and_reference_survives_timeout(self):
        original = self.create_asset
        def processing(**kwargs):
            asset = original(**kwargs)
            asset.status = 'processing'
            return asset
        self.client.assets.create.side_effect = processing
        with patch.object(tl.time, 'monotonic', side_effect=[0, 601]), self.assertRaisesRegex(tl.TwelveLabsError, 'readiness timed out'):
            self.adapter().asset(self.video, tl.video_identity(self.video))
        self.assertEqual(len(list((tl._CACHE_DIR / 'assets').glob('*.json'))), 1)
        self.assets['asset-1'].status = 'ready'
        self.assertEqual(self.adapter().asset(self.video, tl.video_identity(self.video)), 'asset-1')
        self.client.assets.create.assert_called_once()

    def test_processing_asset_transitions_to_ready(self):
        def create(**kwargs):
            asset = self.create_asset(**kwargs)
            return SimpleNamespace(id=asset.id, status='processing', user_metadata=asset.user_metadata)
        self.client.assets.create.side_effect = create
        self.assertEqual(self.adapter().asset(self.video, tl.video_identity(self.video)), 'asset-1')
        self.client.assets.retrieve.assert_called_once()
        tl.time.sleep.assert_called_once()

    def test_failed_asset_is_not_analyzed_or_reuploaded(self):
        adapter = self.adapter()
        identity = tl.video_identity(self.video)
        identifier = adapter.asset(self.video, identity)
        self.assets[identifier].status = 'failed'
        with self.assertRaisesRegex(tl.TwelveLabsError, 'could not process'):
            self.adapter().asset(self.video, identity)
        self.client.assets.create.assert_called_once()
        self.client.analyze.assert_not_called()

    def test_account_change_does_not_reuse_old_remote_identifiers(self):
        identity = tl.video_identity(self.video)
        first = self.adapter().asset(self.video, identity)
        self.config['TWELVELABS_API_KEY'] = 'different-authorized-account'
        second = self.adapter().asset(self.video, identity)
        self.assertNotEqual(first, second)
        self.assertEqual(self.client.assets.create.call_count, 2)
        self.client.assets.retrieve.assert_not_called()

    def test_unauthorized_asset_never_triggers_another_upload(self):
        identity = tl.video_identity(self.video)
        self.adapter().asset(self.video, identity)
        self.client.assets.retrieve.side_effect = Failure(403)
        with self.assertRaises(tl.TwelveLabsError) as raised:
            self.adapter().asset(self.video, identity)
        self.assertEqual(raised.exception.kind, 'authentication')
        self.client.assets.create.assert_called_once()

    def test_mismatched_remote_content_is_not_adopted(self):
        identity = tl.video_identity(self.video)
        identifier = self.adapter().asset(self.video, identity)
        self.assets[identifier].user_metadata['ravee_sha256'] = 'another-video'
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.adapter().asset(self.video, identity)
        self.client.assets.create.assert_called_once()

    def test_missing_asset_is_uploaded_again_only_after_authenticated_404(self):
        identity = tl.video_identity(self.video)
        self.adapter().asset(self.video, identity)
        self.client.assets.retrieve.side_effect = Failure(404)
        self.assertEqual(self.adapter().asset(self.video, identity), 'asset-2')
        self.assertEqual(self.client.assets.create.call_count, 2)

    def test_upload_limits_and_explicit_consent(self):
        identity = tl.video_identity(self.video)
        with patch.object(tl, '_DIRECT_UPLOAD_BYTES', 4), self.assertRaisesRegex(ValueError, '200 MB'):
            self.adapter().asset(self.video, identity)
        with patch.object(tl, '_AUTO_UPLOAD_BYTES', 4):
            for approval in (None, Mock(return_value=False)):
                with self.assertRaisesRegex(RuntimeError, 'explicit approval'):
                    self.adapter(on_upload=approval).asset(self.video, identity)
            self.client.assets.create.assert_not_called()
            approve = Mock(return_value=True)
            self.adapter(on_upload=approve).asset(self.video, identity)
            approve.assert_called_once_with(str(self.video), identity['size'])
        self.client.assets.create.assert_called_once()

    def test_unknown_upload_outcome_is_not_automatically_retried(self):
        self.client.assets.create.side_effect = TimeoutError('ambiguous upload')
        with self.assertRaises(tl.TwelveLabsError):
            self.adapter().asset(self.video, tl.video_identity(self.video))
        self.client.assets.create.assert_called_once()
        tl.time.sleep.assert_not_called()


class LabelTests(PegasusFixture):
    def test_twenty_segments_one_analyze_complete_video_and_exact_boundaries(self):
        records = boundaries()
        snapshot = deepcopy(records)
        stats = {}
        result = self.batch(records, stats=stats)
        self.assertEqual(len(result), 20)
        self.assertEqual(stats['api_calls'], 1)
        self.assertEqual(records, snapshot)
        request = self.client.analyze.call_args.kwargs
        self.assertEqual(request['model_name'], 'pegasus1.6')
        self.assertEqual(request['video'].asset_id, 'asset-1')
        self.assertNotIn('start_time', request)
        manifest = json.loads(request['prompt'].split('\n')[-1])
        self.assertEqual(manifest['segments'][0]['start_frame'], 0)
        self.assertEqual(manifest['segments'][-1]['end_frame'], 299)
        self.assertEqual(manifest['segments'][-1]['end_seconds_exclusive'], 20.)
        remote = request['response_format'].json_schema['properties']['segments']
        self.assertEqual(remote['minItems'], 1)
        self.assertNotIn('maxItems', remote)
        self.client.analyze.assert_called_once()

    def test_individual_window_uses_full_asset_and_preserves_exact_target(self):
        self.assertEqual(labels.suggest_activity(self.video, 100, 114), 'Move object')
        request = self.client.analyze.call_args.kwargs
        manifest = json.loads(request['prompt'].split('\n')[-1])
        self.assertEqual(manifest['target_segment']['start_frame'], 100)
        self.assertEqual(manifest['target_segment']['end_frame'], 114)
        self.assertAlmostEqual(request['start_time'], 100 / 15 - 3)
        self.assertAlmostEqual(request['end_time'], 115 / 15 + 3)
        self.assertEqual(request['response_format'].json_schema, labels._ACTIVITY_SCHEMA)

    def test_subsecond_target_window_is_padded_without_moving_target(self):
        with patch.dict(os.environ, {'RAVEE_SUGGEST_CONTEXT_SECONDS': '0'}):
            labels.suggest_activity(self.video, 299, 299)
        request = self.client.analyze.call_args.kwargs
        manifest = json.loads(request['prompt'].split('\n')[-1])
        self.assertEqual(manifest['target_segment']['start_frame'], 299)
        self.assertGreaterEqual(request['end_time'] - request['start_time'], 1.)
        self.assertLessEqual(request['end_time'], 20.)

    def test_nonzero_container_start_offsets_api_window_but_not_finch_frames(self):
        with patch.object(tl, 'video_metadata', return_value=dict(METADATA, start_time=12.5)):
            labels.suggest_activity(self.video, 100, 114)
        request = self.client.analyze.call_args.kwargs
        self.assertAlmostEqual(request['start_time'], 100 / 15 - 3 + 12.5)
        manifest = json.loads(request['prompt'].split('\n')[-1])
        self.assertEqual(manifest['target_segment']['start_frame'], 100)
        self.assertEqual(manifest['target_segment']['start_seconds'], 100 / 15)

    def test_unverified_time_origin_uses_full_asset_not_a_guessed_api_window(self):
        with patch.object(tl, 'video_metadata', return_value=dict(METADATA, start_time=None)):
            labels.suggest_activity(self.video, 100, 114)
        request = self.client.analyze.call_args.kwargs
        self.assertNotIn('start_time', request)
        self.assertNotIn('end_time', request)
        manifest = json.loads(request['prompt'].split('\n')[-1])
        self.assertEqual(manifest['target_segment']['end_frame'], 114)

    def test_metadata_checks_are_header_only_and_reject_provider_input_limits(self):
        cap = Mock()
        cap.isOpened.return_value = True
        properties = {tl.cv2.CAP_PROP_FPS: 15., tl.cv2.CAP_PROP_FRAME_COUNT: 300,
                      tl.cv2.CAP_PROP_FRAME_WIDTH: 640, tl.cv2.CAP_PROP_FRAME_HEIGHT: 480}
        cap.get.side_effect = lambda prop: properties[prop]
        with patch.object(tl.cv2, 'VideoCapture', return_value=cap), patch.object(tl, '_container_start_time', return_value=0):
            self.assertEqual(REAL_VIDEO_METADATA(self.video)['fps'], 15.)
            cap.read.assert_not_called()
            properties[tl.cv2.CAP_PROP_FRAME_COUNT] = 15 * 3601
            with self.assertRaisesRegex(ValueError, '1-hour'):
                REAL_VIDEO_METADATA(self.video)
            properties[tl.cv2.CAP_PROP_FRAME_COUNT] = 300
            properties[tl.cv2.CAP_PROP_FRAME_HEIGHT] = 240
            with self.assertRaisesRegex(ValueError, 'resolution/aspect'):
                REAL_VIDEO_METADATA(self.video)
        self.factory.assert_not_called()

    def test_ffprobe_reads_only_container_origin_and_failure_is_not_assumed_zero(self):
        with patch.object(tl.shutil, 'which', return_value='ffprobe'), \
             patch.object(tl.subprocess, 'run', return_value=SimpleNamespace(stdout='{"format":{"start_time":"5.25"}}')) as run:
            self.assertEqual(tl._container_start_time(self.video), 5.25)
            self.assertIn('format=start_time', run.call_args.args[0])
            self.assertNotIn('-show_frames', run.call_args.args[0])
        with patch.object(tl.shutil, 'which', return_value=None):
            self.assertIsNone(tl._container_start_time(self.video))

    def test_incomplete_duplicate_unknown_ids_and_invalid_types_never_cache(self):
        records = boundaries()
        for mutate in (lambda p: p['segments'].pop(),
                       lambda p: p['segments'][0].update(segment_id=True),
                       lambda p: p['segments'][0].update(segment_id=999),
                       lambda p: p['segments'][1].update(segment_id=1),
                       lambda p: p['segments'][0].update(target_visible='true')):
            candidate = payload(records)
            mutate(candidate)
            self.client.analyze.return_value = sdk_response(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(runtime.InvalidResponseError):
                batch.suggest_all_activities(self.video, records)
        self.assertFalse((tl._CACHE_DIR / 'batch').exists())

    def test_truncation_duplicate_json_and_nonfinite_json_are_rejected(self):
        for response in (SimpleNamespace(data=json.dumps(activity()), finish_reason='length'),
                         SimpleNamespace(data='{"label":"one","label":"two"}', finish_reason='stop'),
                         SimpleNamespace(data='{"label":NaN}', finish_reason='stop')):
            self.client.analyze.return_value = response
            with self.assertRaises(runtime.InvalidResponseError):
                labels.suggest_activity(self.video, 0, 14)
        self.assertFalse((tl._CACHE_DIR / 'individual').exists())

    def test_result_caches_are_separate_and_warm_hits_make_no_client(self):
        self.batch()
        self.client.analyze.assert_called_once()
        self.client.assets.create.assert_called_once()
        before = {path: path.read_bytes() for path in tl._CACHE_DIR.rglob('*.json')}
        self.factory.reset_mock()
        self.assertEqual(len(self.batch()), 20)
        self.factory.assert_not_called()
        self.client.analyze.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.client.analyze.return_value = sdk_response(activity())
        labels.suggest_activity(self.video, 0, 14)
        self.assertTrue((tl._CACHE_DIR / 'batch').is_dir())
        self.assertTrue((tl._CACHE_DIR / 'individual').is_dir())
        self.client.assets.create.assert_not_called()

    def test_content_boundaries_model_prompt_generation_change_result_identity(self):
        self.batch()
        altered = boundaries()
        altered[-1]['end_frame'] = 298
        self.batch(altered)
        self.config['TWELVELABS_MODEL'] = 'pegasus1.5'
        self.batch()
        with patch.object(batch, '_PROMPT', batch._PROMPT + ' Another instruction.'):
            self.batch()
        with patch.object(tl, 'generation_settings', return_value=dict(temperature=.2, max_tokens=9000)):
            self.batch()
        self.video.write_bytes(b'different synthetic video bytes')
        self.batch()
        self.assertEqual(self.client.analyze.call_count, 6)
        self.assertEqual(len(list((tl._CACHE_DIR / 'batch').glob('*.json'))), 6)

    def test_gemini_and_twelvelabs_custom_cache_directories_are_isolated(self):
        custom = self.root / 'shared-cache'
        custom.mkdir()
        sentinel = custom / 'existing-gemini.json'
        sentinel.write_text('preserved Gemini artifact', encoding='utf-8')
        self.batch(cache_dir=custom)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserved Gemini artifact')
        self.assertEqual(len(list((custom / 'twelvelabs' / 'batch').glob('*.json'))), 1)

    def test_corrupt_cache_is_validated_and_recomputed(self):
        self.batch()
        cache = next((tl._CACHE_DIR / 'batch').glob('*.json'))
        cache.write_text('{broken', encoding='utf-8')
        self.assertEqual(len(self.batch()), 20)
        self.assertEqual(self.client.analyze.call_count, 2)
        self.client.assets.create.assert_called_once()


class AggregationTests(PegasusFixture):
    def setUp(self):
        super().setUp()
        self.df = detailed(['Position decoration', 'Adjust decoration', 'Secure decoration'])
        def analyze(**kwargs):
            manifest = json.loads(kwargs['prompt'].split('\n')[-1])
            return sdk_response(complete_proposal(manifest['segments']))
        self.client.analyze.side_effect = analyze

    def propose(self, **kwargs):
        return semantic.propose_task_groups(self.video, self.df,
            cache_dir=self.root / 'boundary-cache', **kwargs)

    def test_aggregation_uses_reviewed_labels_timestamps_and_existing_boundary_contract(self):
        snapshot = self.df.copy(deep=True)
        stats = {}
        groups = self.propose(stats=stats)
        self.assertEqual(groups[0]['source_segment_ids'], [1, 2, 3])
        self.assertEqual(groups[0]['start_frame'], 0)
        self.assertEqual(groups[0]['end_frame'], 44)
        self.assertTrue(snapshot.equals(self.df))
        self.assertEqual(stats['provenance']['provider'], 'twelvelabs')
        self.assertEqual(stats['api_calls'], 1)
        manifest = json.loads(self.client.analyze.call_args.kwargs['prompt'].split('\n')[-1])
        self.assertEqual(manifest['segments'][1]['activity'], 'Adjust decoration')
        self.assertIn('timestamp_start', manifest['segments'][1])
        self.assertEqual(manifest['segments'][1]['start_seconds'], 1.)
        self.assertEqual(manifest['context_segments'], [])

    def test_warm_boundary_cache_makes_no_client_and_is_separate_from_gemini(self):
        self.propose()
        self.factory.reset_mock()
        stats = {}
        self.propose(stats=stats)
        self.factory.assert_not_called()
        self.assertEqual(stats['cache_hits'], 2)
        records = semantic.segment_records(self.df)
        identity = {'sha256': 'same-input'}
        tl_key = semantic._boundary_cache_key(identity, records, 0, 12)
        self.config['RAVEE_AGGREGATION_PROVIDER'] = 'gemini'
        self.assertNotEqual(tl_key, semantic._boundary_cache_key(identity, records, 0, 12))

    def test_invalid_boundary_contract_reports_limitation_and_keeps_singletons(self):
        for mutate in (lambda p: p['boundaries'].pop(),
                       lambda p: p['boundaries'][0].update(left_segment_id=99),
                       lambda p: p['groups'][0].update(end_frame=999),
                       lambda p: p['boundaries'][0].pop('objective_scope')):
            candidate = complete_proposal(semantic.segment_records(self.df))
            mutate(candidate)
            self.client.analyze.side_effect = None
            self.client.analyze.return_value = sdk_response(candidate)
            stats = {}
            groups = self.propose(stats=stats)
            self.assertEqual(len(groups), 3)
            self.assertEqual(stats['error_kind'], 'invalid_response')
            self.assertEqual(stats['unresolved'], 2)
        self.assertEqual(list((self.root / 'boundary-cache' / 'twelvelabs').glob('*.json')), [])

    def test_process_level_rationale_does_not_override_intermediate_objectives(self):
        candidate = complete_proposal(semantic.segment_records(self.df))
        for decision in candidate['boundaries']:
            decision.update(objective_scope='process', objective_transition='different_intermediate_result')
        self.client.analyze.side_effect = None
        self.client.analyze.return_value = sdk_response(candidate)
        stats = {}
        self.assertEqual(len(self.propose(stats=stats)), 3)
        self.assertTrue(stats['warnings'])

    def test_cached_pegasus_boundaries_still_require_the_full_new_schema(self):
        self.propose()
        path = next((self.root / 'boundary-cache' / 'twelvelabs').glob('*.json'))
        entry = json.loads(path.read_text(encoding='utf-8'))
        entry['decision'].pop('objective_scope')
        entry['sha256'] = tl._digest(entry['decision'])
        path.write_text(json.dumps(entry), encoding='utf-8')
        stats = {}
        self.propose(stats=stats)
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['api_calls'], 1)

    def test_label_provider_can_remain_gemini_while_aggregation_uses_pegasus(self):
        self.config['RAVEE_LABEL_PROVIDER'] = 'gemini'
        self.config['GEMINI_LABEL_MODEL'] = 'existing-v3-model'
        stats = {}
        self.assertEqual(len(self.propose(stats=stats)), 1)
        self.assertEqual(stats['provenance']['label_provider'], 'gemini')
        self.assertEqual(stats['provenance']['label_model'], 'existing-v3-model')


class ErrorTests(PegasusFixture):
    def test_cleanup_failure_does_not_discard_completed_suggestions(self):
        self.client.close.side_effect = RuntimeError('synthetic cleanup failure')
        self.assertEqual(labels.suggest_activity(self.video, 0, 14), 'Move object')
        self.client.close.assert_called_once()

    def test_error_categories_and_only_known_transient_failures_retry(self):
        for error, kind, retry in ((Failure(401), 'authentication', False),
                                   (Failure(403), 'authentication', False),
                                   (Failure(429), 'quota_unknown', False),
                                   (Failure(429, 'usage_limit_exceeded'), 'provider_quota', False),
                                   (Failure(402, 'workload_allowance_exhausted'), 'provider_quota', False),
                                   (Failure(429, 'rate_limit_exceeded'), 'rate_limit', True),
                                   (Failure(503), 'service_unavailable', True),
                                   (TimeoutError(), 'timeout', False)):
            with self.subTest(kind=kind):
                info = tl.error_info(error)
                self.assertEqual((info.kind, info.retryable), (kind, retry))
                wrapped = tl.TwelveLabsError(info)
                self.assertEqual(runtime.classify_error(wrapped), info)

    def test_transient_analysis_retries_are_bounded_and_counted(self):
        self.client.analyze.side_effect = [Failure(503), Failure(429, 'rate_limit_exceeded'), sdk_response(activity())]
        self.assertEqual(labels.suggest_activity(self.video, 0, 14), 'Move object')
        self.assertEqual(self.client.analyze.call_count, 3)
        self.assertEqual([call.args[0] for call in tl.time.sleep.call_args_list], [1, 2])
        self.client.assets.create.assert_called_once()

    def test_unknown_429_does_not_show_daily_quota_or_retry(self):
        self.client.analyze.side_effect = Failure(429)
        callback = Mock()
        with self.assertRaisesRegex(tl.TwelveLabsError, 'quota type unknown'):
            labels.suggest_activity(self.video, 0, 14, on_daily_quota=callback)
        callback.assert_not_called()
        self.client.analyze.assert_called_once()

    def test_confirmed_plan_quota_uses_manual_callback_without_switching(self):
        self.client.analyze.side_effect = Failure(429, 'usage_limit_exceeded')
        callback = Mock(return_value=runtime.Credential('GEMINI_API_KEY', 'never-used'))
        with self.assertRaises(tl.TwelveLabsError):
            labels.suggest_activity(self.video, 0, 14, on_daily_quota=callback)
        callback.assert_called_once()
        self.assertEqual(callback.call_args.args[0].name, 'TWELVELABS_API_KEY')
        self.client.analyze.assert_called_once()
        self.factory.assert_called_once()

    def test_error_keys_are_redacted_even_after_configuration_changes(self):
        adapter = self.adapter()
        self.config['TWELVELABS_API_KEY'] = 'changed-key-secret'
        self.client.analyze.side_effect = RuntimeError('offline-tl-project-secret changed-key-secret')
        with self.assertRaises(tl.TwelveLabsError) as raised:
            adapter.analyze(self.video, 'label', labels._ACTIVITY_SCHEMA, tl.generation_settings())
        self.assertNotIn('offline-tl-project-secret', str(raised.exception))
        self.assertNotIn('changed-key-secret', str(raised.exception))


class GuiTests(PegasusFixture):
    def test_individual_provider_switch_keeps_separate_proposals_and_preserves_drafts(self):
        desktop = Desktop()
        records = boundaries(2)
        original = labels.suggest_activity
        gemini_calls = []
        def suggest(*args, **kwargs):
            if providers.get_provider('label') == 'gemini':
                gemini_calls.append(args)
                return 'Inspect housing'
            return original(*args, **kwargs)
        def exercise():
            entry = desktop.find('Entry')
            entry.value = 'Human pending edit'
            desktop.click('Suggest label')
            desktop.until(lambda: any(w.options.get('text') == 'Use suggestion' for w in desktop.widgets))
            self.assertEqual(entry.value, 'Human pending edit')
            self.assertEqual(gui.LABEL, {})
            self.config.update(RAVEE_LABEL_PROVIDER='gemini', GEMINI_MODEL='existing-gemini-model')
            desktop.click('Use suggestion')  # Requests this provider; no old proposal inserted.
            desktop.until(lambda: any(w.options.get('text') == 'Use suggestion' for w in desktop.widgets))
            self.assertEqual(entry.value, 'Human pending edit')
            self.assertEqual(len(gemini_calls), 1)
            self.config['RAVEE_LABEL_PROVIDER'] = 'twelvelabs'
            desktop.click('Use suggestion')  # Restores the retained Pegasus proposal explicitly.
            self.assertEqual(entry.value, 'Move object')
            self.assertEqual(gui.LABEL, {})
            entry.value = 'Human reviewed action'
            desktop.click('Commit & Next')
            entry.value = 'Second human action'
            desktop.click('Commit & Finish')
        desktop.root.mainloop.side_effect = exercise
        with desktop.patch(gui), patch.object(gui, 'suggest_activity', side_effect=suggest):
            gui.label_video_clusters(self.video, records)
        self.client.analyze.assert_called_once()
        self.assertEqual(gui.LABEL[1], 'Human reviewed action')

    def test_batch_preserves_committed_labels_and_draft_until_explicit_use_commit(self):
        desktop, started, release = Desktop(), Event(), Event()
        records = boundaries()
        def analyze(**kwargs):
            self.assertNotEqual(current_thread().name, 'MainThread')
            started.set()
            release.wait(2)
            return sdk_response(payload(records))
        self.client.analyze.side_effect = analyze
        def exercise():
            entry, tree = desktop.find('Entry'), desktop.find('Treeview')
            entry.value = 'Reviewed first action'
            desktop.click('Commit & Next')
            entry.value = 'Uncommitted human edit'
            desktop.click('Suggest All')
            desktop.until(started.is_set)
            self.assertEqual(desktop.find('Button', 'Suggest All').options['state'], 'disabled')
            self.assertEqual(entry.value, 'Uncommitted human edit')
            release.set()
            desktop.until(lambda: desktop.find('Button', 'Use batch suggestion').options['state'] == 'normal')
            self.assertEqual(entry.value, 'Uncommitted human edit')
            self.assertEqual(gui.LABEL[1], 'Reviewed first action')
            self.assertEqual(tree.selection(), ('1',))
            self.assertTrue(any(w.options.get('text') == 'twelvelabs / pegasus1.6' for w in desktop.widgets))
            desktop.click('Use batch suggestion')
            self.assertEqual(entry.value, 'Move object 2')
            self.assertNotIn(2, gui.LABEL)
            for number in range(2, 21):
                entry.value = f'Human action {number}'
                desktop.click('Commit & Finish' if number == 20 else 'Commit & Next')
        desktop.root.mainloop.side_effect = exercise
        try:
            with desktop.patch(gui):
                gui.label_video_clusters(self.video, records)
        finally:
            release.set()
        self.client.analyze.assert_called_once()

    def test_twelvelabs_quota_dialog_preserves_aggregation_edits_and_history(self):
        desktop = Desktop()
        df = detailed(['Position part', 'Adjust part', 'Inspect part'])
        self.client.analyze.side_effect = Failure(429, 'workload_allowance_exhausted')
        original = semantic.propose_task_groups
        def propose(*args, **kwargs):
            return original(*args, cache_dir=self.root / 'boundaries', **kwargs)
        metadata = {'review': 'preserved'}
        def exercise():
            entry, tree, reason = desktop.find('Entry'), desktop.find('Treeview'), desktop.find('Text')
            entry.value = 'Reviewed intermediate task'
            desktop.click('Apply edits')
            tree.selection_set('1')
            tree.bindings['<<TreeviewSelect>>'](None)
            entry.value, reason.value = 'Pending task edit', 'Pending evidence'
            desktop.click('Suggest task groups')
            desktop.until(lambda: any(w.kind == 'Toplevel' for w in desktop.widgets))
            self.assertEqual(desktop.find('Button', 'Select credential').options['state'], 'disabled')
            self.assertEqual(entry.value, 'Pending task edit')
            desktop.click('Cancel')
            status = next(w for w in desktop.widgets if w.kind == 'StringVar' and
                          w.options.get('value', '').startswith('Review groups manually'))
            desktop.until(lambda: status.value.startswith('TwelveLabs quota request stopped'))
            self.assertEqual(tree.selection(), ('1',))
            self.assertEqual(entry.value, 'Pending task edit')
            self.assertEqual(reason.value, 'Pending evidence')
            self.assertEqual(metadata, {'review': 'preserved'})
            desktop.click('Confirm task groups')
        desktop.root.mainloop.side_effect = exercise
        with desktop.patch(semantic_review), patch.object(semantic_review, 'propose_task_groups', side_effect=propose):
            result = semantic_review.review_task_groups(self.video, df, run_metadata=metadata)
        self.assertEqual(result[0]['suggested_task_label'], 'Reviewed intermediate task')
        self.assertEqual(result[1]['suggested_task_label'], 'Pending task edit')
        self.client.analyze.assert_called_once()

    def test_large_upload_consent_is_on_ui_thread_and_close_unblocks_worker(self):
        desktop = Desktop()
        consent = UploadConsent(desktop.root)
        result = []
        worker = Thread(target=lambda: result.append(consent.authorize(str(self.video), 100 * 1024 * 1024)), daemon=True)
        with patch('gui.ai_upload.messagebox.askyesno', return_value=False) as ask:
            worker.start()
            desktop.until(lambda: consent.pending is not None)
            desktop.pump()
            worker.join(2)
            self.assertEqual(result, [False])
            ask.assert_called_once()
        blocked = Thread(target=lambda: result.append(consent.authorize(str(self.video), 100 * 1024 * 1024)), daemon=True)
        blocked.start()
        desktop.until(lambda: consent.pending is not None)
        consent.close()
        blocked.join(2)
        self.assertFalse(blocked.is_alive())
        self.assertEqual(result, [False, False])


if __name__ == '__main__':
    unittest.main()
