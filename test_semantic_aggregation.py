"""Offline semantic-task regression suite: no real Gemini, GUI or pipeline run."""

from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'toolbox'))

from toolbox import semantic_aggregation as s
from toolbox import event_log_creation as events
import label_suggester as v3
from google import genai

MAKE_MEDIA = s._media_parts


def detailed(labels):
    base = pd.Timestamp('2026-11-15T10:00:00')
    return pd.DataFrame([dict(videoID=1, segment_id=i + 1, cluster_id=i % 2,
        activity=label, frame_start=i * 15, frame_end=(i + 1) * 15 - 1,
        timestamp_start=base + pd.Timedelta(seconds=i),
        timestamp_end=base + pd.Timedelta(seconds=i + 1), duration_seconds=1.)
        for i, label in enumerate(labels)])


RETRIES = ['Pick up decoration', 'Insert decoration', 'Remove decoration',
           'Adjust decoration position', 'Insert decoration']


def group(records, label='Attach decoration', *, uncertainty='low', same=True):
    return dict(start_frame=records[0]['start_frame'], end_frame=records[-1]['end_frame'],
                source_segment_ids=[row['segment_id'] for row in records], suggested_task_label=label,
                explanation='The same decoration is visibly repositioned and reinserted toward one attachment objective',
                same_objective=same, uncertainty=uncertainty)


def boundaries(records, label='Attach decoration', *, object_continuity='same',
               objective_continuity='same', relation='retry', uncertainty='low',
               evidence='The same decoration remains held and is reinserted toward the same attachment objective'):
    return [dict(left_segment_id=left['segment_id'], right_segment_id=right['segment_id'],
                 object_continuity=object_continuity, objective_continuity=objective_continuity,
                 operation_relation=relation, uncertainty=uncertainty, evidence=evidence,
                 suggested_task_label=label) for left, right in zip(records, records[1:])]


def proposal(records, groups=None, decisions=None):
    return {'groups': [group(records)] if groups is None else groups,
            'boundaries': boundaries(records) if decisions is None else decisions}


class GroupingTests(unittest.TestCase):
    def test_segments_9_10_11_trim_same_ribbon_with_inspection_reconcile_initial_singletons(self):
        df = detailed(['Cut ribbon', 'Inspect ribbon length', 'Trim ribbon'])
        df['segment_id'] = [9, 10, 11]
        df['cluster_id'] = [42, 7, 13]
        before = df.copy(deep=True)
        records = s.segment_records(df)
        initial = [group([row], row['activity']) for row in records]
        explanation = 'The person repeatedly trims the same ribbon to adjust its length.'
        decisions = boundaries(records, 'trim ribbon', relation='inspection_pause', evidence=explanation)
        decisions[1]['operation_relation'] = 'refinement'
        result = s._protect_proposals(initial, records, decisions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['source_segment_ids'], [9, 10, 11])
        self.assertEqual(result[0]['suggested_task_label'], 'trim ribbon')
        self.assertEqual(result[0]['explanation'], explanation)
        pd.testing.assert_frame_equal(df, before)

    def test_same_labels_on_different_ribbons_for_different_purposes_stay_separate(self):
        df = detailed(['Cut ribbon', 'Cut ribbon'])
        df['cluster_id'] = [7, 7]
        records = s.segment_records(df)
        result = s._protect_proposals([group(records, 'Cut ribbon')], records,
            boundaries(records, 'Cut ribbon', object_continuity='different',
                       objective_continuity='different', relation='new_operation',
                       evidence='One ribbon is put aside; a different ribbon is taken for another package'))
        self.assertEqual([g['source_segment_ids'] for g in result], [[1], [2]])

    def test_object_or_objective_switch_or_new_operation_each_break_continuity(self):
        records = s.segment_records(detailed(['Handle item', 'Handle item']))
        for update in ({'object_continuity': 'different'}, {'objective_continuity': 'different'},
                       {'operation_relation': 'new_operation'}):
            with self.subTest(update=update):
                decisions = boundaries(records, 'Handle item')
                decisions[0].update(update)
                result = s._protect_proposals([group(records, 'Handle item')], records, decisions)
                self.assertEqual(len(result), 2)

    def test_occluded_or_insufficient_evidence_cannot_join_identical_labels(self):
        records = s.segment_records(detailed(['Trim material', 'Trim material']))
        for update in ({'object_continuity': 'unclear'}, {'objective_continuity': 'unclear'},
                       {'operation_relation': 'unclear'}, {'uncertainty': 'high'}, {'evidence': ' '}):
            with self.subTest(update=update):
                decisions = boundaries(records, 'Trim material')
                decisions[0].update(update)
                self.assertEqual(len(s._protect_proposals([group(records)], records, decisions)), 2)

    def test_unknown_activity_vocabulary_can_group_a_supported_correction_sequence(self):
        records = s.segment_records(detailed(['Polish steel shaft', 'Inspect surface', 'Refine surface finish']))
        initial = [group([row], row['activity']) for row in records]
        decisions = boundaries(records, 'Finish shaft surface', relation='refinement',
                               evidence='The same held shaft is inspected and polished again for a smoother finish')
        result = s._protect_proposals(initial, records, decisions)
        self.assertEqual(result[0]['source_segment_ids'], [1, 2, 3])
        self.assertEqual(result[0]['suggested_task_label'], 'Finish shaft surface')

    def test_a_later_object_switch_does_not_transitively_merge_the_whole_sequence(self):
        records = s.segment_records(detailed(['Trim material', 'Refine material edge', 'Trim material']))
        decisions = boundaries(records, 'Refine edge')
        decisions[1]['object_continuity'] = 'different'
        initial = [group([row], row['activity']) for row in records]
        result = s._protect_proposals(initial, records, decisions)
        self.assertEqual([g['source_segment_ids'] for g in result], [[1, 2], [3]])

    def test_missing_duplicate_out_of_order_or_malformed_boundaries_are_rejected(self):
        records = s.segment_records(detailed(RETRIES[:3]))
        valid = boundaries(records)
        invalid = [valid[:1], valid + [valid[0]], list(reversed(valid)),
                   [dict(valid[0], left_segment_id=True), valid[1]],
                   [dict(valid[0], object_continuity='probably'), valid[1]]]
        for decisions in invalid:
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                s._protect_proposals([group(records)], records, decisions)

    def test_decoration_retries_group_by_objective_across_different_labels_and_clusters(self):
        df = detailed(RETRIES)
        snapshot = df.copy(deep=True)
        records = s.segment_records(df)
        result = s._protect_proposals([group(records)], records, boundaries(records))
        self.assertEqual(result[0]['source_segment_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(result[0]['suggested_task_label'], 'Attach decoration')
        self.assertEqual((result[0]['start_frame'], result[0]['end_frame']), (0, 74))
        pd.testing.assert_frame_equal(df, snapshot)

    def test_cutting_ribbon_and_tying_bow_do_not_auto_merge_even_if_ai_proposes_it(self):
        records = s.segment_records(detailed(['Cut ribbon', 'Tie ribbon bow']))
        safe = s._protect_proposals([group(records, 'Finish gift')], records,
            boundaries(records, objective_continuity='different', relation='new_operation',
                       evidence='The operation changes from shortening material to forming a knot'))
        self.assertEqual([row['source_segment_ids'] for row in safe], [[1], [2]])
        self.assertEqual([row['suggested_task_label'] for row in safe], ['Cut ribbon', 'Tie ribbon bow'])

    def test_ambiguous_actions_do_not_bridge_known_tasks(self):
        records = s.segment_records(detailed(['Insert decoration', 'Unclear activity', 'Tie ribbon bow']))
        safe = s._protect_proposals([group(records)], records, boundaries(records))
        self.assertEqual([row['source_segment_ids'] for row in safe], [[1], [2], [3]])
        self.assertEqual(safe[1]['suggested_task_label'], 'Unclear activity')

    def test_uncertain_or_different_objectives_remain_separate(self):
        records = s.segment_records(detailed(RETRIES[:2]))
        for metadata in ({'uncertainty': 'high'}, {'same': False}):
            with self.subTest(metadata=metadata):
                self.assertEqual(len(s._protect_proposals([group(records, **metadata)], records,
                                                         boundaries(records))), 2)

    def test_identical_labels_do_not_automatically_merge(self):
        records = s.segment_records(detailed(['Attach decoration', 'Attach decoration']))
        editor = s.GroupEditor(records)
        self.assertEqual(len(editor.groups), 2)

    def test_partition_validation_rejects_omissions_duplicates_nonadjacency_and_invented_frames(self):
        records = s.segment_records(detailed(RETRIES[:3]))
        initial = s.singleton_groups(records)
        bad = []
        bad.append(initial[:2])
        bad.append(initial + [initial[0]])
        bad.append([dict(group(records), source_segment_ids=[1, 3])])
        bad.append([dict(group(records), end_frame=999)])
        bad.append([dict(group(records), source_segment_ids=[1, 2, 99])])
        bad.append([dict(group(records), source_segment_ids=[True, 2, 3])])
        for proposal in bad:
            with self.subTest(proposal=proposal):
                with self.assertRaises(ValueError):
                    s.validate_groups(proposal, records)

    def test_frame_or_timestamp_gap_is_not_mergeable(self):
        for mode in ('frame', 'time'):
            df = detailed(RETRIES[:2])
            if mode == 'frame':
                df.loc[1, ['frame_start', 'frame_end']] += 1
            else:
                df.loc[1, ['timestamp_start', 'timestamp_end']] += pd.Timedelta(seconds=1)
            records = s.segment_records(df)
            with self.assertRaises(ValueError):
                s.validate_groups([group(records)], records)
            initial = [group([row], row['activity']) for row in records]
            self.assertEqual(len(s._protect_proposals(initial, records, boundaries(records))), 2)

    def test_multiple_cases_are_rejected(self):
        df = detailed(RETRIES[:2])
        df.loc[1, 'videoID'] = 2
        with self.assertRaisesRegex(ValueError, 'one videoID'):
            s.segment_records(df)


class ReviewEditorTests(unittest.TestCase):
    def test_merge_split_rename_undo_confirm_preserve_sources(self):
        records = s.segment_records(detailed(RETRIES))
        snapshot = deepcopy(records)
        editor = s.GroupEditor(records)
        editor.merge([0, 1, 2, 3, 4], 'Attach decoration')
        editor.split(0, 2)
        self.assertEqual([g['source_segment_ids'] for g in editor.groups], [[1, 2], [3, 4, 5]])
        editor.rename(1, 'Reposition decoration', 'Human observes a separate positioning objective')
        self.assertEqual(editor.confirm()[1]['suggested_task_label'], 'Reposition decoration')
        editor.undo()
        self.assertEqual(editor.groups[1]['suggested_task_label'], 'Attach decoration')
        editor.reset()
        self.assertEqual(len(editor.groups), 5)
        self.assertEqual(records, snapshot)

    def test_user_can_override_ai_separation_with_manual_merge(self):
        records = s.segment_records(detailed(['Cut ribbon', 'Tie ribbon bow']))
        editor = s.GroupEditor(records)
        editor.merge([0, 1], 'Complete ribbon decoration')
        self.assertEqual(editor.confirm()[0]['source_segment_ids'], [1, 2])

    def test_nonadjacent_selection_and_invalid_split_are_rejected_without_mutation(self):
        editor = s.GroupEditor(s.segment_records(detailed(RETRIES[:3])))
        before = deepcopy(editor.groups)
        with self.assertRaises(ValueError):
            editor.merge([0, 2])
        with self.assertRaises(ValueError):
            editor.split(0, 1)
        with self.assertRaises(ValueError):
            editor.rename(0, ' ', 'reason')
        self.assertEqual(editor.groups, before)


class TaskExportTests(unittest.TestCase):
    def test_both_levels_export_offline_with_confirmed_provenance(self):
        df = detailed(RETRIES + ['Cut ribbon', 'Tie ribbon bow'])
        records = s.segment_records(df)
        tasks = events.extractTaskLog(df, confirmed_groups=[group(records[:5])] + s.singleton_groups(records[5:]))
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            for name, frame in (('detailed', df), ('task', tasks)):
                events.exportCSV(frame.copy(deep=True), name, directory)
                events.exportXES(frame.copy(deep=True), name, directory)
            original = pd.read_csv(Path(directory) / 'detailed_eventlog.csv')
            confirmed = pd.read_csv(Path(directory) / 'task_eventlog.csv')
            self.assertEqual(original['activity'].tolist(), df['activity'].tolist())
            self.assertEqual(confirmed['activity'].tolist(), ['Attach decoration', 'Cut ribbon', 'Tie ribbon bow'])
            self.assertEqual(confirmed.iloc[0]['source_segment_ids'], '1,2,3,4,5')
            self.assertIn('group_explanation', confirmed)
            self.assertTrue((Path(directory) / 'detailed_eventlog.xes').is_file())
            self.assertTrue((Path(directory) / 'task_eventlog.xes').is_file())

    def test_confirmed_task_log_keeps_original_times_provenance_and_detailed_log(self):
        df = detailed(RETRIES + ['Cut ribbon', 'Tie ribbon bow'])
        before = df.copy(deep=True)
        records = s.segment_records(df)
        groups = [group(records[:5])] + s.singleton_groups(records[5:])
        result = events.extractTaskLog(df, confirmed_groups=groups)
        self.assertEqual(result['activity'].tolist(), ['Attach decoration', 'Cut ribbon', 'Tie ribbon bow'])
        self.assertEqual(result.iloc[0]['source_segment_ids'], '1,2,3,4,5')
        self.assertEqual(result.iloc[0]['timestamp_start'], df.iloc[0]['timestamp_start'])
        self.assertEqual(result.iloc[0]['timestamp_end'], df.iloc[4]['timestamp_end'])
        self.assertEqual(result.iloc[0]['duration_seconds'], 5)
        self.assertEqual(result['duration_seconds'].sum(), df['duration_seconds'].sum())
        self.assertIn('group_explanation', result)
        pd.testing.assert_frame_equal(df, before)

    def test_existing_task_mapping_still_works_without_semantic_groups(self):
        df = detailed(['crafting left side', 'crafting the left side', 'look down', 'Tie ribbon bow'])
        with contextlib.redirect_stdout(io.StringIO()):
            result = events.extractTaskLog(df)
        self.assertEqual(result['activity'].tolist(), ['Assemble left side', 'Tie ribbon bow'])
        self.assertEqual(result.iloc[0]['source_segment_ids'], '1,2')

    def test_checkpoint_before_review_and_only_confirmed_groups_published(self):
        df = detailed(RETRIES)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "cached gift ' & video.avi"
            video.write_bytes(b'existing cached video; no decoding in this test')
            groups = [group(s.segment_records(df))]
            calls = []
            reviewer = Mock(side_effect=lambda *a, **kw: calls.append('review') or groups)
            with patch.object(events, 'exportCSV', side_effect=lambda *a: calls.append('csv')), \
                 patch.object(events, 'exportXES', side_effect=lambda *a: calls.append('xes')):
                tasks = s.review_semantic_task_log(video, df, directory, 'reviewed', offline=True, reviewer=reviewer)
            self.assertEqual(calls, ['csv', 'xes', 'review'])
            self.assertTrue(reviewer.call_args.kwargs['offline'])
            manifest = Path(directory) / 'reviewed_semantic_groups.json'
            saved = json.loads(manifest.read_text())
            self.assertIs(saved['confirmed'], True)
            self.assertEqual(len(tasks), 1)
            self.assertIsNotNone(s.load_confirmed_review(manifest, saved['identity'], s.segment_records(df)))
            self.assertIsNone(s.load_confirmed_review(manifest, 'changed labels or video', s.segment_records(df)))
            manifest.write_text('{"confirmed":true, "groups":', encoding='utf-8')
            self.assertIsNone(s.load_confirmed_review(manifest, saved['identity'], s.segment_records(df)))

    def test_cancel_preserves_checkpoint_and_does_not_publish_confirmation(self):
        df = detailed(RETRIES)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'cache.avi'
            video.touch()
            reviewer = Mock(side_effect=s.ReviewCancelled('cancelled'))
            with patch.object(events, 'exportCSV') as csv, patch.object(events, 'exportXES') as xes:
                with self.assertRaises(s.ReviewCancelled):
                    s.review_semantic_task_log(video, df, directory, 'reviewed', reviewer=reviewer)
            csv.assert_called_once()
            xes.assert_called_once()
            self.assertFalse((Path(directory) / 'reviewed_semantic_groups.json').exists())


class GeminiProposalTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'GEMINI_API_KEY': 'offline-dummy', 'GOOGLE_API_KEY': '',
                                      'GEMINI_MODEL': 'configured-model', 'RAVEE_SUGGEST_MODE': 'video'})
        env.start()
        self.addCleanup(env.stop)
        patcher = patch.object(genai, 'Client')
        self.client_factory = patcher.start()
        self.addCleanup(patcher.stop)
        self.client = self.client_factory.return_value.__enter__.return_value
        patcher = patch.object(s, '_media_parts', return_value=[])
        self.media = patcher.start()
        self.addCleanup(patcher.stop)

    def test_gemini_receives_labels_timestamps_objective_rules_and_existing_configuration(self):
        df = detailed(RETRIES)
        records = s.segment_records(df)
        self.client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(proposal(records)))
        result = s.propose_task_groups('cached.avi', df)
        self.assertEqual(len(result), 1)
        request = self.client.models.generate_content.call_args.kwargs
        self.assertEqual(request['model'], 'configured-model')
        text = '\n'.join(part.text for part in request['contents'].parts if part.text)
        self.assertIn('shared concrete action objective', text)
        self.assertIn('SAME PHYSICAL OBJECT', text)
        self.assertIn('brief pause to inspect', text)
        self.assertIn('object_continuity', text)
        self.assertIn('timestamp_start', text)
        self.assertIn('Remove decoration', text)
        self.assertNotIn('cluster_id', json.loads(request['contents'].parts[1].text)['segments'][0])
        self.media.assert_called_once()
        self.assertEqual(self.client_factory.call_args.kwargs['http_options'].timeout, 60000)

    def test_batch_boundary_is_reconsidered_for_one_shared_task(self):
        df = detailed(RETRIES)
        records = s.segment_records(df)
        responses = [records[:3], records[3:], records]
        self.client.models.generate_content.side_effect = [SimpleNamespace(text=json.dumps(proposal(rows)))
                                                           for rows in responses]
        result = s.propose_task_groups('cached.avi', df, batch_segments=3)
        self.assertEqual(result[0]['source_segment_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(self.client.models.generate_content.call_count, 3)

    def test_mock_gemini_split_trimming_proposals_are_corrected_by_boundary_evidence(self):
        df = detailed(['Cut ribbon', 'Inspect ribbon length', 'Trim ribbon'])
        df['segment_id'] = [9, 10, 11]
        records = s.segment_records(df)
        initial = [group([row], row['activity']) for row in records]
        evidence = 'The person repeatedly trims the same ribbon to adjust its length.'
        decisions = boundaries(records, 'trim ribbon', relation='refinement', evidence=evidence)
        self.client.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(proposal(records, initial, decisions)))
        result = s.propose_task_groups('existing-cached.avi', df)
        self.assertEqual(result[0]['source_segment_ids'], [9, 10, 11])
        self.assertEqual(result[0]['suggested_task_label'], 'trim ribbon')
        self.assertEqual(result[0]['explanation'], evidence)
        self.client.models.generate_content.assert_called_once()

    def test_model_reports_an_object_or_objective_switch_in_identically_named_actions(self):
        df = detailed(['Cut ribbon', 'Cut ribbon'])
        records = s.segment_records(df)
        decisions = boundaries(records, object_continuity='different', objective_continuity='different',
                               relation='new_operation', evidence='A different ribbon is cut for a different package')
        self.client.models.generate_content.return_value = SimpleNamespace(text=json.dumps(
            proposal(records, [group(records, 'Cut ribbon')], decisions)))
        result = s.propose_task_groups('existing-cached.avi', df)
        self.assertEqual([g['source_segment_ids'] for g in result], [[1], [2]])

    def test_malformed_partial_or_api_failed_proposals_do_not_become_success(self):
        df = detailed(RETRIES[:2])
        missing_boundaries = json.dumps({'groups': [group(s.segment_records(df))]})
        for text in ('', '{"groups": []}', '{"groups": [{"source_segment_ids": [1]}]}', missing_boundaries):
            with self.subTest(text=text):
                self.client.models.generate_content.return_value = SimpleNamespace(text=text)
                with self.assertRaises(RuntimeError):
                    s.propose_task_groups('cached.avi', df)
        self.client.models.generate_content.side_effect = RuntimeError('503 unavailable')
        with self.assertRaisesRegex(RuntimeError, '503'):
            s.propose_task_groups('cached.avi', df)

    def test_missing_key_does_not_create_client(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': '', 'GOOGLE_API_KEY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'Missing'):
                s.propose_task_groups('cached.avi', detailed(RETRIES))
        self.client_factory.assert_not_called()

    def test_media_reuses_v3_encoders_and_identifies_original_segment_boundaries(self):
        from google.genai import types
        records = s.segment_records(detailed(RETRIES[:2]))
        with patch.object(v3, '_context_windows', return_value=(15, [('TARGET SEGMENT', 0, 29),
                         ('AFTER CONTEXT', 30, 74)])), \
             patch.object(v3, '_make_clip', return_value=b'mock clip') as clips:
            parts = MAKE_MEDIA('cached.avi', records, types)
        self.assertEqual([call.args[1:] for call in clips.call_args_list], [(0, 14), (0, 29), (15, 29), (30, 74)])
        texts = '\n'.join(part.text for part in parts if part.text)
        self.assertIn('SEGMENT 1: source frames 0-14', texts)
        self.assertIn('SEGMENT 2: source frames 15-29', texts)
        self.assertIn('AFTER CONTEXT', texts)
        self.assertIn('TRANSITION 1->2: source frames 0-29', texts)
        self.assertIn('not an additional action or attempt', texts)


if __name__ == '__main__':
    unittest.main(verbosity=2)
