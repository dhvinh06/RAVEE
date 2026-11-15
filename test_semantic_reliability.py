"""Offline reliability tests. No network, video decoding, GUI or pipeline runs."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime
import unittest
import sys
from unittest.mock import Mock, patch

from test_semantic_aggregation import detailed, proposal, RETRIES
from toolbox import semantic_aggregation as s
from google import genai


class HttpFailure(Exception):
    def __init__(self, code, retry_after=None):
        super().__init__(f'HTTP {code}')
        self.code = code
        self.response = SimpleNamespace(headers={} if retry_after is None else {'Retry-After': retry_after})


class RetryTests(unittest.TestCase):
    def test_all_transient_statuses_retry_with_bounded_full_jitter(self):
        for code in (429, 500, 502, 503, 504):
            with self.subTest(code=code), patch.object(s.time, 'sleep') as sleep, \
                 patch.object(s.random, 'uniform', side_effect=lambda low, high: high / 2) as jitter:
                client = Mock()
                client.models.generate_content.side_effect = [HttpFailure(code)] * 3 + ['success']
                stats = {'api_calls': 0}
                self.assertEqual(s._generate_with_retry(client, {}, stats=stats), 'success')
                self.assertEqual(stats['api_calls'], 4)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [.5, 1., 2.])
                self.assertEqual([call.args for call in jitter.call_args_list], [(0., 1.), (0., 2.), (0., 4.)])

    def test_retry_after_seconds_and_http_date_are_respected(self):
        client = Mock()
        client.models.generate_content.side_effect = [HttpFailure(429, '3'), 'ok']
        with patch.object(s.time, 'sleep') as sleep, patch.object(s.random, 'uniform', return_value=.1):
            self.assertEqual(s._generate_with_retry(client, {}), 'ok')
            sleep.assert_called_once_with(3.)
        header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
        self.assertGreater(s._retry_after(HttpFailure(503, header)), 8)
        self.assertEqual(s._retry_after(HttpFailure(503, 'invalid')), 0)

    def test_nontransient_and_excessive_retry_after_do_not_retry(self):
        for error in (HttpFailure(400), HttpFailure(403), HttpFailure(503, '120')):
            with self.subTest(error=error), patch.object(s.time, 'sleep') as sleep:
                client = Mock()
                client.models.generate_content.side_effect = error
                with self.assertRaises(HttpFailure):
                    s._generate_with_retry(client, {})
                client.models.generate_content.assert_called_once()
                sleep.assert_not_called()

    def test_retry_count_is_bounded_on_persistent_unavailability(self):
        client = Mock()
        client.models.generate_content.side_effect = HttpFailure(503)
        with patch.object(s.time, 'sleep') as sleep:
            with self.assertRaises(HttpFailure):
                s._generate_with_retry(client, {})
        self.assertEqual(client.models.generate_content.call_count, 4)
        self.assertEqual(sleep.call_count, 3)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="semantic ' & cache ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.video = self.root / 'existing.avi'
        self.video.write_bytes(b'existing processed video identity; never decoded')
        self.df = detailed(RETRIES)
        env = patch.dict(os.environ, {'GEMINI_API_KEY': 'offline', 'GOOGLE_API_KEY': '', 'GEMINI_MODEL': 'test-model'})
        env.start()
        self.addCleanup(env.stop)
        for owner, name, options in ((genai, 'Client', {}), (s, '_media_parts', {'return_value': []}),
                                      (s.time, 'sleep', {})):
            patcher = patch.object(owner, name, **options)
            mocked = patcher.start()
            self.addCleanup(patcher.stop)
            if name == 'Client':
                self.factory = mocked
        self.client = self.factory.return_value.__enter__.return_value
        self.client.models.generate_content.side_effect = self.respond

    @staticmethod
    def respond(**kwargs):
        records = json.loads(kwargs['contents'].parts[1].text)['segments']
        return SimpleNamespace(text=json.dumps(proposal(records)))

    def propose(self, stats=None, progress=None, refresh=False):
        return s.propose_task_groups(self.video, self.df, batch_segments=3,
                                    cache_dir=self.root / 'cache', stats=stats, progress=progress,
                                    refresh_suggestions=refresh)

    def test_forced_refresh_rechecks_all_boundaries_then_normal_request_reuses_fresh_results(self):
        stats = {}
        self.propose(stats)
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 0)
        self.propose(stats, refresh=True)
        self.assertEqual(stats['api_calls'], 2)
        self.assertEqual(stats['fresh'], 4)
        self.assertEqual(stats['cache_hits'], 0)
        self.assertEqual(stats['fallback'], 0)
        self.assertTrue(all(item['source'] == 'fresh' for item in stats['boundary_sources']))
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 0)
        self.assertEqual(stats['cache_hits'], 4)
        self.assertEqual(stats['fresh'], 0)

    def test_partial_refresh_keeps_old_cache_and_labels_old_vs_fresh_boundaries(self):
        self.propose()
        old = {path: path.read_bytes() for path in (self.root / 'cache').glob('*.json')}
        attempts = 0
        def partially_available(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts > 1:
                raise HttpFailure(503)
            response = self.respond(**kwargs)
            payload = json.loads(response.text)
            for boundary in payload['boundaries']:
                boundary['evidence'] = 'Fresh visible continuity evidence'
            response.text = json.dumps(payload)
            return response
        self.client.models.generate_content.side_effect = partially_available
        stats, messages = {}, []
        self.propose(stats, messages.append, refresh=True)
        self.assertEqual(stats['api_calls'], 5)
        self.assertEqual(stats['fresh'], 2)
        self.assertEqual(stats['fallback'], 2)
        self.assertEqual(stats['cache_hits'], 2)
        self.assertEqual(stats['unresolved'], 0)
        self.assertEqual([item['source'] for item in stats['boundary_sources']],
                         ['fresh', 'fresh', 'fallback', 'fallback'])
        self.assertTrue(any('OLD cached fallback, not freshly evaluated' in msg for msg in messages))
        self.assertEqual(sum(path.read_bytes() == data for path, data in old.items()), 2)
        self.client.models.generate_content.side_effect = self.respond
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 0)
        self.assertEqual(stats['fresh'], 0)
        self.assertEqual(stats['cache_hits'], 4)

    def test_all_failed_refresh_preserves_every_old_valid_entry_as_fallback(self):
        old_groups = self.propose()
        old = {path: path.read_bytes() for path in (self.root / 'cache').glob('*.json')}
        self.client.models.generate_content.side_effect = HttpFailure(503)
        stats = {}
        result = self.propose(stats, refresh=True)
        self.assertEqual(result, old_groups)
        self.assertEqual(stats['fresh'], 0)
        self.assertEqual(stats['fallback'], 4)
        self.assertEqual(stats['error'], 'HTTP 503')
        self.assertTrue(all(path.read_bytes() == data for path, data in old.items()))

    def test_refresh_without_old_cache_leaves_failed_boundaries_unresolved(self):
        self.client.models.generate_content.side_effect = HttpFailure(503)
        stats = {}
        result = self.propose(stats, refresh=True)
        self.assertEqual(len(result), 5)
        self.assertEqual(stats['fresh'], 0)
        self.assertEqual(stats['fallback'], 0)
        self.assertEqual(stats['unresolved'], 4)
        self.assertFalse(list((self.root / 'cache').glob('*.json')))

    def test_refresh_is_scoped_to_current_case(self):
        self.propose()
        self.df['videoID'] = 2
        self.propose()
        second_case_files = {path: path.read_bytes() for path in (self.root / 'cache').glob('*.json')}
        self.df['videoID'] = 1
        stats = {}
        self.propose(stats, refresh=True)
        self.df['videoID'] = 2
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 0)
        self.assertEqual(stats['cache_hits'], 4)
        self.assertTrue(all(path.read_bytes() == data for path, data in second_case_files.items()))

    def test_cli_refresh_flag_reaches_review_without_calling_gemini(self):
        from toolbox import event_log_creation as events
        csv = self.root / 'detailed.csv'
        self.df.to_csv(csv, index=False)
        argv = ['semantic_aggregation.py', '--video', str(self.video), '--detailed-csv', str(csv),
                '--output-dir', str(self.root / 'review'), '--case-id', '1', '--refresh-suggestions', '--offline']
        with patch.object(sys, 'argv', argv), \
             patch.object(s, 'review_semantic_task_log', return_value=self.df) as review, \
             patch.object(events, 'exportCSV'), patch.object(events, 'exportXES'):
            self.assertEqual(s.main(), 0)
        self.assertTrue(review.call_args.kwargs['refresh_suggestions'])
        self.assertTrue(review.call_args.kwargs['offline'])
        self.factory.assert_not_called()

    def test_partial_failure_keeps_completed_groups_and_resumes_only_missing_boundaries(self):
        count = 0
        def partial(**kwargs):
            nonlocal count
            count += 1
            if count > 1:
                raise HttpFailure(503)
            return self.respond(**kwargs)
        self.client.models.generate_content.side_effect = partial
        stats, messages = {}, []
        groups = self.propose(stats, messages.append)
        self.assertEqual([group['source_segment_ids'] for group in groups], [[1, 2, 3], [4], [5]])
        self.assertEqual(stats['api_calls'], 5)  # One success + four failed attempts, then circuit break.
        self.assertEqual(stats['unresolved'], 2)
        self.assertEqual(len(list((self.root / 'cache').glob('*.json'))), 2)
        self.assertIn('Checking boundary 1/4', messages)
        self.client.models.generate_content.side_effect = self.respond
        groups = self.propose(stats)
        self.assertEqual(groups[0]['source_segment_ids'], [1, 2, 3, 4, 5])
        self.assertEqual(stats['api_calls'], 1)
        self.assertEqual(stats['cache_hits'], 2)
        self.assertEqual(stats['unresolved'], 0)
        self.factory.reset_mock()
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 0)
        self.assertEqual(stats['cache_hits'], 4)
        self.factory.assert_not_called()

    def test_cold_and_warm_call_counts_and_invalidations(self):
        stats = {}
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 2)  # Previously three for five segments/batch size three.
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 0)
        for mutate in (lambda: os.environ.update(GEMINI_MODEL='changed-model'),
                       lambda: self.video.write_bytes(b'changed video'),
                       lambda: self.df.loc.__setitem__((0, 'activity'), 'Changed label')):
            mutate()
            self.propose(stats)
            self.assertEqual(stats['api_calls'], 2)
        with patch.object(s, '_PROMPT', s._PROMPT + ' version update'):
            self.propose(stats)
            self.assertEqual(stats['api_calls'], 2)

    def test_task_prompt_upgrade_preserves_existing_cache_files(self):
        with patch.object(s, '_PROMPT', 'Previous action-level aggregation prompt'):
            self.propose()
        old = {path: path.read_bytes() for path in (self.root / 'cache').glob('*.json')}
        stats = {}
        self.propose(stats)
        self.assertEqual(stats['fresh'], 4)
        self.assertEqual(stats['cache_hits'], 0)
        self.assertTrue(all(path.read_bytes() == content for path, content in old.items()))
        self.propose(stats)
        self.assertEqual(stats['api_calls'], 0)
        self.assertEqual(stats['cache_hits'], 4)

    def test_partial_cache_file_is_not_accepted_and_only_that_boundary_is_rechecked(self):
        self.propose()
        next((self.root / 'cache').glob('*.json')).write_text('{"complete":', encoding='utf-8')
        stats = {}
        self.propose(stats)
        self.assertEqual(stats['cache_hits'], 3)
        self.assertEqual(stats['api_calls'], 1)

    def test_25_segments_need_three_requests_not_five_and_11_need_one(self):
        for size, expected in ((25, 3), (11, 1)):
            stats = {}
            s.propose_task_groups(self.video, detailed(['Work on item'] * size),
                cache_dir=self.root / f'cache-{size}', stats=stats)
            self.assertEqual(stats['api_calls'], expected)
            self.assertEqual(stats['unresolved'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
