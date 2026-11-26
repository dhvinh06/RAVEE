"""Deterministic CSV/XES regression, including legacy epoch-encoded numbers."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import pm4py
from tests.offline_support import detailed_events

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'toolbox'))
from toolbox import event_log_creation as events
from toolbox.semantic_aggregation import segment_records


class EventSerializationTests(unittest.TestCase):
    def assert_roundtrip(self, directory, *, legacy, case_id, count):
        expected = detailed_events(case_id, count)
        source = expected.copy(deep=True)
        numeric_columns = [column for column in (*events._INTEGER_EVENT_COLUMNS, 'duration_seconds')
                           if column in expected]
        if legacy:
            # Reproduce the old exporter bug without reading a mutable run:
            # object-dtype numbers became epoch nanoseconds; fractional durations
            # were lost and must be recovered from the true start/end timestamps.
            for column in numeric_columns:
                source[column] = source[column].map(
                    lambda value: pd.Timestamp(int(value), unit='ns', tz='UTC').isoformat())
        path = Path(directory) / 'immutable_input.csv'
        source.to_csv(path, index=False)
        original_bytes = path.read_bytes()
        raw = pd.read_csv(path)
        if legacy:
            self.assertTrue(raw['videoID'].str.startswith('1970-01-01').all())
            self.assertTrue(raw[raw['videoID'].astype(str) == str(case_id)].empty)
            with self.assertRaises((ValueError, TypeError)):
                segment_records(raw)
        recovered = events.readDetailedCSV(path)
        case = recovered[recovered['videoID'] == case_id].copy()
        self.assertEqual(len(case), count)
        self.assertEqual([row['segment_id'] for row in segment_records(case)], list(range(1, count + 1)))
        for column in numeric_columns:
            pd.testing.assert_series_equal(case[column], expected[column], check_dtype=False)
        # This object dtype is produced by concat with createDF() in the pipeline.
        untyped = case.copy(deep=True)
        for column in numeric_columns:
            if column in untyped:
                untyped[column] = untyped[column].astype(object)
        broken = pm4py.format_dataframe(untyped.copy(deep=True), case_id='videoID',
                                        activity_key='activity', timestamp_key='timestamp_start')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(broken['segment_id']))
        snapshot = untyped.copy(deep=True)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            events.exportCSV(untyped, 'roundtrip', directory)
            events.exportXES(untyped, 'roundtrip', directory)
            saved = pd.read_csv(Path(directory) / 'roundtrip_eventlog.csv')
            self.assertEqual(saved.columns.tolist(), raw.columns.tolist())
            self.assertEqual(saved['case:concept:name'].astype(str).unique().tolist(), [str(case_id)])
            for column in numeric_columns:
                if column in saved:
                    self.assertTrue(pd.api.types.is_numeric_dtype(saved[column]), column)
                    pd.testing.assert_series_equal(saved[column].reset_index(drop=True),
                        case[column].reset_index(drop=True), check_dtype=False, check_names=False)
            loaded = events.readDetailedCSV(Path(directory) / 'roundtrip_eventlog.csv')
            records = segment_records(loaded[loaded['videoID'] == case_id])
            self.assertEqual([row['segment_id'] for row in records], list(range(1, count + 1)))
            self.assertEqual(saved['activity'].tolist(), case['activity'].tolist())
            xes = pm4py.read_xes(str(Path(directory) / 'roundtrip_eventlog.xes'))
            self.assertEqual(len(xes), count)
            self.assertTrue(pd.api.types.is_numeric_dtype(xes['frame_start']))
            self.assertEqual(xes['segment_id'].tolist(), list(range(1, count + 1)))
            self.assertEqual(xes['concept:name'].tolist(), expected['activity'].tolist())
        pd.testing.assert_frame_equal(untyped, snapshot)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), hashlib.sha256(original_bytes).digest())

    def test_legacy_eleven_segments_roundtrip_and_recover_fractional_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assert_roundtrip(directory, legacy=True, case_id=1, count=11)

    def test_current_twenty_segments_roundtrip_for_a_different_case(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assert_roundtrip(directory, legacy=False, case_id=2, count=20)


if __name__ == '__main__':
    unittest.main(verbosity=2)
