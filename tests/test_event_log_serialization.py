"""Local integration regression using the actual 11-segment cached event log."""
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import pm4py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'toolbox'))
from toolbox import event_log_creation as events
from toolbox.semantic_aggregation import segment_records


class EventSerializationTests(unittest.TestCase):
    def test_existing_cached_case_roundtrips_all_eleven_segments(self):
        path = ROOT / 'EventLog_Output/all_videos_detailed_eventlog.csv'
        original_bytes = path.read_bytes()
        raw = pd.read_csv(path)
        # Reproduce the failure present in the real cached CSV, or the original
        # exporter failure if this cache has since been regenerated correctly.
        if raw['videoID'].astype(str).str.startswith('1970-01-01').any():
            self.assertTrue(raw[raw['videoID'].astype(str) == '1'].empty)
            with self.assertRaises((ValueError, TypeError)):
                segment_records(raw)
        recovered = events.readDetailedCSV(path)
        case = recovered[recovered['videoID'] == 1].copy()
        self.assertEqual(len(case), 11)
        self.assertEqual([row['segment_id'] for row in segment_records(case)], list(range(1, 12)))
        # This object dtype is produced by concat with createDF() in the pipeline.
        numeric_columns = [*events._INTEGER_EVENT_COLUMNS, 'duration_seconds']
        untyped = case.copy(deep=True)
        for column in numeric_columns:
            if column in untyped:
                untyped[column] = untyped[column].astype(object)
        broken = pm4py.format_dataframe(untyped.copy(deep=True), case_id='videoID',
                                        activity_key='activity', timestamp_key='timestamp_start')
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(broken['segment_id']))
        snapshot = untyped.copy(deep=True)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            events.exportCSV(untyped, 'roundtrip', directory)
            events.exportXES(untyped, 'roundtrip', directory)
            saved = pd.read_csv(Path(directory) / 'roundtrip_eventlog.csv')
            self.assertEqual(saved.columns.tolist(), raw.columns.tolist())
            self.assertEqual(saved['case:concept:name'].astype(str).unique().tolist(), ['1'])
            for column in numeric_columns:
                if column in saved:
                    self.assertTrue(pd.api.types.is_numeric_dtype(saved[column]), column)
                    pd.testing.assert_series_equal(saved[column].reset_index(drop=True),
                        case[column].reset_index(drop=True), check_dtype=False, check_names=False)
            loaded = events.readDetailedCSV(Path(directory) / 'roundtrip_eventlog.csv')
            records = segment_records(loaded[loaded['videoID'] == 1])
            self.assertEqual([row['segment_id'] for row in records], list(range(1, 12)))
            self.assertEqual(saved['activity'].tolist(), case['activity'].tolist())
            xes = pm4py.read_xes(str(Path(directory) / 'roundtrip_eventlog.xes'))
            self.assertEqual(len(xes), 11)
            self.assertTrue(pd.api.types.is_numeric_dtype(xes['frame_start']))
        pd.testing.assert_frame_equal(untyped, snapshot)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), hashlib.sha256(original_bytes).digest())


if __name__ == '__main__':
    unittest.main(verbosity=2)
