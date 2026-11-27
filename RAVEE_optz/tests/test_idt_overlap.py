"""Offline regression for IDT chunk ownership and global frame indices."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class IDTOverlapTests(unittest.TestCase):
    def test_uneven_chunks_keep_each_owned_end_frame_once(self):
        path = Path(__file__).resolve().parents[1] / 'IDT' / 'run_idt.py'
        spec = importlib.util.spec_from_file_location('idt_overlap_test_runner', path)
        runner = importlib.util.module_from_spec(spec)
        loaded_windows = []

        def densetrack(frames, *, adjust_camera):
            self.assertTrue(adjust_camera)
            self.assertIsInstance(frames, np.memmap)
            self.assertFalse(frames.flags.writeable)
            loaded_windows.append((int(frames[0, 0, 0]), int(frames[-1, 0, 0])))
            ends = np.repeat(np.arange(15, len(frames)), 2)
            tracks = np.empty(len(ends), dtype=[('frame_num', 'i4'), ('t_pos', 'f4')])
            tracks['frame_num'] = ends
            tracks['t_pos'] = (ends - 7.5) / len(frames)
            return tracks

        class InlineExecutor:
            def __init__(self, **options):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, function, *args):
                result = function(*args)
                return SimpleNamespace(result=lambda: result)

        with patch.dict(sys.modules, {'densetrack': SimpleNamespace(densetrack=densetrack)}):
            spec.loader.exec_module(runner)
        video = np.arange(161, dtype=np.uint8).reshape(-1, 1, 1)
        with tempfile.TemporaryDirectory() as scratch, patch.dict(os.environ), \
                patch.object(runner, 'read_video', return_value=video) as read, \
                patch.object(runner, 'ProcessPoolExecutor', InlineExecutor):
            root = Path(scratch)
            output = root / 'tracks.npy'
            runner.extract_parallel_video(root / 'video.avi', output, workers=4)
            read.assert_called_once()
            self.assertEqual(loaded_windows, [(0, 39), (10, 79), (50, 119), (90, 160)])
            tracks = np.load(output, allow_pickle=False)
            expected = np.repeat(np.arange(15, 161), 2)
            np.testing.assert_array_equal(tracks['frame_num'], expected)
            np.testing.assert_allclose(tracks['t_pos'], (expected - 7.5) / 161)
            self.assertEqual(list(root.iterdir()), [output])
