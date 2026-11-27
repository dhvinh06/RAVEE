import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from toolbox import pipeline_cache as cache


class IDTCUDABackendTests(unittest.TestCase):
    def test_backend_and_docker_flags(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cache.idt_backend(), 'cpu')
            self.assertEqual(cache.idt_worker_count(adopt_legacy=True), 1)
            self.assertEqual(cache.idt_worker_count(backend='cuda'), 1)
        for backend, workers in [('cpu', 6), ('cuda', 1), ('cuda', 4)]:
            with self.subTest(backend=backend, workers=workers), patch.dict(os.environ, {
                'RAVEE_IDT_BACKEND': backend, 'RAVEE_IDT_WORKERS': str(workers),
            }), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepared = root / 'scene.avi'
                prepared.touch()

                def docker(argv, **kwargs):
                    mount = next(value for value in argv if value.endswith(':/densetrack/features'))
                    (Path(mount.removesuffix(':/densetrack/features')) / 'scene-trajectory.npy').touch()

                with patch.object(cache.subprocess, 'run', side_effect=docker) as run:
                    selected = cache.idt_backend()
                    count = cache.idt_worker_count(backend=selected)
                    cache.extract_idt(prepared, root / 'output.npy', 'image',
                                      backend=selected, workers=count)
                argv = run.call_args.args[0]
                self.assertIn(f'RAVEE_IDT_WORKERS={workers}', argv)
                if backend == 'cuda':
                    self.assertEqual(argv[argv.index('--gpus') + 1], 'all')
                    self.assertIn('RAVEE_IDT_BACKEND=cuda', argv)
                else:
                    self.assertNotIn('--gpus', argv)
        with patch.dict(os.environ, {'RAVEE_IDT_BACKEND': 'cuda', 'RAVEE_IDT_WORKERS': '5'}):
            with self.assertRaisesRegex(ValueError, '1 to 4'):
                cache.idt_worker_count(backend=cache.idt_backend())

    def test_cuda_image_and_fingerprint_are_separate(self):
        with patch.object(cache.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([], 1, stdout=''),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0, stdout='signature sha256:cuda\n'),
        ]) as run:
            self.assertEqual(cache.ensure_image('IDT', 'signature', backend='cuda'), 'sha256:cuda')
        self.assertEqual(run.call_args_list[1].args[0], [
            'docker', 'build', '-f', 'Dockerfile.cuda', '--label', 'ravee.build=signature',
            '-t', 'ravee-idt:cuda', '.',
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Dockerfile').touch()
            cpu = cache.build_signature(root)
            (root / 'Dockerfile.cuda').touch()
            cuda = cache.build_signature(root, backend='cuda')
            (root / 'cuda').mkdir()
            (root / 'cuda/flow.cu').touch()
            self.assertEqual(cache.build_signature(root), cpu)
            self.assertNotEqual(cache.build_signature(root, backend='cuda'), cuda)
