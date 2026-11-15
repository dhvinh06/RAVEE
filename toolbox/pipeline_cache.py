"""Input-aware, completed-stage caches for the non-demo RAVEE pipeline.

CSV descriptor tables are video-level summaries, not framewise trajectories.
Only raw IDT trajectories can feed extract_framewise_features and temporal FINCH.
"""

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from fractions import Fraction

import cv2
import numpy as np


VIDEO_SUFFIXES = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm', '.m4v',
                  '.mpg', '.mpeg', '.mts', '.m2ts'}


class CacheMiss(RuntimeError):
    pass


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def code_signature(*paths):
    return [digest_file(path) for path in paths]


def package_versions(*names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def descriptor_tables(directory):
    """Read named rows, including legitimate nonnumeric video names, strictly."""
    result = {}
    for channel in ('hog', 'hof', 'mbh'):
        path = Path(directory) / f'{channel}_features.csv'
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as stream:
            rows = csv.reader(stream)
            header = next(rows, [])
            # pandas.to_csv writes an optional index before the name column.
            start = 1 if header[:1] == ['name'] else 2
            if (len(header) <= start or header[start - 1] != 'name'
                    or header[start:] != [str(i) for i in range(len(header) - start)]):
                raise ValueError(f'Invalid descriptor header: {path}')
            videos = {}
            for row in rows:
                if len(row) != len(header) or not row[start - 1]:
                    raise ValueError(f'Incomplete descriptor row: {path}')
                name = row[start - 1]
                vector = np.asarray(row[start:], dtype=np.float64)
                if name in videos or not np.isfinite(vector).all():
                    raise ValueError(f'Duplicate/nonfinite descriptor row: {path}: {name}')
                videos[name] = vector
            if not videos:
                raise ValueError(f'Empty descriptor table: {path}')
            result[channel] = videos
    return result


def video_info(path, decode=False):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {path}')
        info = {name: cap.get(prop) for name, prop in (
            ('frames', cv2.CAP_PROP_FRAME_COUNT), ('fps', cv2.CAP_PROP_FPS),
            ('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT))}
        if not all(math.isfinite(v) and v > 0 for v in info.values()):
            raise ValueError(f'Invalid video metadata: {path}')
        if decode:
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            if count != int(info['frames']):
                raise ValueError(f'Incomplete video: {path}: {count}/{info["frames"]} frames')
        return info
    finally:
        cap.release()


def validate_preprocessed(path, source, settings):
    """Check the existing reducer's frame count, dimensions and complete decoding."""
    info = video_info(path, decode=True)
    fps, width, height = settings
    expected = int(source['frames'])
    if source['fps'] > fps:
        # Match reduce_video_size's existing keep/drop algorithm exactly.
        ratio = Fraction(source['fps'] / (source['fps'] - fps)).limit_denominator(10000)
        keep, drop = ratio.numerator, ratio.denominator
        cycles, tail = divmod(expected, keep + drop)
        expected = cycles * keep + min(tail, keep)
    resize = source['width'] > width and source['height'] > height
    dimensions = (width, height) if resize else (source['width'], source['height'])
    if (abs(info['frames'] - expected) > 2 or abs(info['fps'] - fps) > 0.01
            or (info['width'], info['height']) != dimensions):
        raise ValueError(f'Preprocessing result does not match input/settings: {path}')


def validate_tracks(path, frame_count):
    tracks = np.load(path, allow_pickle=False, mmap_mode='r')
    fields = ('frame_num', 'trajectory', 'hog', 'hof', 'mbh_x', 'mbh_y')
    if (tracks.ndim != 1 or tracks.size == 0 or tracks.dtype.names is None
            or not set(fields).issubset(tracks.dtype.names)):
        raise ValueError(f'Expected raw structured IDT trajectories: {path}')
    for name, width in (('trajectory', 30), ('hog', 96), ('hof', 108),
                        ('mbh_x', 96), ('mbh_y', 96)):
        values = tracks[name]
        if values.size != len(tracks) * width or not np.isfinite(values).all():
            raise ValueError(f'Invalid IDT descriptor {name}: {path}')
    frames = tracks['frame_num'].reshape(-1)
    if (not np.isfinite(frames).all() or (frames < 14).any()
            or (frames > frame_count).any() or (frames != np.floor(frames)).any()):
        raise ValueError(f'Invalid IDT frame indices: {path}')
    # Reject truncated files and appended bytes, not just a readable header.
    if Path(path).stat().st_size != tracks.offset + tracks.nbytes:
        raise ValueError(f'Incomplete IDT array: {path}')


def framewise_rows(path):
    tracks = np.load(path, allow_pickle=False, mmap_mode='r')
    ends = np.unique(tracks['frame_num'].reshape(-1)).astype(int)
    # Each trajectory contributes [end - 14, end], as in feature_extraction.py.
    return 15 + int(np.minimum(np.diff(ends), 15).sum())


def validate_fv(arrays, rows=None):
    value = arrays['features']
    if (value.ndim != 2 or min(value.shape) < 2 or not np.isfinite(value).all()
            or (rows is not None and value.shape[0] != rows)):
        raise ValueError('Invalid framewise Fisher Vectors')


def validate_finch(arrays, rows, required):
    c, counts, req = arrays['c'], arrays['num_clust'], arrays['req_c']
    if (c.ndim != 2 or c.shape[0] != rows or c.shape[1] < 1
            or counts.shape != (c.shape[1],)):
        raise ValueError('Incomplete FINCH partitions')
    for value in (c, counts, req):
        if not np.isfinite(value).all() or (value != np.floor(value)).any() or (value < 0).any():
            raise ValueError('Invalid FINCH labels/counts')
    if any(len(np.unique(c[:, i])) != count for i, count in enumerate(counts)):
        raise ValueError('FINCH counts do not match partitions')
    if required is None:
        if req.size:
            raise ValueError('Unexpected requested FINCH partition')
    elif req.shape != (rows,) or len(np.unique(req)) != required:
        raise ValueError('Incomplete requested FINCH partition')


class StageCache:
    """A cache hit requires a final manifest, exact inputs and artifact checksum.

    Producers run in private directories. Publish the artifact first, then replace
    the completion manifest atomically; exceptions never mark a stage complete.
    """

    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)

    def location(self, stage, inputs, filename):
        identity = fingerprint({'version': 1, 'inputs': inputs,
                                'implementation': digest_file(__file__)})
        directory = self.root / stage / identity
        return directory / filename, directory / 'complete.json'

    def lookup(self, stage, inputs, filename):
        path, manifest = self.location(stage, inputs, filename)
        try:
            record = json.loads(manifest.read_text(encoding='utf-8'))
            if (record['complete'] is True and record['inputs'] == inputs
                    and record['sha256'] == digest_file(path)):
                return path
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def file(self, stage, inputs, filename, producer, validator):
        hit = self.lookup(stage, inputs, filename)
        if hit is not None:
            print(f'Cache hit: {stage}: {filename}')
            return hit
        with tempfile.TemporaryDirectory(prefix=f'{stage}-', dir=self.root) as work:
            temporary = Path(work) / filename
            producer(temporary)
            validator(temporary)
            checksum = digest_file(temporary)
            destination, manifest = self.location(stage, inputs, filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            pending = Path(work) / 'complete.json'
            pending.write_text(json.dumps({'complete': True, 'inputs': inputs,
                                           'sha256': checksum}, sort_keys=True), encoding='utf-8')
            os.replace(pending, manifest)
            return destination

    def arrays(self, stage, inputs, producer, validator):
        def produce(path):
            arrays = producer()
            validator(arrays)
            with path.open('wb') as stream:
                np.savez(stream, **arrays)

        def read(path):
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            validator(arrays)
            return arrays

        hit = self.lookup(stage, inputs, 'arrays.npz')
        if hit is not None:
            try:
                result = read(hit)
                print(f'Cache hit: {stage}')
                return result
            except (OSError, ValueError, KeyError):
                # An invalid archive must be recomputed, even with a valid checksum.
                _, marker = self.location(stage, inputs, 'arrays.npz')
                marker.unlink(missing_ok=True)
        return read(self.file(stage, inputs, 'arrays.npz', produce, read))


def build_signature(idt_directory):
    """Hash image sources, excluding mounted runtime data and generated files."""
    root = Path(idt_directory)
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (any(part in {'.git', 'data', 'features', '__pycache__', 'build'}
                for part in relative.parts) or path.suffix in {'.pyc', '.pyd', '.so'}):
            continue
        if path.is_file():
            files.append((relative.as_posix(), digest_file(path)))
    return fingerprint(files)


def ensure_image(idt_directory, signature, rebuild=False):
    image = f'ravee-idt:{signature[:20]}'
    result = subprocess.run(
        ['docker', 'image', 'inspect', '--format',
         '{{index .Config.Labels "ravee.build"}}', image],
        check=False, capture_output=True, text=True)
    if rebuild or result.returncode != 0 or result.stdout.strip() != signature:
        subprocess.run(['docker', 'build', '--label', f'ravee.build={signature}',
                        '-t', image, '.'], cwd=str(idt_directory), check=True)
    return image


def extract_idt(prepared, destination, image):
    """Mount only this job's input; baked-in or old data cannot enter the run."""
    with tempfile.TemporaryDirectory(prefix='idt-input-', dir=destination.parent) as work:
        data = Path(work) / 'data'
        data.mkdir()
        shutil.copy2(prepared, data / prepared.name)
        output = Path(work) / 'features'
        output.mkdir()
        subprocess.run(
            ['docker', 'run', '--rm', '-e', 'RAVEE_RAW_ONLY=1',
             '-v', f'{data.resolve()}:/densetrack/data:ro',
             '-v', f'{output.resolve()}:/densetrack/features', image], check=True)
        shutil.copy2(output / f'{prepared.stem}-trajectory.npy', destination)


def prepare_features(project, cache_directory, reduce_video_size, settings,
                     extract_framewise_features, *, allow_idt=False,
                     adopt_preprocessing=False, adopt_idt=False, rebuild=False):
    """Return exact per-input mappings and framewise FVs, without touching labels."""
    project = Path(project)
    cache = StageCache(cache_directory)
    source_dir, idt = project / 'video_input', project / 'IDT'
    sources = sorted(path for path in source_dir.iterdir()
                     if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not sources:
        raise ValueError(f'No input videos in {source_dir}')
    stems = [path.stem.casefold() for path in sources]
    if len(set(stems)) != len(stems):
        raise ValueError('Input videos must have distinct stems (including case)')
    tables = descriptor_tables(idt / 'features')
    signature = build_signature(idt)
    image = None
    results = {}
    for case_id, source in enumerate(sources, 1):
        name = f'{case_id}_{source.stem}_reduced.avi'
        source_hash = digest_file(source)
        source_metadata = video_info(source)
        pre_inputs = {'source': str(source.resolve()), 'sha256': source_hash,
                      'settings': list(settings), 'name': name,
                      'code': code_signature(project / 'toolbox' / 'preprocessing.py',
                                             project / 'toolbox' / 'file_video_stream.py'),
                      'versions': {'opencv': cv2.__version__}}

        def preprocess(destination):
            legacy = idt / 'data' / name
            if adopt_preprocessing and legacy.is_file():
                if legacy.stat().st_mtime_ns < source.stat().st_mtime_ns:
                    raise ValueError(f'Legacy preprocessing predates input: {legacy}')
                shutil.copy2(legacy, destination)
            else:
                if not allow_idt:
                    raise CacheMiss(f'No legacy preprocessing file to adopt: {legacy}')
                reduce_video_size(str(source), str(destination.parent), case_id, source.stem)

        # Resuming without a valid preprocessing cache must be explicit as well.
        if (not allow_idt and not adopt_preprocessing
                and cache.lookup('preprocessing', pre_inputs, name) is None):
            channels = [channel.upper() for channel, rows in tables.items()
                        if Path(name).stem in rows]
            raise CacheMiss(
                f'No verified preprocessing cache for {source.name}. '
                'Use --adopt-existing-preprocessing to validate IDT/data, '
                'or --allow-idt to permit preprocessing and missing extraction. '
                f'Existing global CSV channels: {channels or "none"}; '
                'global descriptors cannot replace framewise trajectories.')
        prepared = cache.file('preprocessing', pre_inputs, name, preprocess,
                              lambda path: validate_preprocessed(path, source_metadata, settings))
        pre_hash = digest_file(prepared)
        frame_count = int(video_info(prepared)['frames'])
        idt_inputs = {'video': pre_hash, 'source': source_hash, 'image_sources': signature,
                      'mode': 'raw', 'adjust_camera': True}

        def trajectories(destination):
            nonlocal image
            legacy = idt / 'features' / f'{prepared.stem}-trajectory.npy'
            if adopt_idt and legacy.is_file():
                if legacy.stat().st_mtime_ns < source.stat().st_mtime_ns:
                    raise ValueError(f'Legacy trajectories predate input: {legacy}')
                shutil.copy2(legacy, destination)
                return
            channels = [channel.upper() for channel, rows in tables.items()
                        if prepared.stem in rows]
            if not allow_idt:
                raise CacheMiss(
                    f'No completed raw IDT cache for {source.name}. '
                    f'Existing video-level CSV descriptors: {channels or "none"}. '
                    'These vectors have no frame indices and cannot feed temporal FINCH. '
                    'Resume verified raw .npy output with --adopt-existing-idt, or explicitly '
                    'allow the missing extraction with --allow-idt. Existing CSVs are preserved.')
            if image is None:
                image = ensure_image(idt, signature, rebuild)
            extract_idt(prepared, destination, image)

        raw = cache.file('idt', idt_inputs, f'{prepared.stem}-trajectory.npy', trajectories,
                         lambda path: validate_tracks(path, frame_count))
        fv_inputs = {'trajectories': digest_file(raw),
                     'code': code_signature(project / 'toolbox' / 'feature_extraction.py'),
                     'versions': package_versions('numpy', 'scikit-learn')}

        def fisher_vectors():
            with tempfile.TemporaryDirectory(prefix='fv-work-', dir=cache.root) as work:
                return {'features': extract_framewise_features(str(raw), work)}

        expected_rows = framewise_rows(raw)
        features = cache.arrays('fisher_vectors', fv_inputs, fisher_vectors,
                                lambda data: validate_fv(data, expected_rows))['features']
        key = f'{case_id}-{source.stem}'
        results[key] = {'features': features, 'video_path': str(prepared), 'case_id': case_id,
                        'video_name': source.stem,
                        'fv_signature': fingerprint(fv_inputs),
                        'fv_sha256': hashlib.sha256(features.tobytes()).hexdigest()}
    return cache, results


def cached_finch(cache, case, required, finch, project):
    inputs = {'fv': case['fv_signature'], 'sha256': case['fv_sha256'],
              'shape': list(case['features'].shape), 'required_cluster': required,
              'tw_finch': True, 'code': code_signature(Path(project) / 'finch' / 'finch.py'),
              'versions': package_versions('numpy', 'scikit-learn', 'scipy', 'pynndescent')}

    def produce():
        c, counts, requested = finch(case['features'], req_clust=required, tw_finch=True)
        return {'c': np.asarray(c).reshape(len(case['features']), -1),
                'num_clust': np.asarray(counts),
                'req_c': np.asarray([] if requested is None else requested)}

    arrays = cache.arrays('finch', inputs, produce,
                          lambda data: validate_finch(data, len(case['features']), required))
    return arrays['c'], arrays['num_clust'], None if required is None else arrays['req_c']
