"""Input-aware, completed-stage caches for the non-demo RAVEE pipeline.

CSV descriptor tables are video-level summaries, not framewise trajectories.
Only raw IDT trajectories can feed extract_framewise_features and temporal FINCH.
"""

import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import BadZipFile

import cv2
import numpy as np

from toolbox.fv_profiling import FVProfile, measure


VIDEO_SUFFIXES = {'.avi', '.mp4', '.mov', '.mkv', '.wmv', '.webm', '.m4v',
                  '.mpg', '.mpeg', '.mts', '.m2ts'}

# Preserve the original v1 namespace. Input fingerprints below still include
# source contents, settings, extractor sources and package versions. Hashing this
# entire orchestration module made unrelated UI changes invalidate every stage.
CACHE_IMPLEMENTATION_V1 = '8279078848cad387cc3285d06b49de91eaecd15880a1abf3858f0c1ca094738b'


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
            raise ValueError(f'Cannot open video: {path}; readable video expected=True, '
                             f'actual=False; path_length={len(str(path))}')
        info = {name: cap.get(prop) for name, prop in (
            ('frames', cv2.CAP_PROP_FRAME_COUNT), ('fps', cv2.CAP_PROP_FPS),
            ('width', cv2.CAP_PROP_FRAME_WIDTH), ('height', cv2.CAP_PROP_FRAME_HEIGHT))}
        invalid = [f'{name}: expected finite > 0, actual={value}'
                   for name, value in info.items() if not math.isfinite(value) or value <= 0]
        if invalid:
            raise ValueError(f'Invalid video metadata: {path}; ' + '; '.join(invalid))
        if decode:
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
            if count != int(info['frames']):
                raise ValueError(f'Incomplete video: {path}; decoded frame_count '
                                 f'expected={int(info["frames"])} from metadata, actual={count}')
        return info
    finally:
        cap.release()


def validate_preprocessed(path, source, settings):
    """Check the existing reducer's frame count, dimensions and complete decoding."""
    info = video_info(path, decode=True)
    fps, width, height = settings
    expected = int(source['frames'])
    if source['fps'] > fps:
        # Match ggT()/kuerzen() and the integer counters in the existing reducer.
        # A rational approximation changes that finite-video schedule: at
        # 30000/1001 FPS the reducer's float GCD is tiny, so a short input can
        # remain entirely inside its first keep run. Do not silently resample or
        # change that legacy contract in a cache validator.
        dividend, divisor = source['fps'], source['fps'] - fps
        while dividend % divisor != 0:
            dividend, divisor = divisor, dividend % divisor
        keep = math.ceil(source['fps'] / divisor)
        drop = math.ceil((source['fps'] - fps) / divisor)
        cycles, tail = divmod(expected, keep + drop)
        expected = cycles * keep + min(tail, keep)
    resize = source['width'] > width and source['height'] > height
    dimensions = (width, height) if resize else (source['width'], source['height'])
    mismatches = []
    if abs(info['frames'] - expected) > 2:
        mismatches.append(f'frame_count expected={expected} (+/- 2), actual={info["frames"]:g}')
    if abs(info['fps'] - fps) > 0.01:
        mismatches.append(f'fps expected={fps:g} (+/- 0.01), actual={info["fps"]:.12g}')
    if (info['width'], info['height']) != dimensions:
        mismatches.append(f'resolution expected={dimensions[0]:g}x{dimensions[1]:g}, '
                          f'actual={info["width"]:g}x{info["height"]:g}')
    if mismatches:
        raise ValueError(f'Preprocessing result does not match input/settings: {path}; '
                         + '; '.join(mismatches))


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
    """A cache hit requires a final manifest, matching inputs and artifact checksum.

    Producers run in private directories. Publish the artifact first, then replace
    the completion manifest atomically; exceptions never mark a stage complete.
    """

    def __init__(self, directory, *, create=True):
        self.root = Path(directory)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)

    def location(self, stage, inputs, filename):
        identity = fingerprint({'version': 1, 'inputs': inputs,
                                'implementation': CACHE_IMPLEMENTATION_V1})
        directory = self.root / stage / identity
        return directory / filename, directory / 'complete.json'

    def lookup(self, stage, inputs, filename):
        path, manifest = self.location(stage, inputs, filename)
        hit = self._verified_file(path, manifest, inputs)
        if hit is not None:
            return hit
        # Earlier versions also salted their namespace with the module's source
        # hash. Exact provenance + checksum + the caller's stage validator let
        # those completed artifacts remain reusable without moving/deleting them.
        for old_manifest in (self.root / stage).glob('*/complete.json'):
            if old_manifest != manifest:
                hit = self._verified_file(old_manifest.parent / filename, old_manifest, inputs)
                if hit is not None:
                    return hit
        return None

    def lookup_preprocessing(self, project, inputs):
        """Reuse verified preprocessing for a video under a relocated project root."""
        hit = self.lookup('preprocessing', inputs, inputs['name'])
        if hit is not None:
            return hit

        source = Path(inputs['source'])
        if source.parent != project.resolve() / 'video_input':
            return None
        _, _, expected_suffix = inputs['name'].partition('_')
        for manifest in sorted((self.root / 'preprocessing').glob('*/complete.json')):
            try:
                record = json.loads(manifest.read_text(encoding='utf-8'))
                if record['complete'] is not True:
                    continue
                previous = record['inputs']
                old_source = Path(previous['source'])
                old_name = previous['name']
                if not isinstance(old_name, str):
                    continue
                old_id, _, suffix = old_name.partition('_')
                if (not old_source.is_absolute() or old_source == source
                        or old_source.parent.name != 'video_input' or old_source.name != source.name
                        or not old_id.isdecimal() or suffix != expected_suffix
                        or previous != dict(inputs, source=str(old_source), name=old_name)):
                    continue
                artifact = manifest.parent / old_name
                if record['sha256'] == digest_file(artifact):
                    return artifact
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None

    @staticmethod
    def _verified_file(path, manifest, inputs):
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
            try:
                validator(hit)
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                pass
            else:
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

    def arrays(self, stage, inputs, producer, validator, *, profile=None):
        def produce(path):
            arrays = producer()
            validator(arrays)
            with measure(profile, 'cache_serialization'):
                with path.open('wb') as stream:
                    np.savez(stream, **arrays)

        def read(path):
            with measure(profile, 'cache_array_loading'):
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
            validator(arrays)
            return arrays

        with measure(profile, 'cache_lookup'):
            hit = self.lookup(stage, inputs, 'arrays.npz')
        if profile is not None:
            profile.metadata['cache_hit'] = False
        if hit is not None:
            try:
                result = read(hit)
                if profile is not None:
                    profile.metadata['cache_hit'] = True
                print(f'Cache hit: {stage}')
                return result
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                # An invalid archive must be recomputed, even with a valid checksum.
                # file() revalidates hits before rebuilding. Keep the old
                # manifest until a valid replacement is published atomically.
                pass
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


def discover_videos(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)


def source_cases(project, videos=None):
    """Select exact paths while retaining the historical directory case IDs."""
    available = discover_videos(Path(project) / 'video_input')
    sources = available if videos is None else [Path(path).resolve() for path in videos]
    if not sources:
        raise ValueError('No input videos selected or available')
    if len(set(path.stem.casefold() for path in sources)) != len(sources):
        raise ValueError('Input videos must have distinct stems (including case)')
    ids = {path.resolve(): i for i, path in enumerate(available, 1)}
    result = []
    for source in sources:
        if not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f'Not a supported video file: {source}')
        if source.resolve() not in ids:
            ids[source.resolve()] = len(ids) + 1
        result.append((ids[source.resolve()], source))
    return result


def preprocessing_inputs(project, source, case_id, settings):
    return {'source': str(source.resolve()), 'sha256': digest_file(source),
            'settings': list(settings), 'name': f'{case_id}_{source.stem}_reduced.avi',
            'code': code_signature(project / 'toolbox' / 'preprocessing.py',
                                   project / 'toolbox' / 'file_video_stream.py'),
            'versions': {'opencv': cv2.__version__}}


def idt_inputs_for(prepared, source_hash, signature):
    return {'video': digest_file(prepared), 'source': source_hash,
            'image_sources': signature, 'mode': 'raw', 'adjust_camera': True}


def fv_inputs_for(project, raw):
    return {'trajectories': digest_file(raw),
            'code': code_signature(project / 'toolbox' / 'feature_extraction.py'),
            'versions': package_versions('numpy', 'scikit-learn')}


def feature_case(source, case_id, prepared, features, fv_inputs):
    return {'features': features, 'video_path': str(prepared), 'case_id': case_id,
            'video_name': source.stem, 'fv_signature': fingerprint(fv_inputs),
            'fv_sha256': hashlib.sha256(features.tobytes()).hexdigest()}


def finch_inputs_for(case, required, project):
    return {'fv': case['fv_signature'], 'sha256': case['fv_sha256'],
            'shape': list(case['features'].shape), 'required_cluster': required,
            'tw_finch': True, 'code': code_signature(Path(project) / 'finch' / 'finch.py'),
            'versions': package_versions('numpy', 'scikit-learn', 'scipy', 'pynndescent')}


def inspect_video_cache(project, cache_directory, source, settings, required=None):
    """Read-only verification using the same keys/validators as execution.

    Never invokes a producer, removes a manifest or starts Docker. A downstream
    stage is only usable when its complete dependency chain is verified.
    """
    project = Path(project)
    cache = StageCache(cache_directory, create=False)
    case_id, source = source_cases(project, [source])[0]
    stages = {name: False for name in ('preprocessing', 'idt', 'fisher_vectors', 'finch')}
    error = None
    try:
        pre = preprocessing_inputs(project, source, case_id, settings)
        prepared = cache.lookup_preprocessing(project, pre)
        if prepared is not None:
            validate_preprocessed(prepared, video_info(source), settings)
            stages['preprocessing'] = True
            raw_inputs = idt_inputs_for(prepared, pre['sha256'], build_signature(project / 'IDT'))
            raw = cache.lookup('idt', raw_inputs, f'{prepared.stem}-trajectory.npy')
            if raw is not None:
                validate_tracks(raw, int(video_info(prepared)['frames']))
                stages['idt'] = True
                fv_inputs = fv_inputs_for(project, raw)
                fv = cache.lookup('fisher_vectors', fv_inputs, 'arrays.npz')
                if fv is not None:
                    with np.load(fv, allow_pickle=False) as archive:
                        arrays = {key: archive[key] for key in archive.files}
                    validate_fv(arrays, framewise_rows(raw))
                    stages['fisher_vectors'] = True
                    case = feature_case(source, case_id, prepared, arrays['features'], fv_inputs)
                    partition = cache.lookup('finch', finch_inputs_for(case, required, project), 'arrays.npz')
                    if partition is not None:
                        with np.load(partition, allow_pickle=False) as archive:
                            arrays = {key: archive[key] for key in archive.files}
                        validate_finch(arrays, len(case['features']), required)
                        stages['finch'] = True
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
        error = str(exc)
    status = 'Cached' if all(stages.values()) else ('Partial Cache' if any(stages.values()) else 'New')
    return {'status': status, 'stages': stages, 'error': error}


def prepare_features(project, cache_directory, reduce_video_size, settings,
                     extract_framewise_features, *, allow_idt=False,
                     adopt_preprocessing=False, adopt_idt=False, rebuild=False,
                     videos=None, progress=None, fv_profile=False):
    """Return exact per-input mappings and framewise FVs, without touching labels."""
    project = Path(project)
    cache = StageCache(cache_directory)
    idt = project / 'IDT'
    sources = source_cases(project, videos)
    report = progress or (lambda message: None)
    tables = descriptor_tables(idt / 'features')
    signature = build_signature(idt)
    image = None
    results = {}
    for case_id, source in sources:
        report(f'Verifying preprocessing cache: {source.name}')
        pre_inputs = preprocessing_inputs(project, source, case_id, settings)
        name, source_hash = pre_inputs['name'], pre_inputs['sha256']
        source_metadata = video_info(source)

        def preprocess(destination):
            legacy = idt / 'data' / name
            if adopt_preprocessing and legacy.is_file():
                if legacy.stat().st_mtime_ns < source.stat().st_mtime_ns:
                    raise ValueError(f'Legacy preprocessing predates input: {legacy}')
                shutil.copy2(legacy, destination)
            else:
                if not allow_idt:
                    raise CacheMiss(f'No legacy preprocessing file to adopt: {legacy}')
                report(f'Preprocessing video: {source.name}')
                reduce_video_size(str(source), str(destination.parent), case_id, source.stem)

        prepared = cache.lookup_preprocessing(project, pre_inputs)
        if prepared is not None:
            try:
                validate_preprocessed(prepared, source_metadata, settings)
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                prepared = None
        if prepared is None:
            if not allow_idt and not adopt_preprocessing:
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
        else:
            print(f'Cache hit: preprocessing: {prepared.name}')
        report(f'Preprocessing ready: {source.name}')
        frame_count = int(video_info(prepared)['frames'])
        idt_inputs = idt_inputs_for(prepared, source_hash, signature)

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
                report('Preparing Docker image for missing IDT extraction')
                image = ensure_image(idt, signature, rebuild)
            report(f'Extracting IDT trajectories: {source.name}')
            extract_idt(prepared, destination, image)

        report(f'Verifying IDT cache: {source.name}')
        raw = cache.file('idt', idt_inputs, f'{prepared.stem}-trajectory.npy', trajectories,
                         lambda path: validate_tracks(path, frame_count))
        report(f'IDT trajectories ready: {source.name}')
        fv_inputs = fv_inputs_for(project, raw)
        profile = FVProfile() if (fv_profile or os.environ.get('RAVEE_PROFILE_FV') == '1') else None
        if profile is not None:
            profile.metadata['video'] = source.name
            profile.metadata['versions'] = fv_inputs['versions']

        def fisher_vectors():
            report(f'Computing Fisher Vectors: {source.name}')
            # Older standalone/injected extractors keep their two-argument API.
            # Inspect supported keywords rather than retrying on TypeError,
            # which could hide a real extraction failure and run work twice.
            try:
                parameters = inspect.signature(extract_framewise_features).parameters
            except (TypeError, ValueError):
                parameters = {}
            options = {}
            if 'progress' in parameters and progress is not None:
                options['progress'] = lambda message: report(f'{message}: {source.name}')
            if 'profile' in parameters:
                options['profile'] = profile
            with tempfile.TemporaryDirectory(prefix='fv-work-', dir=cache.root) as work:
                with measure(profile, 'extraction_total'):
                    return {'features': extract_framewise_features(str(raw), work, **options)}

        expected_rows = framewise_rows(raw)
        if profile is not None:
            profile.metadata['expected_frame_count'] = expected_rows
        report(f'Verifying Fisher Vector cache: {source.name}')
        try:
            with measure(profile, 'cache_stage_total'):
                features = cache.arrays('fisher_vectors', fv_inputs, fisher_vectors,
                                       lambda data: validate_fv(data, expected_rows),
                                       profile=profile)['features']
            if profile is not None:
                profile.matrix('features', features)
        finally:
            if profile is not None:
                print('FV profile: ' + json.dumps(profile.as_dict(), sort_keys=True))
        key = f'{case_id}-{source.stem}'
        report(f'Fisher Vectors ready: {source.name}')
        results[key] = feature_case(source, case_id, prepared, features, fv_inputs)
    return cache, results


def cached_finch(cache, case, required, finch, project, *, progress=None):
    inputs = finch_inputs_for(case, required, project)
    report = progress or (lambda message: None)
    report(f'Verifying FINCH cache: {case["video_name"]}')

    def produce():
        report(f'Clustering with FINCH: {case["video_name"]}')
        c, counts, requested = finch(case['features'], req_clust=required, tw_finch=True)
        return {'c': np.asarray(c).reshape(len(case['features']), -1),
                'num_clust': np.asarray(counts),
                'req_c': np.asarray([] if requested is None else requested)}

    arrays = cache.arrays('finch', inputs, produce,
                          lambda data: validate_finch(data, len(case['features']), required))
    report(f'FINCH ready: {case["video_name"]}')
    return arrays['c'], arrays['num_clust'], None if required is None else arrays['req_c']
