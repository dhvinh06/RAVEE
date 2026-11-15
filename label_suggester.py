"""RAVEE Label Suggester V3: context-aware, evidence-based segment suggestions.

The GUI API and manual confirmation stay unchanged. Separate before/target/after
media parts prevent context actions from being mistaken for the target activity.
Default context: 3 seconds on each side, clipped at video boundaries. Temporary
480px MP4s are sampled at up to 6fps; still frames remain available as fallback.
"""

from __future__ import annotations

import math
import json
import os
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Keep each inline request reasonably small. Files API is intentionally not
# required: short clips avoid the prior upload/processing wait.
_MAX_INLINE_BYTES = 18 * 1024 * 1024
_MAX_DURATION_SECONDS = 20.0
_ENCODE_FPS = 6.0
_GEMINI_SAMPLE_FPS = 3.0
_MAX_IMAGE_EDGE = 480
_CONTEXT_SECONDS = 3.0

_PROMPT = (
    'Label ONLY the TARGET SEGMENT for video-based process mining. '
    'BEFORE CONTEXT and AFTER CONTEXT show object state and continuity only; '
    'never assign an action that occurs only outside the target segment. '
    'Observe visible hand/tool/object motion and state changes inside the target. '
    'Use an open vocabulary: a concise English verb + object label, 2 to 6 words. '
    'There is NO fixed activity list. Use the same plain wording for repeated '
    'observable actions, without inventing differences between continuing segments. '
    'Distinguish forming a NEW ribbon bow from tightening or adjusting an EXISTING '
    'bow. Tie ribbon bow requires visible formation of loops/knot within the target. '
    'Pulling an already formed bow tighter is Tighten ribbon bow; repositioning '
    'existing loops/tails is Adjust ribbon bow. These are conditional bow examples, '
    'not a general candidate label list. Before/after appearance alone does not '
    'prove what occurred behind hands. If hands occlude the knot/loops and the '
    'action cannot be distinguished from direct target evidence, do not guess. '
    'Use exactly Unclear activity when evidence is insufficient, the target action '
    'is obscured, or no single action predominates. Never infer hidden intent. '
    'Return ONLY a JSON object with: label (the short label), evidence (a short '
    'description of directly visible TARGET motion/state change), uncertainty '
    '(low, medium, or high), target_visible (boolean: sufficient direct evidence), '
    'and bow_action (tie_new, tighten_existing, adjust_existing, not_applicable, '
    'or unclear). bow_action describes only the target; use unclear if a bow '
    'action is suspected but cannot be distinguished. For other activities use '
    'not_applicable. With insufficient evidence set uncertainty high, '
    'target_visible false, and label Unclear activity.'
)


def _clip_bounds(start_frame, end_frame, fps):
    max_frames = max(1, int(_MAX_DURATION_SECONDS * fps))
    if end_frame - start_frame + 1 > max_frames:
        start_frame += (end_frame - start_frame + 1 - max_frames) // 2
        end_frame = start_frame + max_frames - 1
    return start_frame, end_frame


def _context_windows(video_path, start_frame, end_frame):
    """Keep roles/bounds explicit, including a disclosed long-target excerpt."""
    context = float(os.getenv('RAVEE_SUGGEST_CONTEXT_SECONDS', str(_CONTEXT_SECONDS)))
    if not math.isfinite(context) or not 0 <= context <= 10:
        raise ValueError('RAVEE_SUGGEST_CONTEXT_SECONDS must be between 0 and 10')
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')
        fps = _get_fps(cap)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError('Cannot determine video boundaries for context sampling')
        if end_frame >= total_frames:
            raise ValueError('Segment ends beyond the end of the video')
    finally:
        cap.release()
    padding = round(context * fps)
    windows = []
    if padding and start_frame > 0:
        windows.append(('BEFORE CONTEXT', max(0, start_frame - padding), start_frame - 1))
    target_start, target_end = _clip_bounds(start_frame, end_frame, fps)
    windows.append(('TARGET SEGMENT', target_start, target_end))
    if padding and end_frame < total_frames - 1:
        windows.append(('AFTER CONTEXT', end_frame + 1, min(total_frames - 1, end_frame + padding)))
    return fps, windows


def _window_description(role, start_frame, end_frame, fps):
    return (f'{role}: source frames {start_frame}-{end_frame} inclusive, '
            f'source time [{start_frame / fps:.3f}, {(end_frame + 1) / fps:.3f}) seconds. '
            'The following media starts at local time 0. '
            + ('Label this interval only.' if role == 'TARGET SEGMENT' else 'Context only; do not label this interval.'))


def _get_fps(cap) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        return 25.0
    return fps


def _resize_even(frame: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    new_width = max(2, int(round(width * scale)) // 2 * 2)
    new_height = max(2, int(round(height * scale)) // 2 * 2)
    if (new_width, new_height) == (width, height):
        return frame
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _make_clip(video_path: str | Path, start_frame: int, end_frame: int) -> bytes:
    """Make an MP4 from consecutive sampled frames in the requested interval.

    A very long segment is cropped to a contiguous 20-second middle excerpt;
    no time-lapse spanning the whole segment is used.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_path = None
    writer = None
    try:
        fps = _get_fps(cap)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and start_frame >= total_frames:
            raise ValueError("Segment starts beyond the end of the video")
        if total_frames > 0 and end_frame >= total_frames:
            raise ValueError("Segment ends beyond the end of the video")

        # All included frames remain chronologically consecutive (apart from
        # uniform downsampling to reduce the size of the MP4).
        cropped_start, cropped_end = _clip_bounds(start_frame, end_frame, fps)
        if (cropped_start, cropped_end) != (start_frame, end_frame):
            print(f"Label Suggester: long segment, analyzing middle frames {cropped_start}-{cropped_end}.")

        stride = max(1, math.ceil(fps / _ENCODE_FPS))
        output_fps = fps / stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, cropped_start)
        temp = tempfile.NamedTemporaryFile(prefix="ravee_label_", suffix=".mp4", delete=False)
        output_path = Path(temp.name)
        temp.close()

        written = 0
        for offset in range(cropped_end - cropped_start + 1):
            if offset % stride == 0:
                success, frame = cap.read()
                if not success:
                    raise RuntimeError('Video decoding stopped before the requested clip ended')
                frame = _resize_even(frame, _MAX_IMAGE_EDGE)
                if writer is None:
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV MP4 encoder (mp4v) is unavailable")
                writer.write(frame)
                written += 1
            else:
                if not cap.grab():
                    raise RuntimeError('Video decoding stopped before the requested clip ended')

        if writer is not None:
            writer.release()
            writer = None
        if written < 2:
            raise RuntimeError("Too few frames to encode a usable video clip")
        clip_bytes = output_path.read_bytes()
        if not clip_bytes:
            raise RuntimeError("OpenCV wrote an empty video clip")
        if len(clip_bytes) > _MAX_INLINE_BYTES:
            raise RuntimeError("MP4 clip is too large for a safe inline request")
        print(f"Label Suggester: video clip, {written} frames at {output_fps:.2f} FPS, {len(clip_bytes)/1048576:.2f} MiB.")
        return clip_bytes
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass


def _sample_frames(video_path: str | Path, start_frame: int, end_frame: int, sample_count: int):
    """Old frame-sampling approach, retained for fallback and A/B comparison."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = _get_fps(cap)
        count = min(sample_count, end_frame - start_frame + 1)
        indices = sorted(set(np.linspace(start_frame, end_frame, count, dtype=int)))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _resize_even(frame, 640)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                frames.append((int(idx), idx / fps, jpeg.tobytes()))
        if not frames:
            raise RuntimeError("Could not read any still frames from the segment")
        return frames
    finally:
        cap.release()


def _clean_label(response) -> str:
    try:
        result = json.loads(response.text or '')
        if not isinstance(result, dict):
            raise ValueError('Expected a JSON object')
        label, evidence = result['label'], result['evidence']
        uncertainty, visible, bow = result['uncertainty'], result['target_visible'], result['bow_action']
        if (not isinstance(label, str) or not isinstance(evidence, str)
                or type(visible) is not bool or uncertainty not in ('low', 'medium', 'high')
                or bow not in ('tie_new', 'tighten_existing', 'adjust_existing', 'not_applicable', 'unclear')):
            raise ValueError('Invalid evidence/uncertainty fields')
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError('Gemini returned an invalid evidence-based activity response') from error
    if not visible or uncertainty == 'high' or not evidence.strip() or bow == 'unclear':
        return 'Unclear activity'
    if label.strip().lower() == 'unclear activity':
        return 'Unclear activity'
    canonical = {'tie_new': 'Tie ribbon bow', 'tighten_existing': 'Tighten ribbon bow',
                 'adjust_existing': 'Adjust ribbon bow'}
    if bow in canonical:
        return canonical[bow]
    label = ' '.join(label.split()).strip(' .')
    if not 2 <= len(label.split()) <= 6 or len(label) > 100:
        raise RuntimeError('Gemini returned an activity label outside the 2-to-6-word limit')
    return label[0].upper() + label[1:].lower()


def suggest_activity(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    sample_count: int = 6,
) -> str:
    """Return an unverified suggested activity label for [start_frame,end_frame].

    The GUI calls this in a worker thread; it should NOT be called directly
    from Tkinter's event loop. 'Use suggestion' and 'Commit & Next' remain
    manual. This function does not change GUI state or LABEL.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    start_frame, end_frame = int(start_frame), int(end_frame)
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError(f"Invalid frame range: {start_frame}-{end_frame}")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    # Imported on demand: RAVEE stays usable for manual labeling if Gemini
    # dependencies are missing or the API is unavailable.
    from google import genai
    from google.genai import types

    mode = os.getenv("RAVEE_SUGGEST_MODE", "video").strip().lower()
    if mode not in ("video", "frames"):
        raise ValueError("RAVEE_SUGGEST_MODE must be 'video' or 'frames'")

    fps, windows = _context_windows(video_path, start_frame, end_frame)
    prompt = (_PROMPT + f' Original target: source frames {start_frame}-{end_frame} inclusive. '
              'Media windows are supplied chronologically with separate roles.')
    target_bounds = next((start, end) for role, start, end in windows if role == 'TARGET SEGMENT')
    if target_bounds != (start_frame, end_frame):
        prompt += (' Only a middle excerpt of the long target is supplied. Do not infer '
                   'actions in unseen target portions; return Unclear activity if this '
                   'is insufficient to identify the segment activity.')
    parts = [types.Part(text=prompt)]
    total_bytes = 0
    for role, start, end in windows:
        parts.append(types.Part(text=_window_description(role, start, end, fps)))
        video_data = None
        if mode == 'video':
            try:
                video_data = _make_clip(video_path, start, end)
                if total_bytes + len(video_data) > _MAX_INLINE_BYTES:
                    raise RuntimeError('Combined context and target clips exceed inline size limit')
            except (OSError, RuntimeError) as exc:
                video_data = None
                warnings.warn(f'{role} clip encoding failed; using still-frame fallback: {exc}')
        if video_data is not None:
            total_bytes += len(video_data)
            parts.append(types.Part(
                inline_data=types.Blob(data=video_data, mime_type='video/mp4'),
                video_metadata=types.VideoMetadata(fps=_GEMINI_SAMPLE_FPS),
            ))
        else:
            count = sample_count if role == 'TARGET SEGMENT' else min(sample_count, 3)
            for index, seconds, jpeg in _sample_frames(video_path, start, end, count):
                total_bytes += len(jpeg)
                if total_bytes > _MAX_INLINE_BYTES:
                    raise RuntimeError('Combined media exceed inline size limit')
                parts.append(types.Part(text=f'{role} frame {index} at source time {seconds:.3f}s'))
                parts.append(types.Part.from_bytes(data=jpeg, mime_type='image/jpeg'))
    contents = types.Content(parts=parts)

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    with genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000)) as client:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=512,
                                               response_mime_type='application/json'),
        )
    return _clean_label(response)
