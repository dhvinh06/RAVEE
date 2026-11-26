"""RAVEE Label Suggester v2: suggest an action from a short moving video clip.

Drop-in replacement for the earlier, 6-frame-only label_suggester.py.
Existing gui/gui_elements.py needs NO changes.

Defaults: encode a short 480px MP4 at up to 6fps, ask Gemini to inspect
at 3fps. No persistent video file is left behind. For an alternative
baseline, set RAVEE_SUGGEST_MODE=frames to use six still frames.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np

# Keep each inline request reasonably small. Files API is intentionally not
# required: short clips avoid the prior upload/processing wait.
_MAX_INLINE_BYTES = 18 * 1024 * 1024
_MAX_DURATION_SECONDS = 20.0
_ENCODE_FPS = 6.0
_GEMINI_SAMPLE_FPS = 3.0
_MAX_IMAGE_EDGE = 480

_PROMPT = (
    "You are labeling ONE contiguous video segment for video-based process mining. "
    "Watch the changes and HAND/TOOL/OBJECT interactions over time, not just the scene. "
    "Identify the SINGLE most specific visible physical action in this segment. "
    "Respond with one concise ENGLISH activity label, 2 to 6 words, ideally an "
    "action verb plus the object being manipulated. "
    "Different actions within the same overall process must receive different labels. "
    "Do NOT describe the overall project, scene, location, or generic goal. "
    "Avoid generic labels such as 'assembling furniture', 'building furniture', "
    "'working on furniture', or 'preparing', unless that is genuinely the only "
    "observable detail; in that case use exactly 'Unclear activity'. "
    "Only name movements you can actually see. "
    "If nothing distinct is visible, or several equally important actions occur "
    "without one dominant action, answer 'Unclear activity'. "
    "Respond with ONLY the label, without quotes, explanations, numbering or punctuation."
)


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
        max_frames = max(1, int(_MAX_DURATION_SECONDS * fps))
        if end_frame - start_frame + 1 > max_frames:
            cropped_start = start_frame + (end_frame - start_frame + 1 - max_frames) // 2
            cropped_end = cropped_start + max_frames - 1
            print(f"Label Suggester: long segment, analyzing middle frames {cropped_start}-{cropped_end}.")
        else:
            cropped_start, cropped_end = start_frame, end_frame

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
                    break
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
                    break

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
    result = (response.text or "").strip()
    if not result:
        raise RuntimeError("Gemini returned an empty activity label")
    label = result.splitlines()[0].strip().strip("`\"' ").strip(".:- ")
    if not label:
        raise RuntimeError("Gemini returned an invalid activity label")
    return label[:100]


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

    video_data = None
    if mode == "video":
        try:
            video_data = _make_clip(video_path, start_frame, end_frame)
        except (OSError, RuntimeError) as exc:
            warnings.warn(f"Clip encoding failed; using still-frame fallback: {exc}")

    if video_data is not None:
        media_part = types.Part(
            inline_data=types.Blob(data=video_data, mime_type="video/mp4"),
            video_metadata=types.VideoMetadata(fps=_GEMINI_SAMPLE_FPS),
        )
        contents = types.Content(parts=[types.Part(text=_PROMPT), media_part])
    else:
        frames = _sample_frames(video_path, start_frame, end_frame, sample_count)
        contents = [_PROMPT]
        for index, seconds, jpeg in frames:
            contents.append(f"Frame {index} at {seconds:.2f}s")
            contents.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        print(f"Label Suggester: still-frame mode, {len(frames)} frames.")

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=128),
    )
    return _clean_label(response)
