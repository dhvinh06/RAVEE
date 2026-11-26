"""Prototype: open-vocabulary activity labeling with Gemini video understanding.

Run from the RAVEE project root:
    python .\\test_gemini_labels_retry.py --allow-upload
    python .\\test_gemini_labels_retry.py --reuse-file files/<id>

Set GEMINI_API_KEY in the environment before running. The script uploads the
video temporarily through Gemini Files API, asks the model to describe each
FINCH segment using the complete video as context, then saves the raw response.
This is a standalone Interactions/Files API research prototype with fixed demo
boundaries and its own --model option. Production Suggest All lives in
batch_label_suggester.py and uses GenerateContent, current GUI boundaries,
validated structured responses and a separate inline-video batch cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
from google import genai
from gemini_runtime import (resolve_credential, create_client, get_project_file,
    upload_project_file, call_with_retry, classify_error, redact_text)

DEFAULT_VIDEO = Path("DemoFiles") / "scan_video_reduced.avi"
DEFAULT_MODEL = "gemini-3.8-flash"

# Segment boundaries from the user's current FINCH/manual-labeling run.
# These are inclusive frame indices. They are for this demo only; later the
# segment list should be read from RAVEE's clustering output automatically.
DEMO_SEGMENTS = [
    (0, 29), (30, 62), (63, 87), (88, 170), (171, 224),
    (225, 252), (253, 287), (288, 313), (314, 375), (376, 449),
    (450, 489), (490, 514), (515, 544), (545, 573), (574, 584),
    (585, 602), (603, 649), (650, 673), (674, 699), (700, 719),
    (720, 755), (756, 773), (774, 834),
]


def get_video_info(video_path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"Invalid video metadata: fps={fps}, frames={frame_count}"
        )
    print(f"Video: {video_path.resolve()}")
    print(f"Duration: {frame_count / fps:.2f}s | FPS: {fps:.3f} | "
          f"Frames: {frame_count} | Resolution: {width}x{height}")
    return fps, frame_count, width


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    minutes, remaining_ms = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(remaining_ms, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def make_segment_description(fps: float, frame_count: int) -> str:
    lines = []
    for idx, (start_frame, end_frame) in enumerate(DEMO_SEGMENTS, start=1):
        if end_frame >= frame_count:
            raise ValueError(
                f"Demo segment {idx} ends at frame {end_frame}, "
                f"but video has only {frame_count} frames. "
                "Update DEMO_SEGMENTS for this video."
            )
        start_s = start_frame / fps
        # End frame is inclusive; use the next frame boundary as end time.
        end_s = (end_frame + 1) / fps
        lines.append(
            f"{idx}. frames {start_frame}-{end_frame} "
            f"(time {format_timestamp(start_s)} to {format_timestamp(end_s)})"
        )
    return "\n".join(lines)


def build_prompt(fps: float, frame_count: int) -> str:
    segments = make_segment_description(fps, frame_count)
    return f"""You are an expert in observational video understanding and business process activity labeling.

Analyze the attached video as an UNSEEN video. The goal is to generate open-vocabulary activity labels, not to choose from a predefined class list. Do not assume this is a furniture assembly video until the visual evidence supports that interpretation.

First, infer the overall scene, the main objects/components, the tools (if visible), and the apparent high-level workflow. Be explicit when an object's identity or the worker's intention is uncertain.

Then analyze each of the following temporal segments. Use the entire video as context, but describe what visibly happens within the specified segment:
{segments}

For EACH segment:
- Describe the observable action using a concise sentence.
- Generate a short task-level activity label using an action verb + object, e.g. "Align side panel" or "Secure roof panel". These are examples of label style only, not candidate labels.
- Prefer meaningful work activities over body postures such as "look down", "bend down", or "stand up", unless the posture itself is the meaningful activity.
- Do not force different labels when multiple segments show the same continuing activity. Use consistent wording for repeated activities.
- If a segment contains a clear transition between two distinct work activities, mention both in the description and choose the predominant task-level activity as its main label.
- Do not infer hidden intent. If the action/object cannot be identified confidently, use a cautious generic label and mark uncertainty.
- Provide visible evidence supporting each label and an uncertainty value of low, medium, or high. This is a qualitative estimate, not a calibrated probability.

Return ONLY valid JSON, with this structure:
{{
  "video_summary": "...",
  "objects_and_tools": ["..."],
  "segments": [
    {{
      "segment_id": 1,
      "start_frame": 0,
      "end_frame": 29,
      "activity_label": "...",
      "action_description": "...",
      "visual_evidence": "...",
      "uncertainty": "low|medium|high"
    }}
  ],
  "repeated_activities": [
    {{"activity_label": "...", "segment_ids": [1, 2], "reason": "..."}}
  ]
}}

There must be exactly {len(DEMO_SEGMENTS)} segment records, in the same order as the supplied list. Preserve frame indices exactly. Do not include Markdown fences or extra commentary."""


def state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return ""
    return str(getattr(state, "name", state)).upper().split(".")[-1]


def upload_and_wait(client: genai.Client, video_path: Path, poll_seconds: int,
                    timeout_seconds: int, *, authorized=False) -> Any:
    print("Uploading video to Gemini Files API...")
    uploaded = upload_project_file(client, video_path, authorized=authorized)
    if not getattr(uploaded, "name", None) or not getattr(uploaded, "uri", None):
        raise RuntimeError("Gemini upload did not return a usable file name/URI.")

    print(f"Uploaded file: {uploaded.name}")
    deadline = time.monotonic() + timeout_seconds

    while True:
        state = state_name(uploaded)
        if state == "ACTIVE":
            print("Video processing: ACTIVE")
            return uploaded
        if state == "FAILED":
            raise RuntimeError("Gemini failed to process the uploaded video.")
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for video processing. Last state: {state or 'unknown'}"
            )
        print(f"Video processing state: {state or 'unknown'}; checking again...")
        time.sleep(poll_seconds)
        uploaded = get_project_file(client, uploaded.name)


def get_existing_upload_and_wait(client: genai.Client, file_name: str,
                                 poll_seconds: int, timeout_seconds: int) -> Any:
    """Reuse an existing Gemini Files API upload instead of uploading again."""
    print(f"Reusing existing Gemini file: {file_name}")
    uploaded = get_project_file(client, file_name)
    deadline = time.monotonic() + timeout_seconds

    while True:
        state = state_name(uploaded)
        if state == "ACTIVE":
            print("Existing video processing: ACTIVE")
            return uploaded
        if state == "FAILED":
            raise RuntimeError(
                f"Gemini file {file_name} is in FAILED state. Upload the video again."
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for existing file. Last state: {state or 'unknown'}"
            )
        print(f"Existing file state: {state or 'unknown'}; checking again...")
        time.sleep(poll_seconds)
        uploaded = get_project_file(client, file_name)


def is_retryable_service_error(exc: Exception) -> bool:
    return classify_error(exc).kind == 'service_unavailable'


def create_interaction_with_retry(client: genai.Client, model: str,
                                  input_parts: list[dict[str, Any]],
                                  max_retries: int) -> Any:
    """Retry transient 503 capacity errors with exponential backoff."""
    return call_with_retry(lambda: client.interactions.create(model=model, input=input_parts),
                           progress=print, attempts=max_retries + 1)


def extract_json(text: str) -> Any | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting the outermost JSON object if the model added prose.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate open-vocabulary activity labels for FINCH segments."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO,
                        help="Video path (default: DemoFiles/scan_video_reduced.avi)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Static video sampling FPS (default: 2.0)")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--reuse-file",
        help="Reuse an existing Gemini Files API name, e.g. files/abc123, "
             "instead of uploading the video again.",
    )
    parser.add_argument(
        "--retries", type=int, default=4,
        help="Number of retries for temporary Gemini 503 capacity errors (default: 4).",
    )
    parser.add_argument('--allow-upload', action='store_true',
                        help='Explicitly authorize a new video upload to the selected project')
    args = parser.parse_args()

    try:
        credential = resolve_credential()
        if not args.reuse_file and not args.allow_upload:
            raise RuntimeError('A new upload requires --allow-upload; --reuse-file verifies ownership in the selected project')
    except RuntimeError as error:
        print('ERROR: ' + redact_text(error), file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("ERROR: --fps must be greater than 0.", file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"ERROR: video file not found: {args.video.resolve()}", file=sys.stderr)
        return 2

    fps, frame_count, _ = get_video_info(args.video)
    prompt = build_prompt(fps, frame_count)

    client = None
    uploaded = None
    try:
        client = create_client(credential)
        if args.reuse_file:
            uploaded = get_existing_upload_and_wait(
                client,
                args.reuse_file,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            uploaded = upload_and_wait(
                client,
                args.video,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
                authorized=args.allow_upload,
            )

        print("Calling model: " + redact_text(args.model))
        print(f"Video sampling: static at {args.fps} FPS")
        input_parts = [
            {
                "type": "video",
                "uri": uploaded.uri,
                "mime_type": uploaded.mime_type or "video/avi",
                "processing": {
                    "type": "static",
                    "fps": args.fps,
                },
            },
            {"type": "text", "text": prompt},
        ]
        interaction = create_interaction_with_retry(
            client,
            model=args.model,
            input_parts=input_parts,
            max_retries=max(0, args.retries),
        )

        output_text = getattr(interaction, "output_text", None)
        if not output_text:
            raise RuntimeError(
                "Gemini returned no output_text. Inspect the API response or SDK version."
            )

        output_text = redact_text(output_text, limit=None)
        result_stem = args.video.stem + '_gemini_labels'
        if any(args.video.with_name(result_stem + suffix).exists() for suffix in ('.txt', '.json')):
            result_stem += '_' + uuid.uuid4().hex
        output_path = args.video.with_name(result_stem + '.txt')
        output_path.write_text(output_text, encoding="utf-8")
        print(f"\nRaw response saved to: {output_path.resolve()}")

        parsed = extract_json(output_text)
        if parsed is not None:
            json_path = args.video.with_name(result_stem + '.json')
            json_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Parsed JSON saved to: {json_path.resolve()}")
            records = parsed.get("segments", []) if isinstance(parsed, dict) else []
            print(f"Segment records returned: {len(records)} / {len(DEMO_SEGMENTS)}")
        else:
            print("WARNING: Response was not valid JSON; inspect the raw TXT file.")

        usage = getattr(interaction, "usage", None)
        if usage is not None:
            print("Usage metadata:", usage)

        print("\nGemini response:\n")
        print(output_text)
        return 0

    except Exception as exc:
        print("\nERROR: " + classify_error(exc).message, file=sys.stderr)
        print(
            "The uploaded file, if any, may remain in Gemini Files storage "
            "temporarily (Files API uploads expire after 48 hours).",
            file=sys.stderr,
        )
        return 1

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                print('Gemini client cleanup failed; saved results remain available.', file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
