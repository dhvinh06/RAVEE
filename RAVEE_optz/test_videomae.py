"""Run a single VideoMAE inference on one video segment.

Default test target (when this file is placed in the RAVEE project root):
    DemoFiles/scan_video_reduced.avi, frames 0..87

Usage:
    python test_videomae.py
    python test_videomae.py path/to/video.mp4 --start 100 --end 220
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification

try:
    # This is the processor class specified by the selected model card.
    from transformers import VideoMAEImageProcessor
except ImportError:
    # Compatibility fallback for Transformers releases exposing the video
    # processor under the newer class name.
    try:
        from transformers import VideoMAEVideoProcessor as VideoMAEImageProcessor
    except ImportError as exc:
        raise ImportError(
            "Could not import a VideoMAE processor. Check the installed "
            "Transformers version with: python -m pip show transformers"
        ) from exc


MODEL_ID = "MCG-NJU/videomae-small-finetuned-kinetics"
DEFAULT_VIDEO = Path(__file__).resolve().parent / "DemoFiles" / "scan_video_reduced.avi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify one video segment with VideoMAE (Kinetics-400)."
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Video path (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame (inclusive).")
    parser.add_argument("--end", type=int, default=87, help="Last frame (inclusive).")
    parser.add_argument("--top-k", type=int, default=5, help="Number of labels to print.")
    return parser.parse_args()


def sample_video_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> list[np.ndarray]:
    """Uniformly sample num_frames RGB frames from an inclusive frame range."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Frame range is invalid: require 0 <= start <= end.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Could not read the video's frame count.")
        if start_frame >= total_frames:
            raise ValueError(
                f"Start frame {start_frame} is outside the video "
                f"(0..{total_frames - 1})."
            )

        end_frame = min(end_frame, total_frames - 1)
        frame_indices = np.linspace(
            start_frame, end_frame, num=num_frames, dtype=np.int64
        )

        frames: list[np.ndarray] = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Could not read frame {int(frame_index)}.")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        print(
            f"Video: {video_path}\n"
            f"Total frames: {total_frames}\n"
            f"Sampled frames: {frame_indices.tolist()}"
        )
        return frames
    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    print(f"Loading processor: {MODEL_ID}")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)

    print(f"Loading model: {MODEL_ID}")
    # Half precision and SDPA reduce memory usage on the 4 GB GTX 1650.
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_ID,
        dtype=model_dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()

    num_frames = int(getattr(model.config, "num_frames", 16))
    frames = sample_video_frames(args.video, args.start, args.end, num_frames)

    print("Preprocessing frames...")
    inputs = processor(frames, return_tensors="pt")
    inputs = {
        key: value.to(device=device, dtype=model_dtype)
        if value.is_floating_point()
        else value.to(device)
        for key, value in inputs.items()
    }

    print("Running inference...")
    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits[0].float(), dim=-1)
        k = min(args.top_k, probabilities.numel())
        scores, indices = torch.topk(probabilities, k=k)

    id2label = model.config.id2label
    print("\nTop predictions (Kinetics-400):")
    for rank, (score, class_index) in enumerate(zip(scores.tolist(), indices.tolist()), 1):
        label = id2label.get(class_index, id2label.get(str(class_index), str(class_index)))
        print(f"{rank}. {label}: {score * 100:.2f}%")

    print(
        "\nNote: these are softmax scores over the Kinetics-400 classes, not "
        "calibrated confidence values or guaranteed process-activity labels."
    )
    if device.type == "cuda":
        print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except torch.cuda.OutOfMemoryError:
        print(
            "\nCUDA ran out of memory. Close GPU-heavy apps and retry. "
            "If it still fails, run CPU inference by temporarily setting "
            "device = torch.device('cpu') in main().",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
