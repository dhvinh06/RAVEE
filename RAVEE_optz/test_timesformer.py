
import argparse
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForVideoClassification,
)

MODEL_ID = "facebook/timesformer-base-finetuned-ssv2"
VIDEO_PATH = "DemoFiles/scan_video_reduced.avi"


def extract_frames(start, end, num_frames=8):
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start < 0 or end >= total_frames or start >= end:
        raise ValueError(
            f"Invalid range {start}-{end}. "
            f"Video has {total_frames} frames."
        )

    frame_ids = np.linspace(
        start, end, num_frames, dtype=int
    )

    frames = []

    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()

    if len(frames) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames, got {len(frames)}"
        )

    return frame_ids, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading TimeSformer...")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForVideoClassification.from_pretrained(
        MODEL_ID
    ).to(device)

    model.eval()

    frame_ids, frames = extract_frames(
        args.start, args.end
    )

    print("Sampled frames:", frame_ids.tolist())
    print("Running video classification...")

    inputs = processor(
        images=frames,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits[0], dim=-1
    )

    top_values, top_indices = torch.topk(
        probabilities, k=5
    )

    id2label = model.config.id2label

    print("\nTop 5 predicted actions:")

    for rank, (score, idx) in enumerate(
            zip(top_values.tolist(), top_indices.tolist()),
            start=1
    ):
        label = id2label.get(
            idx,
            id2label.get(str(idx), str(idx))
        )

        print(f"{rank}. {label}: {score * 100:.2f}%")

    if device == "cuda":
        memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory: {memory:.2f} GB")


if __name__ == "__main__":
    main()
