
import argparse
import cv2
import torch
import numpy as np

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"
VIDEO_PATH = "DemoFiles/scan_video_reduced.avi"

CANDIDATES = {
    "prepare": "a worker preparing materials and tools for a construction task",
    "edge crafting": "a worker crafting, cutting or shaping the edge of a component",
    "balancing": "a worker balancing and aligning a structure or component",
    "roof crafting": "a worker constructing and fitting a roof",
    "roof finishing": "a worker finishing, smoothing or adjusting a roof",
    "finishing": "a worker completing the final finishing touches",
    "inspect": "a worker inspecting and checking the quality of a component",
    "bench pressing": "a person doing a bench press exercise",
    "weight lifting": "a person lifting weights in a gym",
}


def get_frames(start, end, count=8):
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start < 0 or end >= total or start >= end:
        raise ValueError(f"Invalid range {start}-{end}; video has {total} frames")

    frame_ids = np.linspace(start, end, count, dtype=int)
    full_images = []
    crop_images = []

    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Crop central lower area containing the worker and object
        x1, x2 = int(w * 0.25), int(w * 0.75)
        y1, y2 = int(h * 0.35), h

        full_images.append(Image.fromarray(frame))
        crop_images.append(Image.fromarray(frame[y1:y2, x1:x2]))

    cap.release()

    if not full_images:
        raise RuntimeError("No frames were extracted")

    return frame_ids, full_images, crop_images


def predict(model, processor, images, prompts, device):
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    # Average similarity across sampled frames
    mean_logits = outputs.logits_per_image.mean(dim=0)

    # Relative scores among candidate descriptions
    probabilities = torch.softmax(mean_logits, dim=0)

    return probabilities.cpu().numpy()


def show_results(title, scores, labels):
    print(f"\n{title}")

    ranking = np.argsort(scores)[::-1]

    for rank, idx in enumerate(ranking[:5], start=1):
        print(f"{rank}. {labels[idx]}: {scores[idx] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Loading CLIP from local cache...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=True
    )
    model = CLIPModel.from_pretrained(
        MODEL_ID,
        local_files_only=True
    ).to(device)
    model.eval()

    labels = list(CANDIDATES.keys())
    prompts = list(CANDIDATES.values())

    frame_ids, full_images, crop_images = get_frames(
        args.start, args.end
    )

    print("Sampled frames:", frame_ids.tolist())

    full_scores = predict(
        model, processor, full_images, prompts, device
    )
    crop_scores = predict(
        model, processor, crop_images, prompts, device
    )

    show_results("FULL FRAME", full_scores, labels)
    show_results("CROPPED REGION", crop_scores, labels)

    print("\nScores are relative, not calibrated confidence values.")


if __name__ == "__main__":
    main()
