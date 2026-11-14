"""Run one Label Suggester prediction without launching the RAVEE GUI.

From the RAVEE project root:
    python test_label_suggester_v2.py --start 88 --end 170
"""

import argparse
import os
from pathlib import Path

from label_suggester import suggest_activity


def main():
    parser = argparse.ArgumentParser(description="Test Label Suggester on one video segment")
    parser.add_argument("--video", default=str(Path("DemoFiles") / "scan_video_reduced.avi"))
    parser.add_argument("--start", type=int, default=88)
    parser.add_argument("--end", type=int, default=170)
    parser.add_argument("--mode", choices=["video", "frames"], default="video")
    args = parser.parse_args()

    os.environ["RAVEE_SUGGEST_MODE"] = args.mode
    print("Video:", args.video)
    print("Frame range:", args.start, "-", args.end)
    print("Mode:", args.mode)
    print("Model:", os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    print("Proposed label:", suggest_activity(args.video, args.start, args.end))


if __name__ == "__main__":
    main()
