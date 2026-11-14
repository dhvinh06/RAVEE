import cv2
import numpy as np
from pathlib import Path

video_path = Path("DemoFiles/scan_video_reduced.avi")
output_path = Path("DemoFiles/clip_contact_sheet.jpg")

segments = [
    (0, 87, "prepare"),
    (88, 170, "edge crafting"),
    (171, 224, "edge crafting 2"),
    (225, 313, "edge crafting 3"),
    (314, 375, "balancing"),
    (376, 449, "roof crafting"),
    (450, 719, "roof finishing"),
    (720, 834, "finishing"),
]

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

cell_w, cell_h = 320, 180
header_h, gap = 35, 8
cols = 3

sheet_w = cols * cell_w + (cols + 1) * gap
sheet_h = len(segments) * (header_h + cell_h + gap) + gap

sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

for row, (start, end, label) in enumerate(segments):
    y = gap + row * (header_h + cell_h + gap)

    cv2.putText(
        sheet, f"{label} | frames {start}-{end}",
        (gap, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 0), 2
    )

    frame_ids = np.linspace(start, end, 3, dtype=int)

    for col, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]
        scale = min(cell_w / w, cell_h / h)

        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh))

        x = gap + col * (cell_w + gap)
        cy = y + header_h

        px = x + (cell_w - nw) // 2
        py = cy + (cell_h - nh) // 2

        sheet[py:py+nh, px:px+nw] = resized

        cv2.putText(
            sheet, str(frame_id), (x + 5, cy + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 255), 2
        )

cap.release()

if not cv2.imwrite(str(output_path), sheet):
    raise RuntimeError("Cannot save contact sheet")

print(f"Saved: {output_path.resolve()}")