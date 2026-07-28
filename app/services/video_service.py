import os

import cv2


def extract_frames(video_path, out_dir, every_n=10, progress_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    paths = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every_n == 0:
                path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(path, frame)
                paths.append(path)
                if progress_cb:
                    progress_cb(idx, total)
            idx += 1
    finally:
        cap.release()
    return paths
