import os

import cv2
import numpy as np

from app.services.video_service import extract_frames


def _make_video(path, frames=25, size=64):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    for i in range(frames):
        vw.write(np.full((size, size, 3), i * 10 % 255, dtype=np.uint8))
    vw.release()


def test_extract_frames_every_n(tmp_path):
    video = str(tmp_path / "v.mp4")
    _make_video(video, frames=25)
    out = str(tmp_path / "frames")
    paths = extract_frames(video, out, every_n=10)
    assert [os.path.basename(p) for p in paths] == [
        "frame_000000.jpg", "frame_000010.jpg", "frame_000020.jpg"]
    assert all(os.path.isfile(p) for p in paths)
