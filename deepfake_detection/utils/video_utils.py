"""
video_utils.py
---------------
Helper functions for reading video files efficiently with OpenCV and
sampling frames WITHOUT loading the entire video into memory at once.
"""

import cv2
import numpy as np


def get_video_info(video_path: str):
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def extract_sampled_frames(video_path: str, max_frames: int = 30):
    """
    Extract up to `max_frames` frames evenly spaced across the video
    duration. This avoids decoding every single frame into memory,
    which matters for long videos.

    Args:
        video_path: path to the video file on disk.
        max_frames: maximum number of frames to sample.

    Returns:
        List of frames as BGR np.ndarrays (may be shorter than
        max_frames if the video has fewer frames or is unreadable).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback: some containers don't report frame count reliably.
        # Just read sequentially until exhausted, keeping every Nth frame.
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % 5 == 0:  # crude sampling fallback
                frames.append(frame)
            idx += 1
            if len(frames) >= max_frames:
                break
        cap.release()
        return frames

    # Evenly spaced frame indices across the whole video
    sample_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames
