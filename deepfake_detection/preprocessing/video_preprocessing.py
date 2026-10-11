"""
video_preprocessing.py
------------------------
Converts a raw uploaded video file into a normalized tensor sequence
ready for VideoDeepfakeModel.

Pipeline:
    video file -> sample N frames (utils.video_utils) -> detect + crop
    face per frame (utils.face_detection) -> resize 224x224 -> normalize
    -> stack into a (1, T, 3, 224, 224) tensor
"""

import cv2
import torch
from torchvision import transforms

from utils.video_utils import extract_sampled_frames
from utils.face_detection import crop_largest_face

FRAME_SIZE = 224

_frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_video(video_path: str, max_frames: int = 30):
    """
    Args:
        video_path: path to a video file saved temporarily on disk.
        max_frames: number of frames to sample from the video.

    Returns:
        dict with:
            tensor: torch.Tensor (1, T, 3, 224, 224) or None if no usable frames.
            frames_analyzed: int, number of frames actually sampled.
            faces_detected: int, number of those frames where a face was found.
    """
    raw_frames = extract_sampled_frames(video_path, max_frames=max_frames)

    processed = []
    faces_detected = 0

    for frame_bgr in raw_frames:
        face_crop = crop_largest_face(frame_bgr)
        if face_crop is not None:
            faces_detected += 1
            frame_to_use = face_crop
        else:
            # Fallback: use the full frame if no face detected, so the
            # pipeline degrades gracefully instead of dropping the frame.
            frame_to_use = frame_bgr

        rgb = cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2RGB)
        tensor_frame = _frame_transform(rgb)
        processed.append(tensor_frame)

    if len(processed) == 0:
        return {"tensor": None, "frames_analyzed": 0, "faces_detected": 0}

    sequence_tensor = torch.stack(processed, dim=0).unsqueeze(0)  # (1, T, 3, 224, 224)

    return {
        "tensor": sequence_tensor,
        "frames_analyzed": len(processed),
        "faces_detected": faces_detected,
    }
