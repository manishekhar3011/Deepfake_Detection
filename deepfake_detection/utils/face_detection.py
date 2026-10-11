"""
face_detection.py
------------------
Utility functions for detecting and cropping faces from images/frames.

Uses OpenCV's built-in Haar Cascade classifier so that the project has
ZERO extra heavyweight dependencies (no dlib / mediapipe required).
This keeps the project runnable on a normal college laptop with only
`opencv-python` installed.

If you want higher accuracy face detection later, you can swap this
implementation for a DNN-based detector (e.g. OpenCV's DNN face
detector or MTCNN) without changing the function signatures used by
the rest of the project.
"""

import cv2
import numpy as np

# Load the pretrained Haar Cascade face detector that ships with OpenCV.
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def detect_faces(image_bgr: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5):
    """
    Detect faces in a BGR image (as read by cv2.imread / cv2.VideoCapture).

    Args:
        image_bgr: np.ndarray, image in BGR format.
        scale_factor: how much the image size is reduced at each scale.
        min_neighbors: how many neighbors each candidate rectangle should
            retain to be considered a valid face.

    Returns:
        List of bounding boxes: [(x, y, w, h), ...]
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(40, 40)
    )
    return list(faces)


def crop_largest_face(image_bgr: np.ndarray, margin: float = 0.25):
    """
    Detect all faces in the image and return a crop of the LARGEST face
    (most likely the primary subject), with an extra margin around it.

    Args:
        image_bgr: input image in BGR format.
        margin: fraction of the face box size to pad on each side.

    Returns:
        cropped_face (np.ndarray) or None if no face was detected.
    """
    faces = detect_faces(image_bgr)
    if len(faces) == 0:
        return None

    # Pick the face with the largest area (w * h)
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    pad_x = int(w * margin)
    pad_y = int(h * margin)

    h_img, w_img = image_bgr.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)

    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    return face_crop
