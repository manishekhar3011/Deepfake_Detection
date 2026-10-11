"""
image_preprocessing.py
------------------------
Converts a raw uploaded image into a normalized tensor ready for
ImageDeepfakeModel.

Pipeline:
    PIL/np image -> BGR (for OpenCV face detection) -> crop face
    (fallback to full image if no face found) -> resize 224x224
    -> normalize with ImageNet mean/std -> torch tensor (1, 3, 224, 224)
"""

import numpy as np
import cv2
import torch
from torchvision import transforms

from utils.face_detection import crop_largest_face

IMAGE_SIZE = 224

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(pil_image, use_face_detection: bool = True):
    """
    Args:
        pil_image: PIL.Image (RGB) as returned by st.file_uploader + Image.open
        use_face_detection: whether to attempt face cropping first.

    Returns:
        (tensor, face_found: bool)
        tensor: torch.Tensor of shape (1, 3, 224, 224), ready for the model.
    """
    rgb_array = np.array(pil_image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    face_found = False
    working_image = bgr_array

    if use_face_detection:
        face_crop = crop_largest_face(bgr_array)
        if face_crop is not None:
            working_image = face_crop
            face_found = True

    # Convert back to RGB for the torchvision transform pipeline
    rgb_for_model = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb_for_model).unsqueeze(0)  # add batch dimension

    return tensor, face_found
