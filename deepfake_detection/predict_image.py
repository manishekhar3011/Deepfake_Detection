"""
image_preprocessing.py
----------------------

Preprocessing for the Deepfake Image Detection Model.

IMPORTANT:
This preprocessing is kept exactly the same as predict_image.py
and the training inference pipeline.

Pipeline:
PIL Image
    ->
Convert RGB
    ->
Resize (224, 224)
    ->
ToTensor
    ->
ImageNet Normalize
    ->
Add batch dimension

Output:
torch.Tensor with shape:
(1, 3, 224, 224)
"""

import torch
from torchvision import transforms


# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = 224


# ==============================
# IMAGE TRANSFORM
# ==============================

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# PREPROCESS IMAGE
# ==============================

def preprocess_image(pil_image, use_face_detection=False):
    """
    Preprocess an uploaded PIL image for the EfficientNet-B0 model.

    Args:
        pil_image:
            PIL.Image object.

        use_face_detection:
            Kept only for compatibility with existing app.py.
            Face detection is NOT used because the trained model
            was tested using the complete image.

    Returns:
        tensor:
            torch.Tensor with shape (1, 3, 224, 224)

        face_found:
            Always False because face cropping is disabled.
    """

    # Convert image to RGB
    image = pil_image.convert("RGB")

    # Apply EXACT same transform as predict_image.py
    tensor = _transform(image)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Face detection disabled
    face_found = False

    return tensor, face_found