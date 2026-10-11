"""
image_model.py
---------------
Image Deepfake Detection Model.

Architecture:
    Input face image (224x224x3)
        -> Pretrained EfficientNet-B0 (ImageNet weights, transfer learning)
        -> Global pooling (built into EfficientNet's classifier head)
        -> Custom classifier head (Dropout -> Linear -> 2 classes)
        -> REAL / DEEPFAKE

We use EfficientNet-B0 because it offers a strong accuracy/compute
trade-off and is light enough to run on a normal laptop CPU.
"""

import torch
import torch.nn as nn
from torchvision import models


class ImageDeepfakeModel(nn.Module):
    """EfficientNet-B0 backbone with a custom binary classification head."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Replace the final classifier layer (1000-class ImageNet head)
        # with a small binary head: REAL (0) vs DEEPFAKE (1).
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )

        self.backbone = backbone

        if freeze_backbone:
            # Freeze feature-extraction layers; only train the new head.
            # This drastically speeds up training on a laptop CPU/GPU.
            for name, param in self.backbone.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x):
        """x: (batch, 3, 224, 224) normalized tensor -> logits (batch, 2)"""
        return self.backbone(x)

    def unfreeze_backbone(self):
        """Call this for fine-tuning after the head has converged."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def load_image_model(weights_path: str = None, device: str = "cpu"):
    """
    Convenience loader used by the Streamlit app.

    Args:
        weights_path: path to a trained .pth state_dict file. If None
            or the file doesn't exist, returns an untrained model
            (ImageNet-pretrained backbone only) — used for DEMO MODE.
        device: 'cpu' or 'cuda'.

    Returns:
        (model, is_trained: bool)
    """
    model = ImageDeepfakeModel(pretrained=True, freeze_backbone=True)
    is_trained = False

    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            is_trained = True
        except FileNotFoundError:
            is_trained = False
        except Exception:
            is_trained = False

    model.to(device)
    model.eval()
    return model, is_trained
