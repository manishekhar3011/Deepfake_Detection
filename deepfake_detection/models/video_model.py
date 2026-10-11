"""
video_model.py
---------------
Video Deepfake Detection Model.

Architecture:
    Sequence of face crops (T frames, 224x224x3 each)
        -> Shared MobileNetV2 CNN backbone (per-frame feature extractor)
        -> Sequence of feature vectors (T, 1280)
        -> LSTM (temporal modeling across frames)
        -> Dense classifier
        -> REAL / DEEPFAKE (per-video), plus per-frame probabilities

We use MobileNetV2 as the per-frame CNN because it is one of the
lightest pretrained backbones available in torchvision, which matters
since it has to run once per sampled frame.
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNFeatureExtractor(nn.Module):
    """MobileNetV2 backbone (feature extractor only, no classifier head)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v2(weights=weights)
        self.features = mobilenet.features  # conv feature extractor
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280  # MobileNetV2's final channel depth

        for param in self.features.parameters():
            param.requires_grad = False  # frozen for transfer learning

    def forward(self, x):
        """x: (batch, 3, 224, 224) -> (batch, 1280)"""
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class VideoDeepfakeModel(nn.Module):
    """CNN (per-frame) -> LSTM (temporal) -> Dense classifier."""

    def __init__(self, hidden_size: int = 256, num_layers: int = 1, pretrained: bool = True):
        super().__init__()
        self.cnn = CNNFeatureExtractor(pretrained=pretrained)
        self.lstm = nn.LSTM(
            input_size=self.cnn.out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, frame_sequence):
        """
        Args:
            frame_sequence: (batch, T, 3, 224, 224) tensor of face crops.

        Returns:
            video_logits: (batch, 2) — final aggregated prediction.
            frame_logits: (batch, T, 2) — per-frame predictions, useful
                for plotting the fake-probability-per-frame graph.
        """
        batch, T, C, H, W = frame_sequence.shape

        # Run CNN on every frame independently (reshape T into batch dim)
        frames_flat = frame_sequence.view(batch * T, C, H, W)
        features_flat = self.cnn(frames_flat)               # (batch*T, 1280)
        features_seq = features_flat.view(batch, T, -1)      # (batch, T, 1280)

        lstm_out, _ = self.lstm(features_seq)                # (batch, T, hidden)

        # Per-frame logits (share the same classifier head across time steps)
        frame_logits = self.classifier(lstm_out)              # (batch, T, 2)

        # Final video-level prediction: use the last time step's hidden state
        video_logits = self.classifier(lstm_out[:, -1, :])    # (batch, 2)

        return video_logits, frame_logits


def load_video_model(weights_path: str = None, device: str = "cpu"):
    """Convenience loader used by the Streamlit app. See load_image_model."""
    model = VideoDeepfakeModel(pretrained=True)
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
