"""
audio_model.py
---------------
Audio Deepfake Detection Model.

Architecture:
    Mel Spectrogram (1 x 128 x 128, treated as a single-channel "image")
        -> Small custom CNN (Conv -> BN -> ReLU -> Pool, x3)
        -> Fully connected classifier
        -> REAL / DEEPFAKE

Audio deepfakes (voice cloning / synthesis artifacts) show up as
distinctive texture patterns in the Mel Spectrogram, so a small CNN
trained directly on spectrogram "images" works well and is cheap to
train from scratch (no large pretrained audio backbone is required).
"""

import torch
import torch.nn as nn


class AudioDeepfakeModel(nn.Module):
    """Compact CNN classifier operating on log-Mel spectrograms."""

    def __init__(self, n_mels: int = 128, time_frames: int = 128):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (16, n_mels/2, time/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> (32, n_mels/4, time/4)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (64, 4, 4) regardless of input size
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        """x: (batch, 1, n_mels, time_frames) -> logits (batch, 2)"""
        x = self.conv_block(x)
        return self.classifier(x)


def load_audio_model(weights_path: str = None, device: str = "cpu"):
    """Convenience loader used by the Streamlit app. See load_image_model."""
    model = AudioDeepfakeModel()
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
