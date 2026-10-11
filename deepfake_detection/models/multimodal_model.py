"""
multimodal_model.py
---------------------
Multi-Modal Fusion Model: combines VIDEO (visual+temporal) features and
AUDIO (spectrogram) features to make a single joint REAL/DEEPFAKE
prediction, for cases where both a video and its audio track are
available (e.g. a talking-head video).

Architecture:
    Video --> CNN --> LSTM --> Visual Feature Vector (256-d)  --\
                                                                    Concatenate --> Fusion Dense Layers --> REAL/DEEPFAKE
    Audio --> Mel Spectrogram --> CNN --> Audio Feature Vector (64-d) --/

Design notes:
- This is intentionally a LIGHTWEIGHT late-fusion architecture
  (concatenation + small MLP) rather than attention-based fusion,
  because more complex fusion (e.g. cross-modal attention/transformers)
  is not feasible to train on a normal college laptop without a
  dedicated GPU. This is a documented limitation (see README).
- Each modality's sub-network can still be used independently
  (see video_model.py / audio_model.py), matching the "modular /
  each modality also works independently" requirement.
"""

import torch
import torch.nn as nn

from models.video_model import CNNFeatureExtractor
from models.audio_model import AudioDeepfakeModel


class MultiModalDeepfakeModel(nn.Module):
    def __init__(self, video_hidden_size: int = 256, audio_feat_dim: int = 64):
        super().__init__()

        # ---- Visual branch: CNN + LSTM (reuses the same backbone idea as video_model) ----
        self.video_cnn = CNNFeatureExtractor(pretrained=True)
        self.video_lstm = nn.LSTM(
            input_size=self.video_cnn.out_dim,
            hidden_size=video_hidden_size,
            batch_first=True,
        )

        # ---- Audio branch: reuse the conv_block from AudioDeepfakeModel, drop its classifier ----
        audio_backbone = AudioDeepfakeModel()
        self.audio_conv = audio_backbone.conv_block
        self.audio_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, audio_feat_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Fusion head ----
        fusion_input_dim = video_hidden_size + audio_feat_dim
        self.fusion = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, frame_sequence, mel_spectrogram):
        """
        Args:
            frame_sequence: (batch, T, 3, 224, 224) video face crops.
            mel_spectrogram: (batch, 1, n_mels, time_frames) audio input.

        Returns:
            logits: (batch, 2) joint REAL/DEEPFAKE prediction.
        """
        batch, T, C, H, W = frame_sequence.shape
        frames_flat = frame_sequence.view(batch * T, C, H, W)
        video_feats = self.video_cnn(frames_flat).view(batch, T, -1)
        _, (h_n, _) = self.video_lstm(video_feats)
        video_vector = h_n[-1]  # (batch, video_hidden_size)

        audio_conv_out = self.audio_conv(mel_spectrogram)
        audio_vector = self.audio_proj(audio_conv_out)  # (batch, audio_feat_dim)

        fused = torch.cat([video_vector, audio_vector], dim=1)
        return self.fusion(fused)


def load_multimodal_model(weights_path: str = None, device: str = "cpu"):
    """Convenience loader used by the Streamlit app. See load_image_model."""
    model = MultiModalDeepfakeModel()
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
