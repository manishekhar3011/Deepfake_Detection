"""
audio_preprocessing.py
------------------------
Converts a raw uploaded audio file into a normalized Mel Spectrogram
tensor ready for AudioDeepfakeModel.

Pipeline:
    audio file -> load + resample (utils.audio_utils) -> Mel Spectrogram
    -> normalize -> torch tensor (1, 1, n_mels, time_frames)
"""

import torch

from utils.audio_utils import load_audio, get_duration, compute_mel_spectrogram

SAMPLE_RATE = 16000
N_MELS = 128
TARGET_TIME_FRAMES = 128


def preprocess_audio(file_path: str):
    """
    Args:
        file_path: path to an audio file saved temporarily on disk.

    Returns:
        dict with:
            tensor: torch.Tensor (1, 1, n_mels, time_frames) or None on failure.
            mel_db: raw np.ndarray mel spectrogram (for visualization) or None.
            duration_sec: float duration of the audio in seconds.
            success: bool
    """
    waveform, sr = load_audio(file_path, sr=SAMPLE_RATE)

    if waveform is None or len(waveform) == 0:
        return {"tensor": None, "mel_db": None, "duration_sec": 0.0, "success": False}

    duration = get_duration(waveform, sr)
    mel_db = compute_mel_spectrogram(
        waveform, sr, n_mels=N_MELS, target_length=TARGET_TIME_FRAMES
    )

    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,T)

    return {
        "tensor": tensor,
        "mel_db": mel_db,
        "duration_sec": duration,
        "success": True,
    }
