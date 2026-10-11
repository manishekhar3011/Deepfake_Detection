"""
audio_utils.py
---------------
Helper functions for loading audio and converting it into a Mel
Spectrogram suitable for CNN input, using librosa.
"""

import numpy as np
import librosa


def load_audio(file_path: str, sr: int = 16000):
    """
    Load an audio file and resample it to a target sample rate.

    Args:
        file_path: path to the audio file (.wav, .mp3, .m4a, ...)
        sr: target sample rate.

    Returns:
        (waveform, sample_rate) or (None, None) if loading fails.
    """
    try:
        waveform, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return waveform, sample_rate
    except Exception:
        return None, None


def get_duration(waveform: np.ndarray, sample_rate: int) -> float:
    if waveform is None or sample_rate is None or sample_rate == 0:
        return 0.0
    return float(len(waveform) / sample_rate)


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    target_length: int = 128,
):
    """
    Convert a raw waveform into a log-scaled Mel Spectrogram, resized
    (via pad/crop) to a fixed number of time frames so it can be fed
    into a fixed-input-size CNN.

    Args:
        waveform: 1D numpy array of audio samples.
        sample_rate: sample rate of the waveform.
        n_mels: number of Mel frequency bins.
        n_fft: FFT window size.
        hop_length: hop length between STFT windows.
        target_length: fixed number of time frames the spectrogram
            is padded/cropped to.

    Returns:
        mel_db: np.ndarray of shape (n_mels, target_length), float32,
            normalized to roughly [0, 1].
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or crop along time axis to a fixed length
    if mel_db.shape[1] < target_length:
        pad_width = target_length - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :target_length]

    # Normalize to [0, 1] for stable CNN training/inference
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 1e-6:
        mel_db = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_db = np.zeros_like(mel_db)

    return mel_db.astype(np.float32)
