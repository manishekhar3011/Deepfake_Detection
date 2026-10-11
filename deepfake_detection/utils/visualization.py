"""
visualization.py
------------------
Helper functions to build Plotly figures used across the Streamlit app:
- Per-frame deepfake probability graph (video module)
- Mel spectrogram heatmap (audio module)
- REAL vs DEEPFAKE probability bar chart (all modules)
"""

import numpy as np
import plotly.graph_objects as go


def plot_frame_probabilities(frame_probs):
    """
    Line chart of the deepfake probability for each analyzed video frame.

    Args:
        frame_probs: list/array of floats in [0, 1], one per frame.
    """
    x = list(range(1, len(frame_probs) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[p * 100 for p in frame_probs],
            mode="lines+markers",
            name="Deepfake Probability (%)",
            line=dict(color="#FF4B4B", width=2),
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Decision threshold")
    fig.update_layout(
        title="Deepfake Probability Across Analyzed Frames",
        xaxis_title="Frame Index",
        yaxis_title="Deepfake Probability (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=350,
    )
    return fig


def plot_mel_spectrogram(mel_db: np.ndarray):
    """Heatmap visualization of a (n_mels, time) mel spectrogram array."""
    fig = go.Figure(
        data=go.Heatmap(
            z=mel_db,
            colorscale="Magma",
            colorbar=dict(title="Normalized dB"),
        )
    )
    fig.update_layout(
        title="Mel Spectrogram",
        xaxis_title="Time Frames",
        yaxis_title="Mel Frequency Bins",
        template="plotly_white",
        height=350,
    )
    return fig


def plot_probability_bar(real_prob: float, fake_prob: float):
    """Simple horizontal bar chart comparing REAL vs DEEPFAKE probability."""
    fig = go.Figure(
        go.Bar(
            x=[real_prob * 100, fake_prob * 100],
            y=["REAL", "DEEPFAKE"],
            orientation="h",
            marker_color=["#2ECC71", "#FF4B4B"],
            text=[f"{real_prob*100:.1f}%", f"{fake_prob*100:.1f}%"],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis_title="Probability (%)",
        xaxis_range=[0, 100],
        template="plotly_white",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
