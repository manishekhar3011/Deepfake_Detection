# 🛡️ Deepfake Detection Using Multi-Modal Learning

A college-level deep learning project that detects manipulated **images**,
**videos**, and **audio** using transfer learning (CNNs), temporal
modeling (LSTM), and Mel-spectrogram analysis — served through a
Streamlit web application.

---

## 1. Project Description

This project implements three independent deepfake detection pipelines
(image, video, audio) plus an optional multi-modal fusion model that
combines video and audio for a joint prediction. Users upload media
through a browser-based Streamlit UI and receive a **REAL / DEEPFAKE**
prediction with confidence scores.

> ⚠️ **Honesty notice:** This repository ships the full architecture,
> training pipeline, and UI, but **no pretrained weights**. Until you
> train the models on a dataset (see below), the app runs in clearly
> labeled **DEMO MODE**, where predictions come from an untrained
> classification head and must not be treated as real detection
> results.

---

## 2. Features

- 🖼️ Image deepfake detection (face-crop + EfficientNet-B0)
- 🎥 Video deepfake detection (frame sampling + face detection + CNN→LSTM)
- 🔊 Audio deepfake detection (Mel Spectrogram + custom CNN)
- 🔗 Optional video+audio multi-modal fusion model
- 📊 Per-frame probability graphs, Mel spectrogram visualization
- ⚙️ Automatic CPU/GPU detection
- 🧪 Clearly labeled Demo Mode when no trained weights are present
- 🛑 Defensive error handling (missing files, corrupt uploads, no face found, etc.)
- 📈 Evaluation script: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC

---

## 3. Technologies Used

Python · PyTorch · torchvision · OpenCV · Streamlit · Librosa ·
scikit-learn · Plotly · NumPy · Pandas · Pillow

---

## 4. Architecture

```
IMAGE
Input Image -> Face Detection -> EfficientNet-B0 -> Classifier -> Real/Fake

VIDEO
Video -> Frame Extraction -> Face Detection -> MobileNetV2 (CNN)
      -> Sequence of Features -> LSTM -> Dense -> Real/Fake

AUDIO
Audio -> Resample -> Mel Spectrogram -> Custom CNN -> Classifier -> Real/Fake

MULTI-MODAL FUSION (optional, video+audio)
Video -> CNN -> LSTM -> Visual Features  \
                                           + -> Fusion Dense Layers -> Real/Fake
Audio -> Mel Spectrogram -> CNN -> Audio Features /
```

---

## 5. Project Structure

```
deepfake_detection/
├── app.py                     # Streamlit application (entry point)
├── requirements.txt
├── README.md
├── .streamlit/config.toml     # Theme + upload size config
│
├── models/                    # Model architectures (PyTorch nn.Module)
│   ├── image_model.py
│   ├── video_model.py
│   ├── audio_model.py
│   └── multimodal_model.py
│
├── trained_models/            # Saved weights go here (.pth) — empty until trained
│
├── preprocessing/             # Raw upload -> model-ready tensor
│   ├── image_preprocessing.py
│   ├── video_preprocessing.py
│   └── audio_preprocessing.py
│
├── training/                  # Standalone training/evaluation scripts
│   ├── train_image.py
│   ├── train_video.py
│   ├── train_audio.py
│   ├── train_multimodal.py
│   └── evaluate.py
│
├── utils/                     # Shared helpers
│   ├── face_detection.py
│   ├── video_utils.py
│   ├── audio_utils.py
│   └── visualization.py
│
└── dataset/                   # Your data goes here (see dataset/README.md)
    ├── image/{real,fake}/
    ├── video/{real,fake}/
    └── audio/{real,fake}/
```

---

## 6. Installation

```bash
git clone <this-repo>
cd deepfake_detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 7. Dataset Preparation

See [`dataset/README.md`](dataset/README.md) for full instructions,
folder layout, and links to public research datasets
(FaceForensics++, Celeb-DF, DFDC, ASVspoof, WaveFake). No dataset is
bundled with this repository.

---

## 8. Model Training

Train each modality independently:

```bash
python training/train_image.py --data_dir dataset/image --epochs 15
python training/train_video.py --data_dir dataset/video --epochs 15
python training/train_audio.py --data_dir dataset/audio --epochs 20
python training/train_multimodal.py --video_dir dataset/video --audio_dir dataset/audio --epochs 15
```

Each script performs data augmentation, a train/val/test split, early
stopping, checkpointing (best validation accuracy saved automatically
to `trained_models/`), and prints final test accuracy.

Evaluate a trained model in more detail:

```bash
python training/evaluate.py --model image --data_dir dataset/image --weights trained_models/image_model.pth
```

This produces a confusion matrix and ROC curve plot alongside
Accuracy / Precision / Recall / F1 / ROC-AUC in the console.

---

## 9. Running the Streamlit App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`)
in your browser.

- If `trained_models/*.pth` files are missing, the corresponding tab
  runs in **Demo Mode** with a clear on-screen warning.
- Once you've trained a model and saved its `.pth` file into
  `trained_models/`, restart the app to load real weights automatically.

---

## 10. Screenshots

_Add screenshots here after running the app, e.g.:_

```
screenshots/
├── home.png
├── image_detection.png
├── video_detection.png
└── audio_detection.png
```

---

## 11. Results

_Fill this in after training on your dataset_, e.g.:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Image | — | — | — | — | — |
| Video | — | — | — | — | — |
| Audio | — | — | — | — | — |

---

## 12. Limitations

- No trained weights are bundled — results are only as good as the
  dataset you train on.
- The multi-modal fusion model uses lightweight late-fusion
  (concatenation + MLP) rather than attention-based fusion, to remain
  trainable on a normal laptop.
- Face detection uses OpenCV Haar Cascades for zero extra dependencies;
  this is less robust than DNN-based detectors (e.g. MTCNN) in
  low-light or extreme-angle conditions.
- **No deepfake detector is 100% accurate.** Expect both false
  positives and false negatives, especially on data that differs from
  the training distribution (different demographics, compression
  levels, generation methods, etc.).

---

## 13. Future Scope

- Swap Haar Cascade for a DNN-based or MTCNN face detector.
- Add attention-based cross-modal fusion (transformer-style) once GPU
  resources allow.
- Add temporal-frequency (spectral) artifact analysis for video (e.g.
  detecting GAN fingerprints in the frequency domain).
- Support batch/API-based inference for integration into other systems.
- Add explainability (e.g. Grad-CAM heatmaps) so predictions are
  interpretable, not just a probability score.

---

## License / Ethical Use

This project is for **educational and research purposes**: building
detectors that help identify manipulated media, not creating
deepfakes. Please use responsibly and in accordance with the license
terms of any dataset you choose to train on.
