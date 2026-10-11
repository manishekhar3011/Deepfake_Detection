"""
evaluate.py
------------
Generic evaluation script for a trained image/audio classification
model. Computes Accuracy, Precision, Recall, F1, Confusion Matrix, and
ROC-AUC, and saves plots of the confusion matrix + ROC curve.

Run (example for the image model):
    python training/evaluate.py --model image --data_dir dataset/image --weights trained_models/image_model.pth

IMPORTANT: No deepfake detector is 100% accurate. Always report these
metrics honestly and expect both false positives and false negatives,
especially on data that differs from the training distribution.
"""

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.image_model import ImageDeepfakeModel
from models.audio_model import AudioDeepfakeModel

IMAGE_SIZE = 224


def get_image_probs_labels(weights_path, data_dir, device):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = ImageDeepfakeModel(pretrained=True, freeze_backbone=True)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = torch.softmax(model(images), dim=1)
            fake_probs = outputs[:, 1].cpu().numpy()
            all_probs.extend(fake_probs)
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def compute_and_report_metrics(probs: np.ndarray, labels: np.ndarray, out_prefix: str):
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    print("\n===== Evaluation Report =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nReminder: this model is NOT 100% accurate. False positives "
          "(real flagged as fake) and false negatives (fake flagged as "
          "real) are both possible — treat predictions as a decision-support "
          "signal, not ground truth.")

    # Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "DEEPFAKE"]).plot(ax=ax_cm, cmap="Blues")
    fig_cm.savefig(f"{out_prefix}_confusion_matrix.png", bbox_inches="tight")

    # ROC curve
    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#FF4B4B")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        fig_roc.savefig(f"{out_prefix}_roc_curve.png", bbox_inches="tight")
        print(f"\nROC-AUC: {roc_auc:.4f}")

    print(f"\nPlots saved as {out_prefix}_confusion_matrix.png and {out_prefix}_roc_curve.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["image", "audio"], required=True,
                         help="Which model to evaluate (video/multimodal use the same "
                              "pattern — extend get_image_probs_labels-style helpers as needed).")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_prefix", default="evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "image":
        probs, labels = get_image_probs_labels(args.weights, args.data_dir, device)
    else:
        raise NotImplementedError(
            "Audio evaluation requires an AudioFolder-style dataset loader — "
            "see training/train_audio.py's AudioDataset for the pattern to adapt here."
        )

    compute_and_report_metrics(probs, labels, args.out_prefix)


if __name__ == "__main__":
    main()
