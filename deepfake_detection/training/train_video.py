"""
train_video.py
----------------
Training script for the Video Deepfake Detection model (CNN + LSTM).

Expects the dataset in:
    dataset/video/real/*.mp4
    dataset/video/fake/*.mp4

Run:
    python training/train_video.py --data_dir dataset/video --epochs 15

NOTE: Video training is significantly heavier than image training
(each sample requires decoding + face-detecting multiple frames).
On a laptop CPU, expect this to be slow — reduce --frames_per_video
and dataset size for quick experiments.
"""

import os
import sys
import glob
import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.video_model import VideoDeepfakeModel
from preprocessing.video_preprocessing import preprocess_video, FRAME_SIZE


class VideoDataset(Dataset):
    """Loads video file paths and lazily preprocesses them on __getitem__."""

    def __init__(self, data_dir: str, frames_per_video: int = 16):
        self.frames_per_video = frames_per_video
        self.samples = []  # list of (path, label)

        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            folder = os.path.join(data_dir, label_name)
            for ext in ("*.mp4", "*.avi", "*.mov"):
                for path in glob.glob(os.path.join(folder, ext)):
                    self.samples.append((path, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No videos found in {data_dir}/real or {data_dir}/fake. "
                "Please populate the dataset folders first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        result = preprocess_video(path, max_frames=self.frames_per_video)
        tensor = result["tensor"]

        if tensor is None:
            # Fallback: return a zero tensor if the video is unreadable,
            # so a single corrupted file doesn't crash the whole run.
            tensor = torch.zeros(1, self.frames_per_video, 3, FRAME_SIZE, FRAME_SIZE)

        return tensor.squeeze(0), label  # (T, 3, 224, 224), label


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        video_logits, _ = model(sequences)
        loss = criterion(video_logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        preds = video_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/video")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--frames_per_video", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--out", default="trained_models/video_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = VideoDataset(args.data_dir, frames_per_video=args.frames_per_video)
    n_val = max(1, int(len(dataset) * 0.15))
    n_test = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VideoDeepfakeModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best_val_acc, best_state, epochs_no_improve = 0.0, None, 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(best_state, args.out)
            print(f"  -> New best model saved to {args.out} (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
