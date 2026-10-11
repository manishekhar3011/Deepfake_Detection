"""
train_multimodal.py
----------------------
Training script for the Multi-Modal (Video + Audio) fusion model.

Expects PAIRED samples: for each video, a matching audio track
extracted from the same source (e.g. the video's own soundtrack).
Organize your dataset as:

    dataset/video/real/xxx.mp4      dataset/audio/real/xxx.wav
    dataset/video/fake/xxx.mp4      dataset/audio/fake/xxx.wav

Filenames (without extension) must match between the video and audio
folders so this script can pair them correctly. You can extract audio
from a video with ffmpeg, e.g.:

    ffmpeg -i xxx.mp4 -q:a 0 -map a xxx.wav

Run:
    python training/train_multimodal.py --video_dir dataset/video --audio_dir dataset/audio --epochs 15

NOTE: This is the most resource-intensive script in the project since
it runs both the video and audio pipelines together. Expect slow
training on CPU-only laptops — consider a small subset of data first.
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
from models.multimodal_model import MultiModalDeepfakeModel
from preprocessing.video_preprocessing import preprocess_video, FRAME_SIZE
from preprocessing.audio_preprocessing import preprocess_audio, N_MELS, TARGET_TIME_FRAMES


class MultiModalDataset(Dataset):
    def __init__(self, video_dir: str, audio_dir: str, frames_per_video: int = 16):
        self.frames_per_video = frames_per_video
        self.samples = []  # (video_path, audio_path, label)

        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            v_folder = os.path.join(video_dir, label_name)
            a_folder = os.path.join(audio_dir, label_name)

            for ext in ("*.mp4", "*.avi", "*.mov"):
                for v_path in glob.glob(os.path.join(v_folder, ext)):
                    base_name = os.path.splitext(os.path.basename(v_path))[0]
                    matched_audio = None
                    for a_ext in ("wav", "mp3", "m4a"):
                        candidate = os.path.join(a_folder, f"{base_name}.{a_ext}")
                        if os.path.exists(candidate):
                            matched_audio = candidate
                            break
                    if matched_audio:
                        self.samples.append((v_path, matched_audio, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No matched video/audio pairs found. Ensure filenames match "
                "between dataset/video/<class>/ and dataset/audio/<class>/."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_path, a_path, label = self.samples[idx]

        v_result = preprocess_video(v_path, max_frames=self.frames_per_video)
        v_tensor = v_result["tensor"]
        if v_tensor is None:
            v_tensor = torch.zeros(1, self.frames_per_video, 3, FRAME_SIZE, FRAME_SIZE)

        a_result = preprocess_audio(a_path)
        a_tensor = a_result["tensor"]
        if a_tensor is None:
            a_tensor = torch.zeros(1, 1, N_MELS, TARGET_TIME_FRAMES)

        return v_tensor.squeeze(0), a_tensor.squeeze(0), label


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for v, a, labels in loader:
        v, a, labels = v.to(device), a.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(v, a)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * v.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="dataset/video")
    parser.add_argument("--audio_dir", default="dataset/audio")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--frames_per_video", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--out", default="trained_models/multimodal_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MultiModalDataset(args.video_dir, args.audio_dir, frames_per_video=args.frames_per_video)
    n_val = max(1, int(len(dataset) * 0.15))
    n_test = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = MultiModalDeepfakeModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

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
