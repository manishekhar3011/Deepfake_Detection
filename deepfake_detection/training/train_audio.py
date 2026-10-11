"""
train_audio.py
----------------
Training script for the Audio Deepfake Detection model (Mel Spectrogram + CNN).

Expects the dataset in:
    dataset/audio/real/*.wav|mp3|m4a
    dataset/audio/fake/*.wav|mp3|m4a

Run:
    python training/train_audio.py --data_dir dataset/audio --epochs 20
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
from models.audio_model import AudioDeepfakeModel
from preprocessing.audio_preprocessing import preprocess_audio, N_MELS, TARGET_TIME_FRAMES


class AudioDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            folder = os.path.join(data_dir, label_name)
            for ext in ("*.wav", "*.mp3", "*.m4a"):
                for path in glob.glob(os.path.join(folder, ext)):
                    self.samples.append((path, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No audio files found in {data_dir}/real or {data_dir}/fake. "
                "Please populate the dataset folders first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        result = preprocess_audio(path)
        tensor = result["tensor"]

        if tensor is None:
            tensor = torch.zeros(1, 1, N_MELS, TARGET_TIME_FRAMES)

        return tensor.squeeze(0), label  # (1, n_mels, T), label


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(specs)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * specs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/audio")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--out", default="trained_models/audio_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = AudioDataset(args.data_dir)
    n_val = max(1, int(len(dataset) * 0.15))
    n_test = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = AudioDeepfakeModel(n_mels=N_MELS, time_frames=TARGET_TIME_FRAMES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
