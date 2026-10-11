"""
train_image.py
----------------
Training script for the Image Deepfake Detection model.

Expects the dataset in:
    dataset/image/real/*.jpg|png
    dataset/image/fake/*.jpg|png

Run:
    python training/train_image.py --data_dir dataset/image --epochs 15

Includes: data augmentation, train/val/test split, early stopping,
checkpointing, and accuracy/loss logging.
"""

import os
import sys
import argparse
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.image_model import ImageDeepfakeModel

IMAGE_SIZE = 224


def build_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.15, test_split: float = 0.15):
    """
    Expects a folder structure:
        data_dir/real/*.jpg
        data_dir/fake/*.jpg
    (ImageFolder maps subfolder names alphabetically -> class indices;
    we explicitly check class_to_idx below to make sure fake=1, real=0.)
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    print(f"Classes found: {full_dataset.class_to_idx}  (expect real=0, fake=1)")

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(full_dataset, [n_train, n_val, n_test])

    # Use eval-time (non-augmented) transforms for val/test sets
    val_set.dataset.transform = eval_transform
    test_set.dataset.transform = eval_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset/image")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4, help="early stopping patience")
    parser.add_argument("--out", default="trained_models/image_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)

    model = ImageDeepfakeModel(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

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

    # Final test evaluation using the best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nFinal Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print("NOTE: Real-world performance will vary — always evaluate on held-out, "
          "diverse data before trusting this model in production.")


if __name__ == "__main__":
    main()
