import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = "dataset/image"
MODEL_SAVE_PATH = "trained_models/deepfake_image_model.pth"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0003

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TRUTHLENS AI - DEEPFAKE IMAGE MODEL TRAINING")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# DATA TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD DATASETS
# ==========================================

print("\nLoading datasets...")


train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transform
)


test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=val_transform
)


print("Classes:", train_dataset.classes)

print("Train images:", len(train_dataset))

print("Validation images:", len(val_dataset))

print("Test images:", len(test_dataset))


# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading EfficientNet-B0 model...")


model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=2
)


model = model.to(DEVICE)


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ==========================================
# LEARNING RATE SCHEDULER
# ==========================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==========================================
# TRAIN FUNCTION
# ==========================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item() *
            images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


    epoch_loss = (
        running_loss /
        total
    )


    epoch_accuracy = (
        correct /
        total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def validate():

    model.eval()


    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in val_loader:


            images = images.to(DEVICE)

            labels = labels.to(DEVICE)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item() *
                images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predicted == labels
            ).sum().item()


    epoch_loss = (
        running_loss /
        total
    )


    epoch_accuracy = (
        correct /
        total
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ==========================================
# TRAINING LOOP
# ==========================================

best_accuracy = 0.0


os.makedirs(
    "trained_models",
    exist_ok=True
)


print("\nStarting training...")


start_time = time.time()


for epoch in range(NUM_EPOCHS):


    print("\n" + "=" * 60)


    print(
        f"Epoch "
        f"{epoch + 1}"
        f"/"
        f"{NUM_EPOCHS}"
    )


    print("=" * 60)


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        validate()
    )


    scheduler.step(
        val_accuracy
    )


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )


    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )


    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )


    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_accuracy:


        best_accuracy = val_accuracy


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "val_accuracy":
                    best_accuracy
            },

            MODEL_SAVE_PATH
        )


        print(
            "\nBEST MODEL SAVED!"
        )


        print(
            f"Best Accuracy: "
            f"{best_accuracy * 100:.2f}%"
        )


# ==========================================
# TRAINING COMPLETE
# ==========================================

total_time = (
    time.time() -
    start_time
)


print("\n" + "=" * 60)

print(
    "TRAINING COMPLETED!"
)

print("=" * 60)


print(
    f"Training Time: "
    f"{total_time / 60:.2f} minutes"
)


print(
    f"Best Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


# ==========================================
# LOAD BEST MODEL
# ==========================================

print(
    "\nLoading best model "
    "for testing..."
)


checkpoint = torch.load(
    MODEL_SAVE_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


# ==========================================
# TEST MODEL
# ==========================================

correct = 0

total = 0


with torch.no_grad():

    for images, labels in test_loader:


        images = images.to(DEVICE)

        labels = labels.to(DEVICE)


        outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)


        correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    correct /
    total
)


print("\n" + "=" * 60)


print(
    "FINAL TEST RESULTS"
)


print("=" * 60)


print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


print(
    "\nModel saved at:"
)


print(
    MODEL_SAVE_PATH
)


print("\nClasses:")


print(
    checkpoint["classes"]
)