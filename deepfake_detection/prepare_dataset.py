import os
import random
import shutil
from pathlib import Path

# ============================================================
# SOURCE DATASET PATH
# ============================================================

SOURCE_PATH = Path(
    r"C:\Users\manis\.cache\kagglehub\datasets\adham7elmy\faceforencispp-extracted-frames\versions\4"
)

FAKE_SOURCE = SOURCE_PATH / "fake"
REAL_SOURCE = SOURCE_PATH / "real"


# ============================================================
# PROJECT DATASET PATH
# ============================================================

OUTPUT_PATH = Path("dataset/image")


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# Real dataset ke according balanced samples
MAX_SAMPLES_PER_CLASS = 30000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


random.seed(RANDOM_SEED)


# ============================================================
# GET ALL IMAGES
# ============================================================

def get_images(folder):

    extensions = ["*.png", "*.jpg", "*.jpeg"]

    images = []

    for extension in extensions:

        images.extend(
            folder.rglob(extension)
        )

    return images


print("Reading dataset...")

fake_images = get_images(FAKE_SOURCE)

real_images = get_images(REAL_SOURCE)


print(f"Fake images found: {len(fake_images)}")
print(f"Real images found: {len(real_images)}")


# ============================================================
# BALANCE DATASET
# ============================================================

sample_size = min(
    len(real_images),
    len(fake_images),
    MAX_SAMPLES_PER_CLASS
)


print(f"\nUsing {sample_size} images per class")


fake_images = random.sample(
    fake_images,
    sample_size
)

real_images = random.sample(
    real_images,
    sample_size
)


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(images):

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)

    val_end = int(
        total * (TRAIN_RATIO + VAL_RATIO)
    )

    train = images[:train_end]

    val = images[train_end:val_end]

    test = images[val_end:]

    return train, val, test


fake_train, fake_val, fake_test = split_dataset(fake_images)

real_train, real_val, real_test = split_dataset(real_images)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

splits = {

    "train": {
        "fake": fake_train,
        "real": real_train
    },

    "val": {
        "fake": fake_val,
        "real": real_val
    },

    "test": {
        "fake": fake_test,
        "real": real_test
    }

}


# ============================================================
# COPY FILES
# ============================================================

def copy_images(images, destination):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, image_path in enumerate(images):

        new_name = (
            f"{index:06d}_{image_path.name}"
        )

        destination_file = (
            destination / new_name
        )

        shutil.copy2(
            image_path,
            destination_file
        )


print("\nPreparing dataset...\n")


for split_name, classes in splits.items():

    for class_name, images in classes.items():

        destination = (
            OUTPUT_PATH
            / split_name
            / class_name
        )

        print(
            f"Copying {split_name}/{class_name}: "
            f"{len(images)} images"
        )

        copy_images(
            images,
            destination
        )


print("\nDataset preparation completed successfully!")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nFinal Dataset Structure:")

for split in ["train", "val", "test"]:

    for label in ["real", "fake"]:

        path = (
            OUTPUT_PATH
            / split
            / label
        )

        count = len(
            list(path.glob("*"))
        )

        print(
            f"{split}/{label}: {count}"
        )