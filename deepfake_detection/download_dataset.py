import kagglehub

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "adham7elmy/faceforencispp-extracted-frames"
)

print("\nDataset downloaded successfully!")
print("Dataset location:")
print(path)