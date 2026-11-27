
import torch
from torchvision import datasets, transforms

print("PyTorch version:", torch.__version__)

transform = transforms.ToTensor()

print("\nDownloading MNIST dataset...")

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("\nDataset downloaded successfully!")
print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))
print("Image shape:", train_dataset[0][0].shape)
