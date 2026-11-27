
import torch
from torchvision import datasets, transforms

# Resize MNIST images from 28x28 to 14x14
transform = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.ToTensor()
])

# Load training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

# Load testing dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

# Check the processed data
image, label = train_dataset[0]

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))
print("Processed image shape:", image.shape)
print("Flattened image shape:", image.view(-1).shape)
print("Label:", label)
