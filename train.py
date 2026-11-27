
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.softmax_regression import SoftmaxRegression
from models.neural_network import NeuralNetwork

# Configuration
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset preprocessing
transform = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=False, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=False, transform=transform
)

# Data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / total

    return average_loss, accuracy

# Training function
def train_model(model, model_name):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    print(f"\nTraining {model_name}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = 100 * correct / total

        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_accuracy:.2f}%"
        )

    # Save trained model and history
    torch.save(model.state_dict(), f"results/{model_name}.pth")
    torch.save(history, f"results/{model_name}_history.pth")

    print(f"{model_name} training completed!")

    return model, history

# Train both models
softmax_model = SoftmaxRegression()
nn_model = NeuralNetwork()

softmax_model, softmax_history = train_model(
    softmax_model, "softmax"
)

nn_model, nn_history = train_model(
    nn_model, "neural_network"
)

print("\nAll training completed!")
