
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

from models.softmax_regression import SoftmaxRegression
from models.neural_network import NeuralNetwork

# Configuration
BATCH_SIZE = 64
device = torch.device("cpu")

# Load dataset
transform = transforms.Compose([
    transforms.Resize((14, 14)),
    transforms.ToTensor()
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load trained models
softmax_model = SoftmaxRegression()
softmax_model.load_state_dict(
    torch.load("results/softmax.pth", map_location=device, weights_only=True)
)

nn_model = NeuralNetwork()
nn_model.load_state_dict(
    torch.load("results/neural_network.pth", map_location=device, weights_only=True)
)

models = {
    "Softmax Regression": softmax_model,
    "Neural Network (MLP)": nn_model
}

# Generate predictions
for model_name, model in models.items():

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10)
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")

    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"results/confusion_matrix_{filename}.png", dpi=300)
    plt.show()

print("Confusion matrices generated successfully!")
