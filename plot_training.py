
import torch
import matplotlib.pyplot as plt

# Load training history
softmax_history = torch.load(
    "results/softmax_history.pth",
    map_location="cpu",
    weights_only=True
)

nn_history = torch.load(
    "results/neural_network_history.pth",
    map_location="cpu",
    weights_only=True
)

epochs = range(1, 11)

# Plot 1: Loss comparison
plt.figure(figsize=(9, 5))

plt.plot(epochs, softmax_history["train_loss"],
         marker="o", label="Softmax Train Loss")
plt.plot(epochs, softmax_history["test_loss"],
         marker="o", label="Softmax Test Loss")

plt.plot(epochs, nn_history["train_loss"],
         marker="s", label="MLP Train Loss")
plt.plot(epochs, nn_history["test_loss"],
         marker="s", label="MLP Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/loss_comparison.png", dpi=300)
plt.show()

# Plot 2: Accuracy comparison
plt.figure(figsize=(9, 5))

plt.plot(epochs, softmax_history["train_accuracy"],
         marker="o", label="Softmax Train Accuracy")
plt.plot(epochs, softmax_history["test_accuracy"],
         marker="o", label="Softmax Test Accuracy")

plt.plot(epochs, nn_history["train_accuracy"],
         marker="s", label="MLP Train Accuracy")
plt.plot(epochs, nn_history["test_accuracy"],
         marker="s", label="MLP Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Testing Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("results/accuracy_comparison.png", dpi=300)
plt.show()

print("Graphs saved successfully!")
