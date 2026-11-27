
import torch
import pandas as pd

# Load training histories
softmax = torch.load(
    "results/softmax_history.pth",
    map_location="cpu",
    weights_only=True
)

mlp = torch.load(
    "results/neural_network_history.pth",
    map_location="cpu",
    weights_only=True
)

# Create comparison table
results = {
    "Metric": [
        "Number of Parameters",
        "Final Training Accuracy (%)",
        "Final Testing Accuracy (%)",
        "Final Training Loss",
        "Final Testing Loss"
    ],
    "Softmax Regression": [
        1970,
        round(softmax["train_accuracy"][-1], 2),
        round(softmax["test_accuracy"][-1], 2),
        round(softmax["train_loss"][-1], 4),
        round(softmax["test_loss"][-1], 4)
    ],
    "Neural Network (MLP)": [
        20710,
        round(mlp["train_accuracy"][-1], 2),
        round(mlp["test_accuracy"][-1], 2),
        round(mlp["train_loss"][-1], 4),
        round(mlp["test_loss"][-1], 4)
    ]
}

df = pd.DataFrame(results)

print("\n=== FINAL EXPERIMENT RESULTS ===")
print(df.to_string(index=False))

df.to_csv("results/final_results.csv", index=False)

print("\nResults saved to results/final_results.csv")
