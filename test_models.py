
import torch
from models.softmax_regression import SoftmaxRegression
from models.neural_network import NeuralNetwork

# Create both models
softmax_model = SoftmaxRegression()
nn_model = NeuralNetwork()

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("=== Model Comparison ===")

print("\nSoftmax Regression")
print("Parameters:", count_parameters(softmax_model))

print("\nNeural Network (MLP)")
print("Parameters:", count_parameters(nn_model))

# Test model input and output
sample_input = torch.randn(5, 196)

softmax_output = softmax_model(sample_input)
nn_output = nn_model(sample_input)

print("\n=== Output Shape Test ===")
print("Softmax output:", softmax_output.shape)
print("Neural Network output:", nn_output.shape)
