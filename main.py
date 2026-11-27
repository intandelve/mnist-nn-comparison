# ============================================================
# MNIST Digit Recognition - Comparative Study of NN Architectures
# Course: General Introduction to AI
# ============================================================

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------- 0. Setup ----------
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 1. Dataset ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=1000, shuffle=False)

print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")

# ---------- 2. Models ----------
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

class OneHiddenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class TwoHiddenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# ---------- 3. Train & Evaluate ----------
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

def train_model(model, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'test_acc': []}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(acc)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

    elapsed = time.time() - start
    return history, elapsed

# ---------- 4. Run Experiment ----------
models = {
    'Linear':   LinearModel().to(device),
    '1-Hidden': OneHiddenModel().to(device),
    '2-Hidden': TwoHiddenModel().to(device),
}

results = {}
for name, model in models.items():
    print(f"\n===== Training {name} =====")
    n_params = sum(p.numel() for p in model.parameters())
    hist, t = train_model(model, epochs=5)
    results[name] = {'history': hist, 'time': t, 'params': n_params}
    print(f"  Params: {n_params:,} | Time: {t:.1f}s | Final Acc: {hist['test_acc'][-1]:.2f}%")

# ---------- 5. Save Summary ----------
with open("results/summary.txt", "w") as f:
    f.write("Model,Params,Time(s),Final_Acc(%)\n")
    for name, r in results.items():
        f.write(f"{name},{r['params']},{r['time']:.1f},{r['history']['test_acc'][-1]:.2f}\n")
print("\nSummary saved to results/summary.txt")

# ---------- 6. Plot Accuracy ----------
plt.figure(figsize=(8, 5))
for name, r in results.items():
    plt.plot(range(1, 6), r['history']['test_acc'], marker='o', label=name)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('results/accuracy_comparison.png', dpi=150)
plt.show()

# ---------- 7. Plot Loss ----------
plt.figure(figsize=(8, 5))
for name, r in results.items():
    plt.plot(range(1, 6), r['history']['train_loss'], marker='o', label=name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('results/loss_comparison.png', dpi=150)
plt.show()

# ---------- 8. Error Analysis (2-Hidden) ----------
model = models['2-Hidden']
model.eval()
wrong = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        for i in range(len(y)):
            if pred[i] != y[i]:
                wrong.append((x[i].cpu(), y[i].item(), pred[i].item()))
        if len(wrong) >= 8:
            break

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, (img, true, pred) in zip(axes.flat, wrong[:8]):
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"True:{true} Pred:{pred}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('results/errors.png', dpi=150)
plt.show()

print("\nAll done! Check the 'results' folder.")