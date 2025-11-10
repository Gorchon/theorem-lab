import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===========================
# Cargar Fashion-MNIST
# ===========================
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

# Etiquetas (para referencia)
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ===========================
#  Definir modelo simple
# ===========================
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===========================
# Configurar restricciones
# ===========================
target_class = 7  # Sneaker
epsilon = {i: 0.3 for i in range(10)}  # κ_i tolerancia por clase
epochs = 5

# Para registrar las pérdidas
loss_target_hist = []
penalty_hist = []
per_class_hist = []

# ===========================
#  Entrenamiento
# ===========================
for epoch in range(epochs):
    per_class_loss_epoch = torch.zeros(10)
    total_batches = 0

    for imgs, labels in train_loader:
        optimizer.zero_grad()
        logits = model(imgs)
        losses = criterion(logits, labels)

        # Pérdida media por clase
        per_class_loss = []
        for c in range(10):
            if (labels == c).any():
                per_class_loss.append(losses[labels == c].mean())
            else:
                per_class_loss.append(torch.tensor(0.0))
        per_class_loss = torch.stack(per_class_loss)

        # Objetivo: minimizar pérdida de la clase objetivo
        obj = per_class_loss[target_class]

        # Restricciones: penalizar violaciones κ_i
        penalties = torch.relu(per_class_loss - torch.tensor(list(epsilon.values())))
        penalty = penalties[torch.arange(10) != target_class].mean()

        # PPALA-style combinación
        total_loss = obj + 0.6 * penalty
        total_loss.backward()
        optimizer.step()

        per_class_loss_epoch += per_class_loss.detach()
        total_batches += 1

    # Promedios de la época
    per_class_loss_epoch /= total_batches
    per_class_hist.append(per_class_loss_epoch)
    loss_target_hist.append(per_class_loss_epoch[target_class].item())
    penalty_hist.append(penalty.item())

    print(f"Epoch {epoch+1} | Target(Sneaker) Loss={loss_target_hist[-1]:.3f} | Penalty={penalty_hist[-1]:.3f}")



per_class_hist = torch.stack(per_class_hist).numpy()

plt.figure(figsize=(10,5))
plt.plot(loss_target_hist, label='Target Class (Sneaker) Loss')
plt.plot(penalty_hist, label='Penalty (Constraint Violation)')
plt.title("Training Progress – SGM-style with Class Constraints")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Pérdidas promedio por clase
plt.figure(figsize=(12,6))
for i in range(10):
    plt.plot(per_class_hist[:, i], label=classes[i])
plt.title("Average Loss per Class Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.legend()
plt.show()
