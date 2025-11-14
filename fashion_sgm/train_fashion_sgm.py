import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================
#  Crear carpeta de resultados
# ===========================
os.makedirs("results", exist_ok=True)

# ===========================
#  Cargar Fashion-MNIST
# ===========================
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ===========================
# Configurar dispositivo (M1)
# ===========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f" Using device: {device}")

# ===========================
# Definir modelo simple
# ===========================
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),   # wider first layer
    nn.ReLU(),
    nn.Linear(256, 128),       # another hidden layer
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)


criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===========================
#  Configurar restricciones
# ===========================
target_class = 7  # Sneaker
epsilon = {i: 0.3 for i in range(10)}  # κ_i tolerancia por clase
epochs = 140
lambda_penalty = 4  # peso de penalización

# Para registrar las pérdidas
loss_target_hist = []
penalty_hist = []
per_class_hist = []

# ===========================
#  Entrenamiento
# ===========================
for epoch in range(epochs):
    per_class_loss_epoch = torch.zeros(10, device=device)
    total_batches = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(imgs)
        losses = criterion(logits, labels)

        # Pérdida media por clase
        per_class_loss = []
        for c in range(10):
            mask = (labels == c)
            if mask.any():
                per_class_loss.append(losses[mask].mean())
            else:
                per_class_loss.append(torch.tensor(0.0, device=device))
        per_class_loss = torch.stack(per_class_loss)

        # Objetivo principal: minimizar pérdida de Sneaker
        obj = per_class_loss[target_class]

        # ===============================
        # 🔹 Nueva forma de restricción (max_i g_i)
        # ===============================
        # g_i = Loss_i - κ_i
        violations = per_class_loss - torch.tensor(list(epsilon.values()), device=device)
        violations[target_class] = 0.0  # excluir clase objetivo

        # g(w) = max_i g_i(w)
        g_value = torch.max(violations)  # la peor violación entre clases

        # Penalización (solo si hay violación)
        penalty = torch.relu(g_value)

        # ===============================
        # 🔹 Pérdida total (SGM-like combinación)
        # ===============================
        total_loss = obj + lambda_penalty * penalty

        total_loss.backward()
        optimizer.step()

        per_class_loss_epoch += per_class_loss.detach()
        total_batches += 1

    # Promedios de la época
    per_class_loss_epoch /= total_batches
    per_class_hist.append(per_class_loss_epoch.cpu())
    loss_target_hist.append(per_class_loss_epoch[target_class].item())
    penalty_hist.append(penalty.item())

    print(f"Epoch {epoch+1} | Target(Sneaker) Loss={loss_target_hist[-1]:.3f} | Penalty={penalty_hist[-1]:.3f}")

# ===========================
#  Guardar resultados
# ===========================
torch.save(model.state_dict(), "results/fashion_sgm_model_max.pth")
torch.save({
    "loss_target_hist": loss_target_hist,
    "penalty_hist": penalty_hist,
    "per_class_hist": torch.stack(per_class_hist)
}, "results/training_logs.pt")

# ===========================
#  Graficar
# ===========================
per_class_hist = torch.stack(per_class_hist).numpy()

plt.figure(figsize=(10,5))
plt.plot(loss_target_hist, label='Target Class (Sneaker) Loss')
plt.plot(penalty_hist, label='Penalty (Max Constraint Violation)')
plt.title("Training Progress – Max Constraint (SGM-style)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("results/training_progress_max.png")
plt.close()

plt.figure(figsize=(12,6))
for i in range(10):
    plt.plot(per_class_hist[:, i], label=classes[i])
plt.axhline(0.3, color='red', linestyle='--', label='κ_i limit (0.3)')
plt.title("Average Loss per Class – Max Constraint Version")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.legend()
plt.savefig("results/per_class_losses_max.png")
plt.close()

print(" Entrenamiento completado con max-constraint. Resultados en ./results/")
