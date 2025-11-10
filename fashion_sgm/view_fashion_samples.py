import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# === Cargar Fashion-MNIST ==
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# Nombres de las clases
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# === Mostrar algunas imágenes ===
plt.figure(figsize=(10, 5))
for i in range(10):
    img, label = train[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(classes[label])
    plt.axis("off")

plt.tight_layout()
plt.show()
