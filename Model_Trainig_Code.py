# %%
# train_emotion.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# -------------------------
# Config
# -------------------------
data_dir = "C:/Users/HWA/Desktop/AI Project UV"   # <-- change to your dataset folder
batch_size = 64
num_epochs = 20
num_classes = 7
learning_rate = 0.001
save_path = "Final_ModelV2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -------------------------
# Data transforms
# -------------------------
transform = {
    "train": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

# -------------------------
# Datasets and Loaders
# -------------------------
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform["train"])
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform["test"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# -------------------------
# Model (ResNet18 pretrained on ImageNet)
# -------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace final layer

model = model.to(device)

# -------------------------
# Loss and Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------
# Training loop
# -------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Acc: {acc:.2f}%")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")



