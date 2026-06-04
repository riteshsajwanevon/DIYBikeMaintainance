import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ============================================
# SETTINGS
# ============================================

DATASET_PATH = BASE_DIR / "dataset" / "classification"

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# TRANSFORMS
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ============================================
# DATASETS
# ============================================


train_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH / "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=str(DATASET_PATH / "valid"),
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================
# MODEL
# ============================================

model = models.efficientnet_b0(pretrained=True)

num_classes = len(train_dataset.classes)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)

# ============================================
# LOSS + OPTIMIZER
# ============================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================
# TRAINING LOOP
# ============================================

best_acc = 0

for epoch in range(EPOCHS):

    # ======================
    # TRAIN
    # ======================

    model.train()

    running_loss = 0

    for images, labels in tqdm(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    # ======================
    # VALIDATION
    # ======================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print(f"Loss: {running_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save best model
    if accuracy > best_acc:

        best_acc = accuracy

        torch.save(
            model.state_dict(),
            "best_oil_classifier.pth"
        )

        print("Best model saved!")

print("\nTraining Complete!")