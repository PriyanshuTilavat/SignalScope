from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.ml.model import create_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset_gemini_expanded"

MODEL_OUTPUT = (
    BASE_DIR
    / "models"
    / "signalscope_model_gemini.pth"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# Transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 2.0)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=val_transform,
)

print("Class mapping:")
print(train_dataset.class_to_idx)

print(
    f"Training images: {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)


# --------------------------------------------------
# Data loaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = create_model()
model.to(device)


# --------------------------------------------------
# Class weighting
#
# ImageFolder:
# ai   = 0
# real = 1
#
# BCEWithLogitsLoss pos_weight applies
# to the positive class = real.
# --------------------------------------------------

ai_count = len(
    list(
        (DATASET_DIR / "train" / "ai").iterdir()
    )
)

real_count = len(
    list(
        (DATASET_DIR / "train" / "real").iterdir()
    )
)

real_weight = ai_count / real_count

print(f"AI images: {ai_count}")
print(f"Real images: {real_count}")
print(f"Real class weight: {real_weight:.4f}")


criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [real_weight],
        dtype=torch.float32,
        device=device,
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.01,
)


# --------------------------------------------------
# Training
# --------------------------------------------------

EPOCHS = 10

best_accuracy = 0.0

print()
print("=" * 60)
print("STARTING GEMINI-AWARE TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = (
            labels.float()
            .unsqueeze(1)
            .to(device)
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        # Model output represents REAL probability.
        real_probability = torch.sigmoid(outputs)

        predictions = (
            real_probability < 0.5
        ).long()

        # predictions above is AI=1/real=0
        # Convert ImageFolder labels:
        # AI=0 -> AI=1
        # REAL=1 -> AI=0
        true_ai = (
            labels.long() == 0
        ).long()

        correct += (
            predictions == true_ai
        ).sum().item()

        total += images.size(0)

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            real_probability = torch.sigmoid(
                outputs
            ).squeeze(1)

            ai_probability = (
                1.0 - real_probability
            )

            predictions = (
                ai_probability >= 0.5
            ).long()

            true_ai = (
                labels == 0
            ).long()

            val_correct += (
                predictions == true_ai
            ).sum().item()

            val_total += images.size(0)

    val_accuracy = (
        val_correct / val_total
    )


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------
    # Save best candidate
    # --------------------------------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_OUTPUT
        )

        print(
            f"  ✓ Saved best model "
            f"({best_accuracy * 100:.2f}%)"
        )


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Candidate model saved to:"
)

print(MODEL_OUTPUT)