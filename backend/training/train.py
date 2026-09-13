from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.ml.model import create_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "signalscope_model.pth"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.00005

WEIGHT_DECAY = 0.01


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("SignalScope - Transfer Learning Training")
print("=" * 60)

print(f"Device: {device}")

print(f"Dataset: {DATASET_DIR}")

print(f"Model output: {MODEL_PATH}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3
            )
        ],
        p=0.15,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


val_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ],
    ),
])


# ============================================================
# CHECK DATASET
# ============================================================

if not TRAIN_DIR.exists():

    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_DIR}"
    )


if not VAL_DIR.exists():

    raise FileNotFoundError(
        f"Validation dataset not found:\n{VAL_DIR}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform,
)


print("\nClasses:")
print(train_dataset.class_to_idx)

print(
    f"Training images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)


# ============================================================
# IMPORTANT LABEL CHECK
# ============================================================

print("\nLabel mapping:")

for class_name, class_index in train_dataset.class_to_idx.items():

    print(
        f"{class_name} = {class_index}"
    )

print(
    "\nThe model output represents REAL probability."
)

print(
    "AI probability = 1 - REAL probability."
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating pretrained ResNet18...")

model = create_model()


# ============================================================
# FREEZE BACKBONE
# ============================================================

print("\nFreezing ResNet18 backbone...")

for parameter in model.backbone.parameters():

    parameter.requires_grad = False


# ============================================================
# UNFREEZE LAYER 4
# ============================================================

print("Unfreezing ResNet18 Layer 4...")

for parameter in model.backbone.layer4.parameters():

    parameter.requires_grad = True


# ============================================================
# UNFREEZE CLASSIFIER
# ============================================================

print("Unfreezing classifier...")

for parameter in model.backbone.fc.parameters():

    parameter.requires_grad = True


model = model.to(device)


# ============================================================
# COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()

    if parameter.requires_grad
)


total_parameters = sum(

    parameter.numel()

    for parameter in model.parameters()
)


print(
    f"\nTrainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    filter(
        lambda parameter:
        parameter.requires_grad,
        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2,
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print("=" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.float().to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        ).squeeze(1)


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


        probabilities = torch.sigmoid(
            outputs
        )


        predictions = (
            probabilities >= 0.5
        ).float()


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.float().to(device)


            outputs = model(
                images
            ).squeeze(1)


            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                * images.size(0)
            )


            probabilities = torch.sigmoid(
                outputs
            )


            predictions = (
                probabilities >= 0.5
            ).float()


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += labels.size(0)


    val_loss = (
        val_loss_total / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # ========================================================
    # LEARNING RATE UPDATE
    # ========================================================

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Train Loss:      "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss:        "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy:    "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate:   "
        f"{current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(

            model.state_dict(),

            MODEL_PATH,

        )


        print(
            "\n[OK] Best model saved!"
        )

        print(
            f"[OK] {MODEL_PATH}"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)

print("TRANSFER LEARNING COMPLETE")

print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Model saved to: "
    f"{MODEL_PATH}"
)

print("=" * 60)