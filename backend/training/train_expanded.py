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

DATASET_DIR = BASE_DIR / "dataset_expanded"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT:
# Save this experiment separately.
# This will NOT overwrite the original 87% model.
MODEL_PATH = MODEL_DIR / "signalscope_model_expanded.pth"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.0001

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
print("SignalScope - Expanded Dataset Training")
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
        p=0.10,
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
        f"Training directory not found:\n"
        f"{TRAIN_DIR}"
    )


if not VAL_DIR.exists():

    raise FileNotFoundError(
        f"Validation directory not found:\n"
        f"{VAL_DIR}"
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


# ============================================================
# CLASS MAPPING
# ============================================================

print("\nClasses:")

print(
    train_dataset.class_to_idx
)


print(
    f"Training images: "
    f"{len(train_dataset)}"
)


print(
    f"Validation images: "
    f"{len(val_dataset)}"
)


# ============================================================
# VERIFY LABEL MAPPING
# ============================================================

print("\nLabel mapping:")

for class_name, class_index in train_dataset.class_to_idx.items():

    print(
        f"{class_name} = {class_index}"
    )


print(
    "\nIMPORTANT:"
)

print(
    "ai = 0"
)

print(
    "real = 1"
)

print(
    "\nThe model output represents REAL probability."
)

print(
    "AI probability = 1 - REAL probability."
)


# ============================================================
# DATASET CLASS COUNTS
# ============================================================

ai_index = train_dataset.class_to_idx["ai"]

real_index = train_dataset.class_to_idx["real"]


ai_count = 0
real_count = 0


for _, label in train_dataset.samples:

    if label == ai_index:

        ai_count += 1

    elif label == real_index:

        real_count += 1


print("\nTraining class distribution:")

print(
    f"AI:   {ai_count}"
)

print(
    f"Real: {real_count}"
)

print(
    f"Total: {ai_count + real_count}"
)


# ============================================================
# VALIDATION COUNTS
# ============================================================

val_ai_count = 0
val_real_count = 0


for _, label in val_dataset.samples:

    if label == val_dataset.class_to_idx["ai"]:

        val_ai_count += 1

    elif label == val_dataset.class_to_idx["real"]:

        val_real_count += 1


print("\nValidation class distribution:")

print(
    f"AI:   {val_ai_count}"
)

print(
    f"Real: {val_real_count}"
)

print(
    f"Total: {val_ai_count + val_real_count}"
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

print("\nCreating SignalScope model...")

model = create_model()

model = model.to(device)


# ============================================================
# CLASS WEIGHT
# ============================================================

# ImageFolder:
#
# ai   = 0
# real = 1
#
# BCEWithLogitsLoss(pos_weight=...)
# applies pos_weight to label 1.
#
# Therefore REAL needs the weight because
# REAL is the smaller class.

real_weight = (
    ai_count / real_count
)


print(
    f"\nReal class loss weight: "
    f"{real_weight:.4f}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(

    pos_weight=torch.tensor(
        [real_weight],
        device=device
    )

)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

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
# TRAINING VARIABLES
# ============================================================

best_val_accuracy = 0.0

best_epoch = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):


    # ========================================================
    # EPOCH HEADER
    # ========================================================

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


        # ----------------------------------------------------
        # Reset gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        ).squeeze(1)


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

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


    # ========================================================
    # TRAIN METRICS
    # ========================================================

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


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                images
            ).squeeze(1)


            # ------------------------------------------------
            # Validation loss
            # ------------------------------------------------

            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                * images.size(0)
            )


            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

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


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

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


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


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

        best_epoch = epoch + 1


        torch.save(

            model.state_dict(),

            MODEL_PATH

        )


        print(
            "\n[OK] New best model saved!"
        )


        print(
            f"[OK] Validation accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )


        print(
            f"[OK] Saved to: "
            f"{MODEL_PATH}"
        )


    else:

        print(
            "\nNo improvement."
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)

print(
    "EXPANDED DATASET TRAINING COMPLETE"
)

print("=" * 60)


print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


print(
    f"Best epoch: "
    f"{best_epoch}"
)


print(
    f"Model saved to:"
)


print(
    MODEL_PATH
)


print("=" * 60)