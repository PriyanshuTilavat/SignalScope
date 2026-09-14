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

DATASET_DIR = (
    BASE_DIR
    / "dataset_universal"
)

SOURCE_MODEL = (
    BASE_DIR
    / "models"
    / "signalscope_model.pth"
)

OUTPUT_MODEL = (
    BASE_DIR
    / "models"
    / "signalscope_model_universal.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print("=" * 70)
print("SIGNALSCOPE UNIVERSAL MODEL FINE-TUNING")
print("=" * 70)

print(
    f"Device: {device}"
)

print(
    f"Starting model: {SOURCE_MODEL}"
)

print(
    f"Output model: {OUTPUT_MODEL}"
)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

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
        hue=0.03,
    ),

    transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 1.5)
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
    transforms.Resize((224, 224)),

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
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=val_transform
)


print()
print("Class mapping:")
print(train_dataset.class_to_idx)

print(
    f"Training images: {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)


# ============================================================
# DATA LOADERS
# ============================================================

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


# ============================================================
# LOAD EXISTING MODEL
# ============================================================

print()
print("Loading existing production model...")

model = create_model()

checkpoint = torch.load(
    SOURCE_MODEL,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    checkpoint
)

model.to(device)

print(
    "Existing model loaded successfully."
)


# ============================================================
# CLASS COUNTS
# ============================================================

ai_count = len(
    list(
        (
            DATASET_DIR
            / "train"
            / "ai"
        ).iterdir()
    )
)

real_count = len(
    list(
        (
            DATASET_DIR
            / "train"
            / "real"
        ).iterdir()
    )
)


print()
print(
    f"AI images: {ai_count}"
)

print(
    f"Real images: {real_count}"
)


# ============================================================
# LOSS
# ============================================================

# ImageFolder mapping:
#
# ai   = 0
# real = 1
#
# The model outputs REAL probability.
#
# Therefore the positive class for BCE is REAL.

real_weight = (
    ai_count
    / real_count
)


criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [real_weight],
        dtype=torch.float32,
        device=device
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

# Small learning rate because we are
# fine-tuning an already trained model.

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.00003,
    weight_decay=0.01,
)


# ============================================================
# TRAINING CONFIG
# ============================================================

EPOCHS = 5

best_val_accuracy = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 70)
print("STARTING UNIVERSAL FINE-TUNING")
print("=" * 70)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_correct = 0
    train_total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = (
            labels
            .float()
            .unsqueeze(1)
            .to(device)
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


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


        # ----------------------------------------------------
        # Prediction
        #
        # sigmoid(output) = REAL probability
        # AI probability = 1 - REAL probability
        # ----------------------------------------------------

        real_probability = (
            torch.sigmoid(
                outputs
            )
        )

        ai_probability = (
            1.0
            - real_probability
        )


        predictions = (
            ai_probability >= 0.5
        ).long()


        true_ai = (
            labels.long()
            == 0
        ).long()


        train_correct += (
            predictions
            == true_ai
        ).sum().item()


        train_total += (
            images.size(0)
        )


    train_loss = (
        running_loss
        / train_total
    )


    train_accuracy = (
        train_correct
        / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            real_probability = (
                torch.sigmoid(
                    outputs
                ).squeeze(1)
            )


            ai_probability = (
                1.0
                - real_probability
            )


            predictions = (
                ai_probability >= 0.5
            ).long()


            true_ai = (
                labels == 0
            ).long()


            val_correct += (
                predictions
                == true_ai
            ).sum().item()


            val_total += (
                images.size(0)
            )


    val_accuracy = (
        val_correct
        / val_total
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        f"Epoch "
        f"{epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy * 100:.2f}% | "
        f"Val Acc: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        torch.save(
            model.state_dict(),
            OUTPUT_MODEL
        )


        print(
            f"  ✓ Saved best universal model "
            f"({best_val_accuracy * 100:.2f}%)"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("UNIVERSAL FINE-TUNING COMPLETE")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print()
print(
    "Candidate model:"
)

print(
    OUTPUT_MODEL
)

print()
print(
    "Production model was NOT modified."
)

print("=" * 70)