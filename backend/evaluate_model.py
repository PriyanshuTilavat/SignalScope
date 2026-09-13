from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from app.ml.model import create_model


# ==========================================
# SignalScope Model Evaluation
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "val"
MODEL_PATH = BASE_DIR / "models" / "signalscope_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("\n==========================================")
print("SignalScope Model Evaluation")
print("==========================================")

print(f"Device: {device}")
print(f"Validation dataset: {DATASET_DIR}")
print(f"Model: {MODEL_PATH}")


# ==========================================
# Validation preprocessing
# ==========================================

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ==========================================
# Load validation dataset
# ==========================================

val_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=val_transform
)

print("\nClasses:")
print(val_dataset.class_to_idx)

print(f"Validation images: {len(val_dataset)}")


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ==========================================
# Load trained model
# ==========================================

print("\nLoading trained model...")

model = create_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded successfully.")


# ==========================================
# Run predictions
# ==========================================

all_true = []
all_pred = []
all_probabilities = []

print("\nRunning evaluation...")

with torch.no_grad():

    for batch_number, (images, labels) in enumerate(val_loader, start=1):

        images = images.to(device)

        outputs = model(images)

        # ==========================================
        # IMPORTANT LABEL MAPPING
        # ==========================================
        #
        # ImageFolder:
        #
        # ai   = 0
        # real = 1
        #
        # The trained model therefore learned:
        #
        # sigmoid(output) = REAL probability
        #
        # So we convert it to:
        #
        # AI probability = 1 - REAL probability
        # ==========================================

        real_probabilities = torch.sigmoid(
            outputs
        ).squeeze(1)

        ai_probabilities = 1.0 - real_probabilities

        # AI >= 50% means predicted AI
        predictions = (
            ai_probabilities >= 0.5
        ).long()

        # Convert ImageFolder labels:
        #
        # ai = 0 → AI target = 1
        # real = 1 → AI target = 0

        true_ai_labels = (
            labels == val_dataset.class_to_idx["ai"]
        ).long()

        all_true.extend(
            true_ai_labels.cpu().numpy()
        )

        all_pred.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            ai_probabilities.cpu().numpy()
        )

        print(
            f"\rProcessed "
            f"{batch_number * BATCH_SIZE if batch_number * BATCH_SIZE < len(val_dataset) else len(val_dataset)}"
            f"/{len(val_dataset)} images",
            end=""
        )


print("\n")


# ==========================================
# Calculate metrics
# ==========================================

accuracy = accuracy_score(
    all_true,
    all_pred
)

precision = precision_score(
    all_true,
    all_pred,
    zero_division=0
)

recall = recall_score(
    all_true,
    all_pred,
    zero_division=0
)

f1 = f1_score(
    all_true,
    all_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_true,
    all_probabilities
)


# ==========================================
# Confusion Matrix
# ==========================================

matrix = confusion_matrix(
    all_true,
    all_pred
)


# ==========================================
# Main Results
# ==========================================

print("==========================================")
print("SIGNALSCOPE RESULTS")
print("==========================================")

print(
    f"\nAccuracy:  {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall:    {recall * 100:.2f}%"
)

print(
    f"F1 Score:  {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)


# ==========================================
# Confusion Matrix
# ==========================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print()
print("                 Predicted")
print("              Real     AI")

print(
    f"Actual Real  {matrix[0][0]:6d}  {matrix[0][1]:6d}"
)

print(
    f"Actual AI    {matrix[1][0]:6d}  {matrix[1][1]:6d}"
)


# ==========================================
# Classification Report
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        all_true,
        all_pred,
        target_names=["Real", "AI"],
        zero_division=0
    )
)


# ==========================================
# Final
# ==========================================

print("\n==========================================")
print("Evaluation complete!")
print("==========================================")