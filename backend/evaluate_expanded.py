
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


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset_expanded"
VAL_DIR = DATASET_DIR / "val"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "signalscope_model_expanded.pth"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SignalScope Expanded Model Evaluation")
print("=" * 60)

print(f"Device: {device}")
print(f"Validation dataset: {VAL_DIR}")
print(f"Model: {MODEL_PATH}")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("\nClasses:")
print(dataset.class_to_idx)

print(f"Validation images: {len(dataset)}")


# ============================================================
# MODEL
# ============================================================

model = create_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
    )
)

model.to(device)
model.eval()


# ============================================================
# PREDICTION
# ============================================================

all_labels = []
all_probs = []
all_preds = []

print("\nEvaluating...\n")

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images).squeeze(1)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities >= 0.5
        ).long()

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_probs.extend(
            probabilities.cpu().numpy().tolist()
        )

        all_preds.extend(
            predictions.cpu().numpy().tolist()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_preds,
)

precision = precision_score(
    all_labels,
    all_preds,
)

recall = recall_score(
    all_labels,
    all_preds,
)

f1 = f1_score(
    all_labels,
    all_preds,
)

roc_auc = roc_auc_score(
    all_labels,
    all_probs,
)


cm = confusion_matrix(
    all_labels,
    all_preds,
)

report = classification_report(
    all_labels,
    all_preds,
    target_names=["AI", "Real"],
)


# ============================================================
# RESULTS
# ============================================================

print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"Accuracy :  {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}%")
print(f"Recall   :  {recall * 100:.2f}%")
print(f"F1 Score :  {f1 * 100:.2f}%")
print(f"ROC-AUC  :  {roc_auc:.4f}")


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\nPredicted")
print("           AI      Real")
print(
    f"Actual AI   {cm[0][0]:4d}    {cm[0][1]:4d}"
)
print(
    f"Actual Real {cm[1][0]:4d}    {cm[1][1]:4d}"
)


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)

print("=" * 60)
print("Evaluation complete!")
print("=" * 60)