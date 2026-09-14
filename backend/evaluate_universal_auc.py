from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

from app.ml.model import create_model


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "signalscope_model_universal.pth"
)

GEMINI_TEST_DIR = (
    BASE_DIR
    / "dataset_gemini"
    / "test"
    / "ai"
)

EXTERNAL_DATASET = (
    "TheKernel01/AIGC-Detection-Benchmark"
)

IMAGES_PER_CLASS = 100

GENERATOR_IDS = {
    "DALL-E 2": 4,
    "SDXL": 11,
    "CycleGAN": 3,
    "Real": 0,
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
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
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("SIGNALSCOPE UNIVERSAL MODEL ROC-AUC ANALYSIS")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print(
    f"Model: {MODEL_PATH}"
)


model = create_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(
    checkpoint
)

model.to(DEVICE)
model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# PREDICT
# ============================================================

def predict(image):

    image = image.convert("RGB")

    tensor = transform(
        image
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        real_probability = (
            torch.sigmoid(
                output
            ).item()
        )

        ai_probability = (
            1.0
            - real_probability
        )

    return ai_probability


# ============================================================
# GEMINI
# ============================================================

print()
print("=" * 70)
print("GEMINI")
print("=" * 70)

gemini_images = [
    p
    for p in GEMINI_TEST_DIR.iterdir()
    if (
        p.is_file()
        and p.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]
    )
]

gemini_images = gemini_images[
    :IMAGES_PER_CLASS
]

gemini_scores = []

for path in gemini_images:

    image = Image.open(
        path
    )

    score = predict(
        image
    )

    gemini_scores.append(
        score
    )

gemini_scores = np.array(
    gemini_scores
)

print(
    f"Images: {len(gemini_scores)}"
)

print(
    f"Average AI probability: "
    f"{gemini_scores.mean() * 100:.2f}%"
)

print(
    f"Median AI probability: "
    f"{np.median(gemini_scores) * 100:.2f}%"
)

print(
    f"Detected AI @ 0.5: "
    f"{np.sum(gemini_scores >= 0.5)}/"
    f"{len(gemini_scores)}"
)


# ============================================================
# EXTERNAL DATASET
# ============================================================

results = {}


for generator_name, generator_id in GENERATOR_IDS.items():

    print()
    print("=" * 70)
    print(generator_name)
    print("=" * 70)

    dataset = load_dataset(
        EXTERNAL_DATASET,
        split="test",
        streaming=True,
    )

    scores = []

    true_labels = []

    total = 0

    for item in dataset:

        if item["generator"] != generator_id:
            continue

        image = (
            item["image"]
            .convert("RGB")
        )

        score = predict(
            image
        )

        scores.append(
            score
        )

        # AI = 1
        # Real = 0

        true_labels.append(
            0
            if generator_name == "Real"
            else 1
        )

        total += 1

        print(
            f"\rProcessed "
            f"{total}/{IMAGES_PER_CLASS}",
            end=""
        )

        if total >= IMAGES_PER_CLASS:
            break

    print()

    scores = np.array(
        scores
    )

    true_labels = np.array(
        true_labels
    )

    predictions = (
        scores >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    print(
        f"Accuracy @ 0.5: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average AI probability: "
        f"{scores.mean() * 100:.2f}%"
    )

    print(
        f"Median AI probability: "
        f"{np.median(scores) * 100:.2f}%"
    )


    # --------------------------------------------------------
    # AUC needs both classes.
    #
    # A single generator alone cannot produce ROC-AUC because
    # it contains only AI OR only real images.
    #
    # Store scores for combined analysis.
    # --------------------------------------------------------

    results[
        generator_name
    ] = {
        "scores": scores,
        "labels": true_labels,
        "accuracy": accuracy,
    }


# ============================================================
# COMBINED AUC
# ============================================================

print()
print("=" * 70)
print("COMBINED ROC-AUC")
print("=" * 70)


all_scores = []
all_labels = []


for name, result in results.items():

    all_scores.extend(
        result["scores"]
    )

    all_labels.extend(
        result["labels"]
    )


all_scores = np.array(
    all_scores
)

all_labels = np.array(
    all_labels
)


combined_auc = roc_auc_score(
    all_labels,
    all_scores
)


print(
    f"Combined ROC-AUC: "
    f"{combined_auc:.4f}"
)


# ============================================================
# CYCLEGAN VS REAL AUC
# ============================================================

if (
    "CycleGAN" in results
    and "Real" in results
):

    cycle_scores = results[
        "CycleGAN"
    ]["scores"]

    real_scores = results[
        "Real"
    ]["scores"]


    unseen_scores = np.concatenate([
        cycle_scores,
        real_scores
    ])


    unseen_labels = np.concatenate([
        np.ones(
            len(cycle_scores)
        ),
        np.zeros(
            len(real_scores)
        )
    ])


    unseen_auc = roc_auc_score(
        unseen_labels,
        unseen_scores
    )


    print()
    print(
        "=" * 70
    )

    print(
        "CYCLEGAN + REAL "
        "UNSEEN ROC-AUC"
    )

    print(
        "=" * 70
    )

    print(
        f"Unseen ROC-AUC: "
        f"{unseen_auc:.4f}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()
print(
    "Important: CycleGAN remains "
    "an unseen-generator evaluation."
)

print(
    "It was NOT included in the "
    "universal training dataset."
)

print()