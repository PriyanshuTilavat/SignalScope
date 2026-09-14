from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score

from app.ml.model import create_model


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PRODUCTION_PATH = (
    BASE_DIR / "models" / "signalscope_model.pth"
)

UNIVERSAL_PATH = (
    BASE_DIR / "models" / "signalscope_model_universal.pth"
)

GEMINI_DIR = (
    BASE_DIR / "dataset_gemini" / "test" / "ai"
)

DATASET_NAME = (
    "TheKernel01/AIGC-Detection-Benchmark"
)

N = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(path):

    model = create_model()

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


print("=" * 70)
print("SIGNALSCOPE MAX-SCORE DETECTOR")
print("=" * 70)

production = load_model(
    PRODUCTION_PATH
)

universal = load_model(
    UNIVERSAL_PATH
)

print("Both models loaded.")


# ============================================================
# PREDICT
# ============================================================

def predict(model, image):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        output = model(tensor)

        # Model output = REAL probability
        real_probability = (
            torch.sigmoid(output).item()
        )

        ai_probability = (
            1.0 - real_probability
        )

    return ai_probability


# ============================================================
# LOAD EXTERNAL DATA
# ============================================================

def load_generator(
    generator_id,
    name,
    label
):

    print()
    print(
        f"Loading {name}..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        streaming=True
    )

    samples = []

    count = 0

    for item in dataset:

        if item["generator"] != generator_id:
            continue

        image = (
            item["image"]
            .convert("RGB")
        )

        samples.append(
            (image, label)
        )

        count += 1

        print(
            f"\r{count}/{N}",
            end=""
        )

        if count >= N:
            break

    print()

    return samples


# ============================================================
# GEMINI
# ============================================================

gemini = []

paths = [
    p for p in GEMINI_DIR.iterdir()
    if (
        p.is_file()
        and p.suffix.lower()
        in [".jpg", ".jpeg", ".png", ".webp"]
    )
]

for path in paths[:N]:

    image = (
        Image.open(path)
        .convert("RGB")
    )

    gemini.append(
        (image, 1)
    )


# ============================================================
# OTHER DATASETS
# ============================================================

dalle2 = load_generator(
    4,
    "DALL-E 2",
    1
)

sdxl = load_generator(
    11,
    "SDXL",
    1
)

real = load_generator(
    0,
    "Real",
    0
)

cyclegan = load_generator(
    3,
    "CycleGAN",
    1
)


datasets = {
    "Gemini": gemini,
    "DALL-E 2": dalle2,
    "SDXL": sdxl,
    "Real": real,
    "CycleGAN": cyclegan
}


# ============================================================
# EVALUATION
# ============================================================

all_scores = []
all_labels = []

results = {}


for name, samples in datasets.items():

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    production_scores = []
    universal_scores = []
    max_scores = []
    labels = []

    for image, label in samples:

        p = predict(
            production,
            image
        )

        u = predict(
            universal,
            image
        )

        m = max(p, u)

        production_scores.append(p)
        universal_scores.append(u)
        max_scores.append(m)
        labels.append(label)


    production_scores = np.array(
        production_scores
    )

    universal_scores = np.array(
        universal_scores
    )

    max_scores = np.array(
        max_scores
    )

    labels = np.array(labels)


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    predictions = (
        max_scores >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        f"Max detector accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Average max AI probability: "
        f"{max_scores.mean() * 100:.2f}%"
    )

    print(
        f"Detected AI: "
        f"{predictions.sum()}/{len(predictions)}"
    )


    results[name] = {
        "scores": max_scores,
        "labels": labels
    }


# ============================================================
# COMBINED AUC
# ============================================================

scores = np.concatenate([
    results["Gemini"]["scores"],
    results["DALL-E 2"]["scores"],
    results["SDXL"]["scores"],
    results["Real"]["scores"]
])

labels = np.concatenate([
    results["Gemini"]["labels"],
    results["DALL-E 2"]["labels"],
    results["SDXL"]["labels"],
    results["Real"]["labels"]
])


combined_auc = roc_auc_score(
    labels,
    scores
)


# ============================================================
# UNSEEN AUC
# ============================================================

cycle_scores = results[
    "CycleGAN"
]["scores"]

cycle_labels = results[
    "CycleGAN"
]["labels"]

real_scores = results[
    "Real"
]["scores"]

real_labels = results[
    "Real"
]["labels"]


unseen_scores = np.concatenate([
    cycle_scores,
    real_scores
])

unseen_labels = np.concatenate([
    cycle_labels,
    real_labels
])


unseen_auc = roc_auc_score(
    unseen_labels,
    unseen_scores
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("MAX DETECTOR FINAL RESULTS")
print("=" * 70)

print(
    f"Combined ROC-AUC : "
    f"{combined_auc:.4f}"
)

print(
    f"CycleGAN + Real "
    f"Unseen AUC      : "
    f"{unseen_auc:.4f}"
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)