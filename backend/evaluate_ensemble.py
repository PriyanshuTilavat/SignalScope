from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score

from app.ml.model import create_model


# ============================================================
# SIGNALSCOPE ENSEMBLE EVALUATION
# Production Model + Universal Model
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# MODEL PATHS
# ============================================================

PRODUCTION_PATH = (
    BASE_DIR
    / "models"
    / "signalscope_model.pth"
)

UNIVERSAL_PATH = (
    BASE_DIR
    / "models"
    / "signalscope_model_universal.pth"
)

# ============================================================
# DATASET
# ============================================================

EXTERNAL_DATASET = (
    "TheKernel01/AIGC-Detection-Benchmark"
)

IMAGES_PER_GENERATOR = 100

# ============================================================
# GEMINI TEST
# ============================================================

GEMINI_TEST_DIR = (
    BASE_DIR
    / "dataset_gemini"
    / "test"
    / "ai"
)

# ============================================================
# GENERATOR IDs
# ============================================================

# AIGC Detection Benchmark
REAL_ID = 0
DALLE2_ID = 4
CYCLEGAN_ID = 3
SDXL_ID = 11


# ============================================================
# PREPROCESSING
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
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):

    print(
        f"Loading model: {model_path.name}"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{model_path}"
        )

    model = create_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# PREDICTION
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

        # IMPORTANT:
        # Model output represents REAL probability

        real_probability = (
            torch.sigmoid(output).item()
        )

        # AI probability is inverse

        ai_probability = (
            1.0 - real_probability
        )

    return ai_probability


# ============================================================
# LOAD EXTERNAL GENERATOR
# ============================================================

def load_generator(
    generator_id,
    name,
    expected_label
):

    print()
    print(
        f"Loading {name}..."
    )

    dataset = load_dataset(
        EXTERNAL_DATASET,
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
            (
                image,
                expected_label
            )
        )

        count += 1

        print(
            f"\rLoaded "
            f"{count}/{IMAGES_PER_GENERATOR}",
            end=""
        )

        if count >= IMAGES_PER_GENERATOR:
            break

    print()

    return samples


# ============================================================
# LOAD GEMINI
# ============================================================

def load_gemini():

    print()
    print("Loading Gemini test set...")

    if not GEMINI_TEST_DIR.exists():

        raise FileNotFoundError(
            f"Gemini test directory not found:\n"
            f"{GEMINI_TEST_DIR}"
        )

    paths = [
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

    paths = paths[
        :IMAGES_PER_GENERATOR
    ]

    samples = []

    for path in paths:

        image = (
            Image.open(path)
            .convert("RGB")
        )

        # Gemini is AI
        samples.append(
            (
                image,
                1
            )
        )

    print(
        f"Loaded Gemini images: "
        f"{len(samples)}"
    )

    return samples


# ============================================================
# SCORE DATASET WITH BOTH MODELS
# ============================================================

def score_dataset(
    name,
    samples,
    production_model,
    universal_model
):

    print()
    print("=" * 70)
    print(f"SCORING: {name}")
    print("=" * 70)

    production_scores = []
    universal_scores = []
    labels = []

    for index, (image, label) in enumerate(samples):

        production_score = predict(
            production_model,
            image
        )

        universal_score = predict(
            universal_model,
            image
        )

        production_scores.append(
            production_score
        )

        universal_scores.append(
            universal_score
        )

        labels.append(label)

        print(
            f"\rProcessed "
            f"{index + 1}/{len(samples)}",
            end=""
        )

    print()

    return {
        "production": np.array(
            production_scores
        ),

        "universal": np.array(
            universal_scores
        ),

        "labels": np.array(
            labels
        )
    }


# ============================================================
# ENSEMBLE
# ============================================================

def ensemble_probability(
    production_scores,
    universal_scores,
    production_weight
):

    universal_weight = (
        1.0 - production_weight
    )

    return (
        production_scores
        * production_weight
        +
        universal_scores
        * universal_weight
    )


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("SIGNALSCOPE ENSEMBLE MODEL EVALUATION")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print()
print("Production model:")
print(PRODUCTION_PATH)

print()
print("Universal model:")
print(UNIVERSAL_PATH)


# ============================================================
# LOAD MODELS
# ============================================================

production_model = load_model(
    PRODUCTION_PATH
)

universal_model = load_model(
    UNIVERSAL_PATH
)


# ============================================================
# LOAD DATA
# ============================================================

gemini_samples = load_gemini()

dalle2_samples = load_generator(
    DALLE2_ID,
    "DALL-E 2",
    1
)

sdxl_samples = load_generator(
    SDXL_ID,
    "SDXL",
    1
)

real_samples = load_generator(
    REAL_ID,
    "Real Images",
    0
)

cyclegan_samples = load_generator(
    CYCLEGAN_ID,
    "CycleGAN",
    1
)


# ============================================================
# SCORE DATA
# ============================================================

datasets = {

    "Gemini":
        gemini_samples,

    "DALL-E 2":
        dalle2_samples,

    "SDXL":
        sdxl_samples,

    "Real":
        real_samples,

    "CycleGAN":
        cyclegan_samples
}


all_data = {}


for name, samples in datasets.items():

    all_data[name] = score_dataset(
        name,
        samples,
        production_model,
        universal_model
    )


# ============================================================
# TEST DIFFERENT WEIGHTS
# ============================================================

weights = [
    0.75,
    0.60,
    0.50,
    0.40,
    0.25
]


print()
print("=" * 70)
print("ENSEMBLE WEIGHT COMPARISON")
print("=" * 70)


for production_weight in weights:

    universal_weight = (
        1.0
        - production_weight
    )

    print()
    print(
        f"Production "
        f"{production_weight * 100:.0f}% "
        f"+ Universal "
        f"{universal_weight * 100:.0f}%"
    )

    print("-" * 70)

    for name, data in all_data.items():

        scores = ensemble_probability(
            data["production"],
            data["universal"],
            production_weight
        )

        labels = data["labels"]

        predictions = (
            scores >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            labels,
            predictions
        )

        print(
            f"{name:15} "
            f"Accuracy = "
            f"{accuracy * 100:6.2f}%"
        )


# ============================================================
# COMBINED AUC
# ============================================================

print()
print("=" * 70)
print("COMBINED AI VS REAL ROC-AUC")
print("=" * 70)


combined_datasets = [
    "Gemini",
    "DALL-E 2",
    "SDXL",
    "Real"
]


for production_weight in weights:

    all_scores = []
    all_labels = []

    for name in combined_datasets:

        data = all_data[name]

        scores = ensemble_probability(
            data["production"],
            data["universal"],
            production_weight
        )

        all_scores.extend(
            scores
        )

        all_labels.extend(
            data["labels"]
        )

    all_scores = np.array(
        all_scores
    )

    all_labels = np.array(
        all_labels
    )

    auc = roc_auc_score(
        all_labels,
        all_scores
    )

    print(
        f"Production "
        f"{production_weight * 100:.0f}% + "
        f"Universal "
        f"{(1 - production_weight) * 100:.0f}% "
        f"-> AUC = {auc:.4f}"
    )


# ============================================================
# UNSEEN GENERATOR AUC
# ============================================================

print()
print("=" * 70)
print("CYCLEGAN + REAL UNSEEN ROC-AUC")
print("=" * 70)


for production_weight in weights:

    cycle_data = all_data[
        "CycleGAN"
    ]

    real_data = all_data[
        "Real"
    ]

    cycle_scores = ensemble_probability(
        cycle_data["production"],
        cycle_data["universal"],
        production_weight
    )

    real_scores = ensemble_probability(
        real_data["production"],
        real_data["universal"],
        production_weight
    )

    scores = np.concatenate([
        cycle_scores,
        real_scores
    ])

    labels = np.concatenate([
        np.ones(
            len(cycle_scores)
        ),
        np.zeros(
            len(real_scores)
        )
    ])

    auc = roc_auc_score(
        labels,
        scores
    )

    print(
        f"Production "
        f"{production_weight * 100:.0f}% + "
        f"Universal "
        f"{(1 - production_weight) * 100:.0f}% "
        f"-> Unseen AUC = {auc:.4f}"
    )


# ============================================================
# GEMINI DETAIL
# ============================================================

print()
print("=" * 70)
print("GEMINI ENSEMBLE DETAIL")
print("=" * 70)


gemini_data = all_data[
    "Gemini"
]


for production_weight in weights:

    scores = ensemble_probability(
        gemini_data["production"],
        gemini_data["universal"],
        production_weight
    )

    predictions = (
        scores >= 0.5
    )

    accuracy = (
        predictions.mean()
        * 100
    )

    average_probability = (
        scores.mean()
        * 100
    )

    print(
        f"Production "
        f"{production_weight * 100:.0f}% + "
        f"Universal "
        f"{(1 - production_weight) * 100:.0f}% "
        f"-> "
        f"Accuracy={accuracy:.2f}% "
        f"AvgAI={average_probability:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("ENSEMBLE EVALUATION COMPLETE")
print("=" * 70)

print()
print("No model files were modified.")

print(
    "Production model remains unchanged."
)

print(
    "Universal model remains unchanged."
)

print()