from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torchvision import transforms

from app.ml.model import create_model


# ============================================================
# SIGNALSCOPE FINAL MODEL COMPARISON
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL PATHS
# ============================================================

PRODUCTION_MODEL = (
    BASE_DIR
    / "models"
    / "signalscope_model.pth"
)

UNIVERSAL_MODEL = (
    BASE_DIR
    / "models"
    / "signalscope_model_universal.pth"
)


# ============================================================
# GEMINI TEST SET
# ============================================================

GEMINI_TEST_DIR = (
    BASE_DIR
    / "dataset_gemini"
    / "test"
    / "ai"
)


# ============================================================
# EXTERNAL DATASET
# ============================================================

EXTERNAL_DATASET = (
    "TheKernel01/AIGC-Detection-Benchmark"
)

IMAGES_PER_GENERATOR = 100


# ============================================================
# GENERATOR IDs
# ============================================================

REAL_GENERATOR = 0

DALLE2_GENERATOR = 4

SDXL_GENERATOR = 11

# CycleGAN was not used in our selected
# Tiny-GenImage/TIGAS training data.
CYCLEGAN_GENERATOR = 3


# ============================================================
# IMAGE TRANSFORM
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

def load_model(model_path):

    print(
        f"\nLoading model: "
        f"{model_path.name}"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = create_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================

def predict_image(
    model,
    image
):

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

        # SignalScope model output
        # represents REAL probability.

        real_probability = (
            torch.sigmoid(
                output
            ).item()
        )

        # AI probability is inverse.

        ai_probability = (
            1.0
            - real_probability
        )

    predicted_ai = (
        ai_probability >= 0.5
    )

    return (
        ai_probability,
        predicted_ai
    )


# ============================================================
# GEMINI TEST
# ============================================================

def test_gemini(
    production_model,
    universal_model
):

    print()
    print("=" * 70)
    print("TEST 1: GEMINI / NANO BANANA")
    print("=" * 70)

    images = [
        path
        for path in GEMINI_TEST_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]
        )
    ]

    images = images[:IMAGES_PER_GENERATOR]

    print(
        f"Images: {len(images)}"
    )

    results = {}

    for name, model in [
        (
            "Production",
            production_model
        ),
        (
            "Universal",
            universal_model
        ),
    ]:

        correct = 0

        probabilities = []

        for image_path in images:

            image = Image.open(
                image_path
            )

            ai_probability, predicted_ai = (
                predict_image(
                    model,
                    image
                )
            )

            probabilities.append(
                ai_probability
            )

            if predicted_ai:
                correct += 1

        accuracy = (
            correct
            / len(images)
            * 100
        )

        average_probability = (
            np.mean(probabilities)
            * 100
        )

        results[name] = accuracy

        print()
        print(name)

        print(
            f"Correct: "
            f"{correct}/{len(images)}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            f"Average AI probability: "
            f"{average_probability:.2f}%"
        )

    return results


# ============================================================
# EXTERNAL GENERATOR TEST
# ============================================================

def test_external_generator(
    generator_id,
    generator_name,
    expected_ai,
    production_model,
    universal_model
):

    print()
    print("=" * 70)
    print(
        f"TEST: {generator_name}"
    )
    print("=" * 70)

    print(
        f"Generator ID: "
        f"{generator_id}"
    )

    dataset = load_dataset(
        EXTERNAL_DATASET,
        split="test",
        streaming=True,
    )

    models = {
        "Production": production_model,
        "Universal": universal_model,
    }

    correct = {
        "Production": 0,
        "Universal": 0,
    }

    probabilities = {
        "Production": [],
        "Universal": [],
    }

    total = 0

    for item in dataset:

        generator = item[
            "generator"
        ]

        if generator != generator_id:
            continue

        image = item[
            "image"
        ].convert("RGB")


        for model_name, model in models.items():

            ai_probability, predicted_ai = (
                predict_image(
                    model,
                    image
                )
            )

            probabilities[
                model_name
            ].append(
                ai_probability
            )


            is_correct = (
                predicted_ai
                == expected_ai
            )

            if is_correct:

                correct[
                    model_name
                ] += 1


        total += 1

        print(
            f"\rProcessed "
            f"{total}/{IMAGES_PER_GENERATOR}",
            end=""
        )

        if (
            total
            >= IMAGES_PER_GENERATOR
        ):
            break

    print()

    if total == 0:

        print(
            "ERROR: No images found."
        )

        return None


    results = {}


    for model_name in models:

        accuracy = (
            correct[model_name]
            / total
            * 100
        )

        average_probability = (
            np.mean(
                probabilities[
                    model_name
                ]
            )
            * 100
        )

        results[
            model_name
        ] = accuracy

        print()
        print(
            model_name
        )

        print(
            f"Correct: "
            f"{correct[model_name]}/{total}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

        print(
            f"Average AI probability: "
            f"{average_probability:.2f}%"
        )


    return results


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("SIGNALSCOPE FINAL MODEL COMPARISON")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

print()
print(
    "Models:"
)

print(
    f"Production: "
    f"{PRODUCTION_MODEL.name}"
)

print(
    f"Universal: "
    f"{UNIVERSAL_MODEL.name}"
)


# ============================================================
# LOAD MODELS
# ============================================================

production_model = load_model(
    PRODUCTION_MODEL
)

universal_model = load_model(
    UNIVERSAL_MODEL
)


# ============================================================
# RESULTS
# ============================================================

all_results = {}


# ------------------------------------------------------------
# Gemini
# ------------------------------------------------------------

all_results[
    "Gemini"
] = test_gemini(
    production_model,
    universal_model
)


# ------------------------------------------------------------
# DALL-E 2
# ------------------------------------------------------------

all_results[
    "DALL-E 2"
] = test_external_generator(
    DALLE2_GENERATOR,
    "DALL-E 2",
    True,
    production_model,
    universal_model
)


# ------------------------------------------------------------
# SDXL
# ------------------------------------------------------------

all_results[
    "SDXL"
] = test_external_generator(
    SDXL_GENERATOR,
    "SDXL",
    True,
    production_model,
    universal_model
)


# ------------------------------------------------------------
# REAL
# ------------------------------------------------------------

all_results[
    "Real"
] = test_external_generator(
    REAL_GENERATOR,
    "Real Images",
    False,
    production_model,
    universal_model
)


# ------------------------------------------------------------
# UNSEEN GENERATOR
# ------------------------------------------------------------

all_results[
    "CycleGAN (Unseen)"
] = test_external_generator(
    CYCLEGAN_GENERATOR,
    "CycleGAN (Unseen Generator)",
    True,
    production_model,
    universal_model
)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print()
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Dataset':25}"
    f"{'Production':15}"
    f"{'Universal':15}"
)

print("-" * 70)


for dataset_name, result in all_results.items():

    if result is None:
        continue

    production_accuracy = (
        result.get(
            "Production",
            0
        )
    )

    universal_accuracy = (
        result.get(
            "Universal",
            0
        )
    )

    print(
        f"{dataset_name:25}"
        f"{production_accuracy:>10.2f}%"
        f"{universal_accuracy:>14.2f}%"
    )


print()
print("=" * 70)
print("TESTING COMPLETE")
print("=" * 70)