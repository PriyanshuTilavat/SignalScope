from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.ml.model import create_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = (
    BASE_DIR
    / "dataset_gemini"
    / "test"
    / "ai"
)

MODELS = {
    "Production": (
        BASE_DIR
        / "models"
        / "signalscope_model.pth"
    ),

    "Gemini Candidate": (
        BASE_DIR
        / "models"
        / "signalscope_model_gemini.pth"
    ),

    "Universal Candidate": (
        BASE_DIR
        / "models"
        / "signalscope_model_universal.pth"
    ),
}


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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
        ],
    ),
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(path):

    print(
        f"Loading: {path.name}"
    )

    model = create_model()

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# TEST MODEL
# ============================================================

def evaluate(model, name):

    images = [
        p
        for p in TEST_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            ]
        )
    ]

    correct = 0

    probabilities = []


    with torch.no_grad():

        for image_path in images:

            image = Image.open(
                image_path
            ).convert("RGB")


            tensor = transform(
                image
            )

            tensor = (
                tensor
                .unsqueeze(0)
                .to(device)
            )


            output = model(
                tensor
            )


            # Model output = REAL probability

            real_probability = (
                torch.sigmoid(
                    output
                ).item()
            )


            ai_probability = (
                1.0
                - real_probability
            )


            probabilities.append(
                ai_probability
            )


            # Gemini images are AI

            if ai_probability >= 0.5:

                correct += 1


    accuracy = (
        correct
        / len(images)
        * 100
    )


    average_probability = (
        sum(probabilities)
        / len(probabilities)
        * 100
    )


    print()
    print("=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Gemini test images : "
        f"{len(images)}"
    )

    print(
        f"Correctly detected : "
        f"{correct}/{len(images)}"
    )

    print(
        f"Gemini accuracy    : "
        f"{accuracy:.2f}%"
    )

    print(
        f"Average AI prob.   : "
        f"{average_probability:.2f}%"
    )

    return accuracy


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 65)
print("SIGNALSCOPE — GEMINI MODEL COMPARISON")
print("=" * 65)

print(
    f"Device: {device}"
)

print(
    f"Gemini test set: {TEST_DIR}"
)


results = {}


for name, path in MODELS.items():

    if not path.exists():

        print()
        print(
            f"WARNING: {name} model not found:"
        )

        print(path)

        continue


    model = load_model(
        path
    )


    accuracy = evaluate(
        model,
        name
    )


    results[name] = accuracy


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 65)
print("FINAL GEMINI COMPARISON")
print("=" * 65)


for name, accuracy in results.items():

    print(
        f"{name:25} "
        f"{accuracy:.2f}%"
    )


print()
print("=" * 65)

if "Universal Candidate" in results:

    production = results.get(
        "Production",
        0
    )

    universal = results[
        "Universal Candidate"
    ]

    improvement = (
        universal
        - production
    )

    print(
        "Universal vs Production:"
    )

    print(
        f"{improvement:+.2f} "
        f"percentage points"
    )

print("=" * 65)