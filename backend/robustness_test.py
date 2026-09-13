from pathlib import Path
import io

import torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.ml.model import create_model
from app.ml.preprocessing import preprocess_image


# ==========================================
# SignalScope Robustness Testing
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset" / "val"
MODEL_PATH = BASE_DIR / "models" / "signalscope_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGES_PER_CLASS = 100


print("\n==========================================")
print("SignalScope Robustness Testing")
print("==========================================")

print(f"Device: {DEVICE}")
print(f"Images per class: {IMAGES_PER_CLASS}")


# ==========================================
# Load dataset
# ==========================================

dataset = datasets.ImageFolder(
    DATASET_DIR
)

print("\nClasses:")
print(dataset.class_to_idx)


# ==========================================
# Select balanced images
# ==========================================

ai_indices = []
real_indices = []

ai_class = dataset.class_to_idx["ai"]
real_class = dataset.class_to_idx["real"]


for index, (_, label) in enumerate(dataset.samples):

    if label == ai_class and len(ai_indices) < IMAGES_PER_CLASS:
        ai_indices.append(index)

    elif label == real_class and len(real_indices) < IMAGES_PER_CLASS:
        real_indices.append(index)

    if (
        len(ai_indices) >= IMAGES_PER_CLASS
        and len(real_indices) >= IMAGES_PER_CLASS
    ):
        break


selected_indices = ai_indices + real_indices

print(f"Selected AI images:   {len(ai_indices)}")
print(f"Selected Real images: {len(real_indices)}")
print(f"Total images:         {len(selected_indices)}")


# ==========================================
# Load model
# ==========================================

print("\nLoading trained model...")

model = create_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ==========================================
# Image modifications
# ==========================================

def original_image(image):
    return image


def jpeg_compression(image):
    """
    Simulate social-media style JPEG compression.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=40
    )

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def blur_image(image):
    """
    Apply mild Gaussian blur.
    """

    return image.filter(
        ImageFilter.GaussianBlur(radius=1.5)
    )


def brightness_change(image):
    """
    Slightly reduce brightness.
    """

    enhancer = ImageEnhance.Brightness(image)

    return enhancer.enhance(0.75)


def resize_restore(image):
    """
    Reduce resolution and resize back.
    """

    width, height = image.size

    small_width = max(
        32,
        width // 2
    )

    small_height = max(
        32,
        height // 2
    )

    small = image.resize(
        (small_width, small_height),
        Image.Resampling.LANCZOS
    )

    return small.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )


# ==========================================
# Test configurations
# ==========================================

tests = {
    "Original": original_image,
    "JPEG Compression": jpeg_compression,
    "Blur": blur_image,
    "Brightness Reduction": brightness_change,
    "Resize + Restore": resize_restore,
}


# ==========================================
# Evaluation function
# ==========================================

def evaluate_test(test_name, modification):

    true_labels = []
    predictions = []
    probabilities = []

    print(
        f"\nTesting: {test_name}"
    )

    for count, index in enumerate(
        selected_indices,
        start=1
    ):

        image_path, label = dataset.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        # Apply modification
        image = modification(image)

        # Prepare tensor
        input_tensor = preprocess_image(
            image
        )

        input_tensor = input_tensor.to(
            DEVICE
        )

        # Model prediction
        with torch.no_grad():

            output = model(
                input_tensor
            )

            # Model output represents REAL probability
            real_probability = torch.sigmoid(
                output
            ).item()

            # Convert to AI probability
            ai_probability = (
                1.0 - real_probability
            )

        # Convert dataset label to AI target
        #
        # ai = 1
        # real = 0

        true_ai = (
            1
            if label == ai_class
            else 0
        )

        predicted_ai = (
            1
            if ai_probability >= 0.5
            else 0
        )

        true_labels.append(
            true_ai
        )

        predictions.append(
            predicted_ai
        )

        probabilities.append(
            ai_probability
        )

        print(
            f"\rProcessed {count}/{len(selected_indices)}",
            end=""
        )

    # ======================================
    # Metrics
    # ======================================

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    print("\n")

    print(
        f"Accuracy:  {accuracy * 100:.2f}%"
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

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ==========================================
# Run all robustness tests
# ==========================================

results = {}


for test_name, modification in tests.items():

    results[test_name] = evaluate_test(
        test_name,
        modification
    )


# ==========================================
# Final Summary
# ==========================================

print("\n==========================================")
print("ROBUSTNESS TEST SUMMARY")
print("==========================================")

print()

for test_name, metrics in results.items():

    print(
        f"{test_name:<22}"
        f"Accuracy: {metrics['accuracy'] * 100:6.2f}%   "
        f"F1: {metrics['f1'] * 100:6.2f}%"
    )


print("\n==========================================")
print("Robustness testing complete!")
print("==========================================")