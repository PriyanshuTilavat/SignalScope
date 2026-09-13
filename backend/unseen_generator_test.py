from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, classification_report

from app.ml.model import create_model
from app.ml.preprocessing import preprocess_image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "signalscope_model_expanded.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGES_PER_CLASS = 100

REAL_GENERATOR = 0
DALLE2_GENERATOR = 4
SDXL_GENERATOR = 11


# ============================================================
# HEADER
# ============================================================

print("\n==========================================")
print("SignalScope Unseen Generator Analysis")
print("==========================================")
print(f"Device: {DEVICE}")
print(f"Images per class: {IMAGES_PER_CLASS}")

print("\nTesting:")
print("  DALL-E 2")
print("  SDXL")
print("  Real Images")

print("\nThese AI generators were not used")
print("in the Tiny-GenImage training set.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SignalScope model...")

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


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_generator(
    generator_id,
    generator_name,
    expected_ai,
):
    print("\n------------------------------------------")
    print(f"Testing: {generator_name}")
    print("------------------------------------------")

    # New stream for every generator
    dataset = load_dataset(
        "TheKernel01/AIGC-Detection-Benchmark",
        split="test",
        streaming=True,
    )

    probabilities = []
    predictions = []
    true_labels = []

    total = 0

    for item in dataset:

        generator = item["generator"]

        if generator != generator_id:
            continue

        image = item["image"].convert("RGB")

        input_tensor = preprocess_image(image).to(DEVICE)

        with torch.no_grad():

            output = model(input_tensor)

            # Model output represents REAL probability
            real_probability = torch.sigmoid(output).item()

            # Therefore AI probability is the inverse
            ai_probability = 1.0 - real_probability

        predicted_ai = ai_probability >= 0.5

        probabilities.append(ai_probability)

        predictions.append(
            1 if predicted_ai else 0
        )

        true_labels.append(
            1 if expected_ai else 0
        )

        total += 1

        print(
            f"\rProcessed {total}/{IMAGES_PER_CLASS}",
            end=""
        )

        if total >= IMAGES_PER_CLASS:
            break

    print()

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if total == 0:

        print(
            f"ERROR: No matching images found for {generator_name}"
        )

        return None

    # Convert to numpy
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = np.mean(
        predictions == true_labels
    )

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    average_probability = np.mean(probabilities)
    minimum_probability = np.min(probabilities)
    maximum_probability = np.max(probabilities)
    median_probability = np.median(probabilities)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(f"\nGenerator: {generator_name}")

    print(f"Images tested: {total}")

    print(
        f"Correct: "
        f"{np.sum(predictions == true_labels)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nAI Probability Statistics")

    print(
        f"Average: "
        f"{average_probability * 100:.2f}%"
    )

    print(
        f"Minimum: "
        f"{minimum_probability * 100:.2f}%"
    )

    print(
        f"Maximum: "
        f"{maximum_probability * 100:.2f}%"
    )

    print(
        f"Median: "
        f"{median_probability * 100:.2f}%"
    )

    print("\nPredictions")

    print(
        f"Predicted AI: "
        f"{np.sum(predictions == 1)}"
    )

    print(
        f"Predicted Real: "
        f"{np.sum(predictions == 0)}"
    )

    print("\nConfusion Matrix")

    print(
        "              Predicted"
    )

    print(
        "              Real    AI"
    )

    print(
        f"Actual Real   "
        f"{cm[0][0]:4d}   "
        f"{cm[0][1]:4d}"
    )

    print(
        f"Actual AI     "
        f"{cm[1][0]:4d}   "
        f"{cm[1][1]:4d}"
    )

    return {
        "name": generator_name,
        "accuracy": accuracy,
        "average_probability": average_probability,
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "median_probability": median_probability,
        "confusion_matrix": cm,
    }


# ============================================================
# RUN TESTS
# ============================================================

dalle2_result = evaluate_generator(
    DALLE2_GENERATOR,
    "DALL-E 2",
    expected_ai=True,
)

sdxl_result = evaluate_generator(
    SDXL_GENERATOR,
    "SDXL",
    expected_ai=True,
)

real_result = evaluate_generator(
    REAL_GENERATOR,
    "Real Images",
    expected_ai=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("UNSEEN GENERATOR TEST SUMMARY")
print("==========================================")


results = [
    dalle2_result,
    sdxl_result,
    real_result,
]

for result in results:

    if result is None:
        continue

    print(
        f"\n{result['name']}"
    )

    print(
        f"Accuracy: "
        f"{result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Average AI Probability: "
        f"{result['average_probability'] * 100:.2f}%"
    )

    print(
        f"Median AI Probability: "
        f"{result['median_probability'] * 100:.2f}%"
    )


print("\n==========================================")
print("Unseen generator analysis complete!")
print("==========================================")