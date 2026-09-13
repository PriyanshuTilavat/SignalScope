from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from app.ml.inference import inference_service
from app.ml.gradcam import GradCAM


# ==========================================
# SignalScope Grad-CAM Test
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE_DIR
    / "dataset"
    / "val"
    / "ai"
    / "ai_00000.jpg"
)

OUTPUT_PATH = BASE_DIR / "gradcam_result.jpg"


print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")


# ==========================================
# Load model
# ==========================================

print("Loading SignalScope model...")

inference_service.load_model()

model = inference_service.model
device = inference_service.device


# ==========================================
# Generate Grad-CAM
# ==========================================

print("Generating Grad-CAM heatmap...")

gradcam = GradCAM(
    model,
    device
)

heatmap = gradcam.generate(image)


# ==========================================
# Prediction
# ==========================================

prediction = inference_service.predict(image)


print("\n==========================================")
print("SignalScope Grad-CAM Test")
print("==========================================")

print(
    f"AI Probability:   {prediction['ai_percent']}%"
)

print(
    f"Real Probability: {prediction['real_percent']}%"
)

print(
    f"Confidence:       {prediction['confidence']}%"
)

print(
    f"Verdict:          {prediction['verdict']}"
)


# ==========================================
# Create visualization
# ==========================================

plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)

plt.imshow(image)

plt.title("Original Image")

plt.axis("off")


# Heatmap
plt.subplot(1, 2, 2)

plt.imshow(image)

plt.imshow(
    heatmap,
    alpha=0.5,
    cmap="jet"
)

plt.title("Grad-CAM")

plt.axis("off")


plt.tight_layout()


# ==========================================
# Save result
# ==========================================

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

gradcam.close()


print("\n==========================================")
print("Grad-CAM generated successfully!")
print(f"Saved to: {OUTPUT_PATH}")
print("==========================================")