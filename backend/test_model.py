from pathlib import Path

from PIL import Image

from app.ml.inference import inference_service


# Test with an AI image from the validation dataset
IMAGE_PATH = Path(
    r"E:\SignalScope\backend\dataset\val\ai\ai_00000.jpg"
)


print("Loading SignalScope model...")

image = Image.open(IMAGE_PATH)

result = inference_service.predict(image)

print("\n==============================")
print("SignalScope Prediction")
print("==============================")

print(f"AI Probability:   {result['ai_percent']}%")
print(f"Real Probability: {result['real_percent']}%")
print(f"Confidence:       {result['confidence']}%")
print(f"Verdict:          {result['verdict']}")