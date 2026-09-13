from pathlib import Path

import torch
from PIL import Image

from .model import create_model
from .preprocessing import preprocess_image


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "signalscope_model.pth"


class SignalScopeInference:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None

    def load_model(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        self.model = create_model()

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)

        self.model.eval()

    def predict(self, image: Image.Image):

        if self.model is None:
            self.load_model()

        input_tensor = preprocess_image(image)

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():

            output = self.model(input_tensor)

            # ImageFolder used:
            # ai = 0
            # real = 1
            #
            # Therefore sigmoid(output)
            # represents REAL probability.

            real_probability = torch.sigmoid(output).item()

            # AI probability is the opposite.
            ai_probability = 1.0 - real_probability

        ai_percent = round(
            ai_probability * 100,
            2
        )

        real_percent = round(
            real_probability * 100,
            2
        )

        if ai_probability >= 0.5:
            verdict = "Likely AI-generated"
        else:
            verdict = "Likely real"

        confidence = max(
            ai_probability,
            real_probability
        ) * 100

        return {
            "ai_probability": ai_probability,
            "real_probability": real_probability,
            "ai_percent": ai_percent,
            "real_percent": real_percent,
            "confidence": round(confidence, 2),
            "verdict": verdict,
        }


inference_service = SignalScopeInference()