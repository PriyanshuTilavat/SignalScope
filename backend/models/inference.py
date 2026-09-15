from pathlib import Path

import torch
from PIL import Image

from .model import create_model
from .preprocessing import preprocess_image


# ============================================================
# SignalScope Inference Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# Production model
#
# The expanded model has been copied to this filename:
#
# models/signalscope_model.pth
#
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "signalscope_model.pth"
)


# ============================================================
# SignalScope Inference Service
# ============================================================

class SignalScopeInference:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None


    # ========================================================
    # Load Model
    # ========================================================

    def load_model(self):

        # Check whether model exists
        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )


        print(
            f"Loading SignalScope model from: {MODEL_PATH}"
        )


        # Create model architecture
        self.model = create_model()


        # Load trained weights
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True,
        )


        # Load weights into model
        self.model.load_state_dict(checkpoint)


        # Move model to CPU/GPU
        self.model.to(self.device)


        # Evaluation mode
        self.model.eval()


        print(
            f"SignalScope model loaded successfully "
            f"on {self.device}"
        )


    # ========================================================
    # Predict Image
    # ========================================================

    def predict(self, image: Image.Image):

        # Load model if not already loaded
        if self.model is None:
            self.load_model()


        # ----------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------

        input_tensor = preprocess_image(image)

        input_tensor = input_tensor.to(
            self.device
        )


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        with torch.no_grad():

            output = self.model(
                input_tensor
            )


            # ------------------------------------------------
            # IMPORTANT LABEL MAPPING
            # ------------------------------------------------
            #
            # ImageFolder created:
            #
            # ai   = 0
            # real = 1
            #
            # Therefore:
            #
            # sigmoid(output)
            #
            # represents REAL probability.
            #
            # AI probability is:
            #
            # 1 - real_probability
            # ------------------------------------------------

            real_probability = (
                torch.sigmoid(output).item()
            )

            ai_probability = (
                1.0 - real_probability
            )


        # ----------------------------------------------------
        # Convert to percentages
        # ----------------------------------------------------

        ai_percent = round(
            ai_probability * 100,
            2
        )

        real_percent = round(
            real_probability * 100,
            2
        )


        # ----------------------------------------------------
        # Determine verdict
        # ----------------------------------------------------

        if ai_probability >= 0.5:

            verdict = "Likely AI-generated"

        else:

            verdict = "Likely real"


        # ----------------------------------------------------
        # Calculate confidence
        # ----------------------------------------------------

        confidence = (
            max(
                ai_probability,
                real_probability
            )
            * 100
        )


        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {

            "ai_probability":
                ai_probability,

            "real_probability":
                real_probability,

            "ai_percent":
                ai_percent,

            "real_percent":
                real_percent,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "verdict":
                verdict,
        }


# ============================================================
# Global Inference Service
# ============================================================

inference_service = SignalScopeInference()