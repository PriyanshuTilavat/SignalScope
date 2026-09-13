from fastapi import APIRouter, UploadFile, File, HTTPException

from PIL import Image
import io
import base64

import numpy as np
import matplotlib.cm as cm

from app.ml.inference import inference_service
from app.ml.gradcam import GradCAM


router = APIRouter(
    prefix="/api",
    tags=["Prediction"]
)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    # ==========================================
    # 1. Check file type
    # ==========================================

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # ==========================================
    # 2. Read uploaded file
    # ==========================================

    contents = await file.read()

    # ==========================================
    # 3. Open and validate image
    # ==========================================

    try:

        image = Image.open(
            io.BytesIO(contents)
        )

        image.verify()

        # verify() consumes the image.
        # Reopen it for actual processing.

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    # ==========================================
    # 4. Run AI prediction
    # ==========================================

    try:

        result = inference_service.predict(image)

    except FileNotFoundError:

        raise HTTPException(
            status_code=500,
            detail="Trained SignalScope model was not found."
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}"
        )

    # ==========================================
    # 5. Generate Grad-CAM
    # ==========================================

    try:

        # Make sure model is loaded
        if inference_service.model is None:
            inference_service.load_model()

        gradcam = GradCAM(
            inference_service.model,
            inference_service.device
        )

        heatmap = gradcam.generate(image)

        gradcam.close()

        # ==========================================
        # 6. Convert heatmap to image
        # ==========================================

        # Convert heatmap (0-1) to RGBA colors
        colored_heatmap = cm.jet(heatmap)

        colored_heatmap = (
            colored_heatmap[:, :, :3] * 255
        ).astype(np.uint8)

        heatmap_image = Image.fromarray(
            colored_heatmap
        )

        # Make sure heatmap and original
        # have the same size
        heatmap_image = heatmap_image.resize(
            image.size
        )

        # ==========================================
        # 7. Blend original + heatmap
        # ==========================================

        overlay = Image.blend(
            image,
            heatmap_image.convert("RGB"),
            alpha=0.45
        )

        # ==========================================
        # 8. Convert overlay to Base64
        # ==========================================

        buffer = io.BytesIO()

        overlay.save(
            buffer,
            format="JPEG",
            quality=90
        )

        heatmap_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        heatmap_url = (
            f"data:image/jpeg;base64,{heatmap_base64}"
        )

    except Exception as e:

        # Prediction should still work even if
        # Grad-CAM fails.
        heatmap_url = None

        print(
            f"Grad-CAM generation failed: {e}"
        )

    # ==========================================
    # 9. Return complete result
    # ==========================================

    return {

        "filename": file.filename,

        "message": "Image analyzed successfully",

        "status": "analysis_complete",

        # Prediction
        "ai_probability": result["ai_probability"],
        "real_probability": result["real_probability"],

        "ai_percent": result["ai_percent"],
        "real_percent": result["real_percent"],

        "confidence": result["confidence"],

        "verdict": result["verdict"],

        # Explainability
        "gradcam": heatmap_url,
    }