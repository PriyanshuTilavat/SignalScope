from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

router = APIRouter(
    prefix="/api",
    tags=["Prediction"]
)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    # Read uploaded file
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image."
        )

    return {
        "filename": file.filename,
        "message": "Image received successfully",
        "status": "ready_for_analysis"
    }