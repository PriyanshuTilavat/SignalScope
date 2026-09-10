from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction import router as prediction_router


app = FastAPI(
    title="SignalScope API",
    description="AI-generated image detection backend",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "SignalScope API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "SignalScope Backend"
    }


app.include_router(prediction_router)