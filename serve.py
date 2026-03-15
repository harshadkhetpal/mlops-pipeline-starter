# Author: Harshad Khetpal <harshadkhetpal@gmail.com>
"""FastAPI model serving script."""

import logging
import pickle
from pathlib import Path
from typing import List

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Serving API", version="1.0.0")

# Global model variable
model = None
model_metadata = {
    "model_type": "RandomForestClassifier",
    "framework": "scikit-learn",
    "version": "1.0.0"
}


class PredictionRequest(BaseModel):
    """Prediction request schema."""
    features: List[float]


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    prediction: int
    confidence: float


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model
    try:
        model_path = Path("models/model.pkl")
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info("Model loaded from local file")
        else:
            model = mlflow.pyfunc.load_model("models:/model/production")
            logger.info("Model loaded from MLflow registry")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model/info")
async def model_info():
    """Get model metadata."""
    return {
        "metadata": model_metadata,
        "status": "ready" if model else "not_loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import numpy as np
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        
        # Get prediction probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
        else:
            confidence = 1.0
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
