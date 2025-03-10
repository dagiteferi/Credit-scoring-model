import sys
import os
import traceback
from fastapi import FastAPI, HTTPException

# Adjust sys.path to include the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from credit_scoring_app.schemas import RawInputData
from models.predictor import load_model, predict
from credit_scoring_app.config import logger

MODEL_PATH = os.path.join(ROOT_DIR, "models", "RandomForest_best_model.pkl")
app = FastAPI(title="Credit Scoring Prediction API")

@app.get("/predict/info")
async def predict_info():
    return {"message": "This endpoint accepts POST requests with a JSON payload matching the RawInputData schema. Use /docs for details."}

@app.on_event("startup")
async def startup_event():
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {traceback.format_exc()}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.post("/predict")
async def predict_route(data: RawInputData):
    try:
        logger.info(f"Received data: {data.dict()}")
        model = app.state.model
        if model is None:
            raise ValueError("Model not loaded")
        prediction = predict(model, data.dict())
        logger.info(f"Prediction successful: {prediction}")
        return prediction  # Returns {"prediction": 0 or 1, "rfms_score": float}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")