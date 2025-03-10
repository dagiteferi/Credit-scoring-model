import sys
import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from schemas import RawInputData
from models.predictor import load_model, predict
from config import logger

# Define the root directory (one level up from credit_scoring_app/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add root directory to sys.path to find the models/ directory
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Debug: Print sys.path to verify
print("Initial sys.path:", sys.path)
print("Added root directory to sys.path:", ROOT_DIR)

# Define MODEL_PATH based on root directory
MODEL_PATH = os.path.join(ROOT_DIR, "models", "RandomForest_best_model.pkl")

# Initialize FastAPI app
app = FastAPI(title="Credit Scoring Prediction API")

# Mount static files (relative to credit_scoring_app/)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/predict/info")
async def predict_info():
    """Return information about the predict endpoint."""
    return {
        "message": "This endpoint accepts POST requests with a JSON payload matching the RawInputData schema. Use /docs for details."
    }

@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts."""
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
    """API endpoint to predict credit score from raw input data."""
    try:
        logger.info(f"Received data: {data.dict()}")
        model = app.state.model
        if model is None:
            raise ValueError("Model not loaded")
        prediction = predict(model, data.dict())
        logger.info(f"Prediction successful: {prediction}")
        return prediction
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")