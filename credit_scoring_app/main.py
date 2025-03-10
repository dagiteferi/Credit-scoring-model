import sys
import os
import traceback
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from schemas import RawInputData
from models.predictor import load_model, predict

# Add the root directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Scoring Prediction API")

# Define MODEL_PATH with absolute path to the root models directory
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "RandomForest_best_model.pkl")

@app.get("/predict/info")
def predict_info():
    """Return information about the predict endpoint."""
    return {
        "message": "This endpoint accepts POST requests with a JSON payload matching the RawInputData schema. Use /docs for details."
    }

@app.on_event("startup")
async def startup_event():
    """Load the model when the app starts."""
    try:
        app.state.model = load_model(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {traceback.format_exc()}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.post("/predict")
async def predict_route(data: RawInputData):
    """API endpoint to predict credit score from raw input data."""
    try:
        model = app.state.model
        prediction = predict(model, data.dict())
        logger.info(f"Prediction successful: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Mount static files for the front-end
app.mount("/static", StaticFiles(directory="static"), name="static")