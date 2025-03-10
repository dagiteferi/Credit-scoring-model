
import sys
import os

# Add the root directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from schemas import RawInputData
from models.predictor import load_model, predict
from config import MODEL_PATH, logger

app = FastAPI(title="Credit Scoring Prediction API")

@app.get("/")  # New root endpoint
def read_root():
    """Return a welcome message for the API."""
    return {"message": "Welcome to the Credit Scoring Prediction API. Use POST /predict to make predictions. Docs available at /docs."}

@app.on_event("startup")
def startup_event():
    """Load the model when the app starts."""
    app.state.model = load_model(MODEL_PATH)

@app.post("/predict")
def predict_route(data: RawInputData):
    """API endpoint to predict credit score from raw input data."""
    try:
        model = app.state.model
        prediction = predict(model, data.dict())
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")