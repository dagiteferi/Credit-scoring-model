import sys
import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- Path Adjustments for Imports ---
# Get the root directory (parent of credit_scoring_app)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Now import your modules
from credit_scoring_app.schemas import RawInputData
from models.predictor import load_model, predict
from credit_scoring_app.config import logger

# --- Model Path Configuration ---
MODEL_PATH = os.path.join(ROOT_DIR, "models", "RandomForest_best_model.pkl")

app = FastAPI(title="Credit Scoring Prediction API")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://credit-scoring-frontend.onrender.com",
        "http://127.0.0.1:8080",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Startup Event (Fixed Version) ---
@app.on_event("startup")
async def startup_event():
    try:
        # Model loading
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # CORS confirmation (simplified logging)
        logger.info("CORS Middleware configured successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {traceback.format_exc()}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

# --- API Endpoints ---
@app.get("/predict/info")
async def predict_info():
    return {"message": "Use /predict/poor for poor credit (0) or /predict/good for good credit (1)."}

@app.post("/predict/poor")
async def predict_poor_credit(data: RawInputData):
    try:
        logger.info(f"Received data for poor credit prediction: {data.dict()}")
        model = app.state.model
        if model is None:
            raise ValueError("Model not loaded")
        prediction = predict(model, data.dict(), target_label="FALSE")
        logger.info(f"Prediction successful: {prediction}")
        return prediction
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/good")
async def predict_good_credit(data: RawInputData):
    try:
        logger.info(f"Received data for good credit prediction: {data.dict()}")
        model = app.state.model
        if model is None:
            raise ValueError("Model not loaded")
        prediction = predict(model, data.dict(), target_label="TRUE")
        logger.info(f"Prediction successful: {prediction}")
        return prediction
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")