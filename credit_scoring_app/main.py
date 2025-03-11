# credit_scoring_app/main.py
import sys
import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- Critical Path Fixes ---
# Get the absolute path to the repository root (dagiteferi-credit-scoring-model/)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path to credit_scoring_app/
REPO_ROOT = os.path.dirname(current_dir)                 # Path to dagiteferi-credit-scoring-model/
sys.path.insert(0, REPO_ROOT)  # Add repository root to Python path

# Debugging logs (check Render logs)
print(f"REPO_ROOT: {REPO_ROOT}")
print(f"Contents of REPO_ROOT: {os.listdir(REPO_ROOT)}")

# Now import modules
from credit_scoring_app.schemas import RawInputData
from models.predictor import load_model, predict
from credit_scoring_app.config import logger

# --- Model Path Configuration ---
MODEL_PATH = os.path.join(REPO_ROOT, "models", "RandomForest_best_model.pkl")

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

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    try:
        logger.info(f"Model path: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        app.state.model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {traceback.format_exc()}")
        raise RuntimeError(f"Startup error: {str(e)}")


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Credit Scoring API - Use /docs for documentation"}


@app.get("/predict/info")
async def predict_info():
    return {"message": "Use /predict/poor for poor credit (0) or /predict/good for good credit (1)."}


@app.post("/predict/poor")
async def predict_poor_credit(data: RawInputData):
    try:
        logger.info(f"Poor credit prediction request: {data.dict()}")
        prediction = predict(app.state.model, data.dict(), target_label="FALSE")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/good")
async def predict_good_credit(data: RawInputData):
    try:
        logger.info(f"Good credit prediction request: {data.dict()}")
        prediction = predict(app.state.model, data.dict(), target_label="TRUE")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
