# config.py
import logging
import os

# Path to the saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "RandomForest_best_model.pkl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)