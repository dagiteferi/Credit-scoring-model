import joblib

# Load the model
model_path = "C:\\Users\\HP\\Documents\\Dagii\\Credit-scoring-model\\models\\RandomForest_best_model.pkl"
model = joblib.load(model_path)

# Print the expected feature names and their order
if hasattr(model, 'feature_names_in_'):
    print("Expected feature names (in order):")
    print(model.feature_names_in_)
else:
    print("Model does not have feature_names_in_ attribute. Scikit-learn version might be too old.")