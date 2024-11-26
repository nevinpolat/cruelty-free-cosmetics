from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)

# Define the path to the models directory
MODEL_DIR = 'models'

# Function to load a Pickle model
def load_pickle_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Successfully loaded model: {model_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file '{model_path}' not found in '{MODEL_DIR}' directory.")
        return None
    except Exception as e:
        logging.error(f"Error loading model '{model_path}': {str(e)}")
        return None

# Ensure the models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    logging.info(f"Created models directory at '{MODEL_DIR}'.")

# Load feature names
feature_names = []
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
try:
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    logging.info("Feature names loaded successfully.")
except FileNotFoundError:
    logging.error(f"'feature_names.pkl' not found in '{MODEL_DIR}'.")
except Exception as e:
    logging.error(f"Error loading 'feature_names.pkl': {str(e)}")

# Load all models into a dictionary for easy access
models = {}
model_files = [
    'linear_regression_model.pkl',
    'decision_tree_model.pkl',
    'random_forest_model.pkl',
    'xgboost_model_1.pkl',
    'xgboost_model_2.pkl',
    'xgboost_model_3.pkl',
    'tuned_random_forest_model.pkl',
    'tuned_xgboost_model.pkl'
]

for model_file in model_files:
    model_name = model_file.replace('.pkl', '')
    model_path = os.path.join(MODEL_DIR, model_file)
    model = load_pickle_model(model_path)
    if model:
        models[model_name] = model

# Helper function to make predictions with validation
def make_prediction(model, data):
    try:
        # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Log the received feature names
        logging.info(f"Received features: {list(input_df.columns)}")
        
        # Check if input features match the model's expected features (order-insensitive)
        if set(input_df.columns) != set(feature_names):
            missing_features = set(feature_names) - set(input_df.columns)
            extra_features = set(input_df.columns) - set(feature_names)
            error_messages = []
            if missing_features:
                error_messages.append(f"Missing features: {', '.join(missing_features)}")
            if extra_features:
                error_messages.append(f"Unexpected features: {', '.join(extra_features)}")
            raise ValueError("; ".join(error_messages))
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Return prediction as a list
        return prediction.tolist()
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return str(e)

# Route to get feature names (for verification)
@app.route('/features', methods=['GET'])
def get_features():
    if feature_names:
        return jsonify({'feature_names': feature_names})
    else:
        return jsonify({'error': "Feature names not found."}), 500

# Generic predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint that accepts a model name and input features.
    Expects JSON in the format:
    {
        "model_name": "random_forest_model",
        "input": {
            "CFC_Lag3": 24.0,
            "Save_Ralph_Campaign": 0,
            "cruelty_free_cosmetics": 27,
            "elf": 13,
            "week": 50
        }
    }
    """
    try:
        data = request.get_json(force=True)
        
        # Extract model name and input
        model_name = data.get('model_name')
        input_data = data.get('input')
        
        # Validate presence of model_name and input
        if not model_name or not input_data:
            return jsonify({'error': "Please provide both 'model_name' and 'input' in the JSON payload."}), 400
        
        # Check if the specified model exists
        if model_name not in models:
            available_models = ', '.join(models.keys())
            return jsonify({'error': f"Model '{model_name}' not found. Available models: {available_models}"}), 400
        
        # Make prediction
        prediction = make_prediction(models[model_name], input_data)
        
        # Check if prediction was successful
        if isinstance(prediction, list):
            return jsonify({'prediction': prediction})
        else:
            # If prediction is not a list, it's an error message
            return jsonify({'error': prediction}), 400
    except Exception as e:
        # Catch any unexpected errors and return them
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Home route
@app.route('/', methods=['GET'])
def home():
    return "Machine Learning Model Deployment with Flask!"

if __name__ == '__main__':
    app.run(debug=True)
