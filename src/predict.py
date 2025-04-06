import numpy as np
from keras.models import load_model
from src.feature_extraction import extract_features_from_file
from src.config import MODEL_DIR

import os

def predict(file_path):
    """
    Loads a saved model and predicts the class for a given audio file.
    """
    print(f"Loading model from {MODEL_DIR}...")
    model_path = os.path.join(MODEL_DIR, "best_model.h5")
    model = load_model(model_path)
    
    print(f"Extracting features from {file_path}...")
    features = extract_features_from_file(file_path)
    
    # Ensure correct input shape
    input_data = np.expand_dims(features, axis=0)  # batch dimension
    input_data = np.expand_dims(input_data, axis=-1)  # channel dimension if needed
    
    print("Predicting...")
    prediction = model.predict(input_data)
    
    predicted_class = int(prediction[0][0] > 0.5)
    print(f"Predicted class: {predicted_class}")
    return predicted_class

if __name__ == "__main__":
    # Example prediction
    example_file = "data/raw/example.wav"
    predict(example_file)