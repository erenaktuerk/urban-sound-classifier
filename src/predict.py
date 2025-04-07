import os
import numpy as np
from keras.models import load_model
import joblib

from src.feature_extraction import extract_features_from_file
from src.config import MODEL_DIR

def predict(file_path):
    """
    Loads the trained model and label encoder, extracts features from the audio file,
    and returns the predicted class label.
    """
    print(f"Loading model from {MODEL_DIR}...")
    model_path = os.path.join(MODEL_DIR, "best_model.h5")
    model = load_model(model_path)

    # Load the label encoder to convert numeric predictions back to class labels
    label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    label_encoder = joblib.load(label_encoder_path)

    print(f"Extracting features from {file_path}...")
    features = extract_features_from_file(file_path)

    # Prepare input data shape (batch_size, num_features, 1)
    input_data = np.expand_dims(features, axis=0)  # Add batch dimension
    input_data = np.expand_dims(input_data, axis=-1)  # Add channel dimension

    print("Predicting...")
    prediction = model.predict(input_data)

    # Get the class index with the highest probability
    predicted_index = np.argmax(prediction[0])
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    print(f"Predicted class: {predicted_label}")
    return predicted_label

if __name__ == "__main__":
    # Example usage
    example_file = "data/raw/example.wav"
    predict(example_file)