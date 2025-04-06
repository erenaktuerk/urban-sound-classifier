import os
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src.data_preprocessing import preprocess
from src.model import build_model
from src.config import BATCH_SIZE, EPOCHS, MODEL_DIR, PROCESSED_DATA_DIR

def load_processed_data(filename='features.csv'):
    """
    Load preprocessed features and labels from the CSV file.
    
    Args:
        filename (str): The name of the CSV file to load.
    
    Returns:
        tuple: Extracted features (X) and labels (y).
    """
    print(f"Loading processed data from {filename}...")
    data_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df = pd.read_csv(data_path)
    
    # Split the data into features and labels
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    return X, y

def train_model():
    """
    Load preprocessed data or preprocess the raw data if necessary,
    then train a TensorFlow model using MFCC features.
    """
    print("Starting model training...")

    # Check if features.csv exists
    features_file = os.path.join(PROCESSED_DATA_DIR, "features.csv")
    
    if not os.path.exists(features_file):
        print("No preprocessed data found, starting preprocessing...")
        features, labels = preprocess()  # Preprocess the data if necessary
        
        # Save the processed features to CSV for future use
        df = pd.DataFrame(features)
        df['label'] = labels
        df.to_csv(features_file, index=False)
        print(f"Processed data saved to {features_file}")
    else:
        print("Preprocessed data found, loading from CSV...")
        features, labels = load_processed_data()

    # Convert features and labels to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Reshape X for Conv1D (add an extra dimension for channels)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Shape: (batch_size, num_features, 1)

    # Encode class labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    # Optional: Save the label encoder for inference later
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Step 3: Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
    )

    # Step 4: Build and train model
    model = build_model(input_shape=X.shape[1:], num_classes=y_categorical.shape[1])
    model.summary()

    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )

    # Optional: Save the trained model
    model.save(os.path.join(MODEL_DIR, "audio_classifier_model.h5"))

    print("Training complete. Model and label encoder saved.")

if __name__ == "__main__":
    train_model()