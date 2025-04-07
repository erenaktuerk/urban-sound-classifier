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
    print(f"Loading processed data from {filename}...")
    data_path = os.path.join(PROCESSED_DATA_DIR, filename)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if 'label' not in df.columns:
        raise ValueError("'label' column not found in the data")
    
    X = df.drop(columns=['label']).values
    y = df['label'].values
    
    print(f"Data loaded successfully. Features shape: {X.shape}, Labels shape: {y.shape}")
    
    return X, y

def train_model():
    print("Starting model training...")
    features_file = os.path.join(PROCESSED_DATA_DIR, "features.csv")
    
    if not os.path.exists(features_file):
        print("No preprocessed data found, starting preprocessing...")
        features, labels = preprocess()  # Preprocess the data if necessary
        df = pd.DataFrame(features)
        df['label'] = labels
        df.to_csv(features_file, index=False)
        print(f"Processed data saved to {features_file}")
    else:
        print("Preprocessed data found, loading from CSV...")
        features, labels = load_processed_data()

    X = np.array(features)
    y = np.array(labels)

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Features or labels array is empty. Check data preprocessing.")
    
    X = X.reshape((X.shape[0], X.shape[1], 1))  # Shape: (batch_size, num_features, 1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Encoded labels shape: {y_encoded.shape}")

    if len(np.unique(y_encoded)) == 0:
        raise ValueError("Encoded labels are empty. Check the data.")
    
    y_categorical = to_categorical(y_encoded)
    print(f"Categorical labels shape: {y_categorical.shape}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
    )

    model = build_model(input_shape=X.shape[1:], num_classes=y_categorical.shape[1])
    model.summary()

    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val)
    )

    model.save(os.path.join(MODEL_DIR, "best_model.h5"))
    print("Training complete. Model and label encoder saved.")

if __name__ == "__main__":
    train_model()