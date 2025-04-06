# main.py

from src.data_preprocessing import preprocess_data
from src.feature_extraction import extract_and_save_features
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict

import os

def main():
    # Step 1: Preprocess raw audio files
    print("Step 1: Preprocessing data...")
    preprocess_data()
    
    # Step 2: Extract features and save processed data
    print("Step 2: Extracting features...")
    extract_and_save_features()
    
    # Step 3: Train the model
    print("Step 3: Training model...")
    train_model()
    
    # Step 4: Evaluate the model
    print("Step 4: Evaluating model...")
    evaluate_model()
    
    # Step 5: Make a sample prediction
    print("Step 5: Predicting sample...")
    example_audio = "data/raw/example.wav"
    if os.path.exists(example_audio):
        predict(example_audio)
    else:
        print(f"No example file found at {example_audio}. Skipping prediction.")

if __name__ == "__main__":
    main()