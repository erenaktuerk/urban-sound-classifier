from src.config import PROCESSED_DATA_DIR
from src.data_preprocessing import preprocess
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict
import os

def main():
    print("Step 1: Preprocessing data...")
    preprocess()  # Preprocess the data
    
    print("Step 2: Extracting features...")
    # Hier wird nun der Preprocessing-Schritt genutzt, um die Features zu extrahieren
    
    print("Step 3: Training model...")
    train_model()  # Model training
    
    print("Step 4: Evaluating model...")
    evaluate_model()  # Evaluate model
    
    print("Step 5: Predicting sample...")
    example_audio = "data/raw/example.wav"
    if os.path.exists(example_audio):
        predict(example_audio)
    else:
        print(f"No example file found at {example_audio}. Skipping prediction.")

if __name__ == "__main__":
    main()