import os
import librosa
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_audio_files(directory):
    """
    Load all audio files from the given directory
    and return them as a list.
    """
    audio_files = []
    for foldername in os.listdir(directory):
        folder_path = os.path.join(directory, foldername)
        if os.path.isdir(folder_path):  # Only loop through directories
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    audio_files.append(os.path.join(folder_path, filename))
    return audio_files

def preprocess_audio(file_path):
    """
    Load, normalize, and convert audio data to a suitable format.
    """
    audio, sr = librosa.load(file_path, sr=None)
    audio = audio / np.max(np.abs(audio))  # Normalize audio
    return audio

def save_processed_data(features, labels, filename='features.csv'):
    """
    Save extracted features and labels to a CSV file.
    """
    df = pd.DataFrame(features)
    df['label'] = labels  # Add labels to the dataframe
    df.to_csv(os.path.join(PROCESSED_DATA_DIR, filename), index=False)

def preprocess():
    """
    Main function for preprocessing: loading audio, extracting features, and saving data.
    """
    audio_files = load_audio_files(RAW_DATA_DIR)
    features = []
    labels = []
    
    for file in audio_files:
        audio = preprocess_audio(file)
        features.append(audio)
        label = os.path.basename(os.path.dirname(file))  # Extract label from folder name
        labels.append(label)
    
    save_processed_data(features, labels)
    return features, labels  # Return features and labels for training