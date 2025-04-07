import librosa
import os
import numpy as np
import pandas as pd
from src.config import N_MFCC, PROCESSED_DATA_DIR

def extract_features(audio):
    """
    Extract MFCCs from an audio signal.
    """
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=N_MFCC)
    return np.mean(mfcc.T, axis=0)

def extract_features_from_file(file_path):
    """
    Extract features from an audio file.
    """
    audio = librosa.load(file_path, sr=None)[0]
    return extract_features(audio)

def extract_all_features(directory):
    """
    Extract features from all audio files in the directory.
    """
    features = []
    labels = []
    for foldername in os.listdir(directory):
        folder_path = os.path.join(directory, foldername)
        if os.path.isdir(folder_path):  # Only loop through directories
            for filename in os.listdir(folder_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(folder_path, filename)
                    features.append(extract_features_from_file(file_path))
                    labels.append(foldername)  # Use folder name as label
    return features, labels

def save_features(features, labels, filename='features.csv'):
    """
    Save extracted features and labels to a CSV file.
    """
    # Convert to a DataFrame
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    # Ensure the labels are string type
    labels_array = labels_array.astype(str)
    
    # Combine features and labels into a single DataFrame
    df = pd.DataFrame(features_array)
    df['label'] = labels_array
    
    # Save the DataFrame to a CSV file
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

def save_features_to_file(directory):
    """
    Extract all features from audio files and save them.
    """
    features, labels = extract_all_features(directory)
    save_features(features, labels)