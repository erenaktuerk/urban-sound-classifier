import os
import librosa
import numpy as np
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_audio_files_by_folder(directory):
    """
    Loads all .wav files grouped by subfolder from the given directory.

    Args:
        directory (str): Path to the root directory with fold-subfolders.

    Returns:
        dict: A dictionary where each key is a folder name, and each value is a list of .wav file paths.
    """
    print(f"Scanning for audio files in: {directory}")
    audio_files_by_folder = {}

    for foldername in sorted(os.listdir(directory)):
        folder_path = os.path.join(directory, foldername)
        if os.path.isdir(folder_path):
            wav_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith('.wav')
            ]
            if wav_files:
                audio_files_by_folder[foldername] = wav_files

    print(f"  Found {len(audio_files_by_folder)} folders with audio files.")
    return audio_files_by_folder

def preprocess_audio(file_path):
    """
    Loads and normalizes an audio file.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        tuple: Normalized audio signal and sampling rate.
    """
    audio, sr = librosa.load(file_path, sr=None)
    if np.max(np.abs(audio)) != 0:
        audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
    return audio, sr

def extract_features(audio, sr):
    """
    Extracts MFCC features from the audio signal.

    Args:
        audio (np.ndarray): Normalized audio signal.
        sr (int): Sampling rate of the audio.

    Returns:
        np.ndarray: MFCC feature vector (mean of MFCCs over time).
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_mels=40, fmax=sr//2)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Average over time frames
    return mfccs_mean

def save_processed_data(features, labels, filename='features.csv'):
    """
    Saves features and labels into a CSV file.

    Args:
        features (list of np.ndarray): Extracted features.
        labels (list of str): Corresponding labels.
        filename (str): Output filename.
    """
    print(f"Saving processed dataset to: {filename}")
    features_array = np.array([f.flatten() for f in features])
    df = pd.DataFrame(features_array)
    df['label'] = labels
    output_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"  -> Data successfully saved to {output_path}")

def preprocess():
    """
    Main preprocessing pipeline.
    """
    print("Starting full audio preprocessing...")

    audio_folders = load_audio_files_by_folder(RAW_DATA_DIR)
    all_features = []
    all_labels = []

    for folder, file_list in audio_folders.items():
        print(f"Processing folder: {folder} ({len(file_list)} files)...")

        folder_features = []
        for file_path in file_list:
            audio, sr = preprocess_audio(file_path)
            features = extract_features(audio, sr)
            folder_features.append(features)
            all_labels.append(folder)

        all_features.extend(folder_features)
        print(f"  -> Finished processing {len(file_list)} files from '{folder}'.")

    save_processed_data(all_features, all_labels)
    print("All folders processed successfully.")
    return all_features, all_labels

if __name__ == "__main__":
    preprocess()