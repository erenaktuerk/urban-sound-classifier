import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from src.config import MODEL_DIR, PROCESSED_DATA_DIR

def load_test_data(filename='features.csv'):
    """
    Load and split the processed feature dataset into training and test sets.

    Returns:
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): One-hot encoded test labels.
        y_test_raw (np.ndarray): Original string labels before encoding.
    """
    data_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df = pd.read_csv(data_path)

    # Split features and labels
    X = df.drop(columns=['label']).values
    y_raw = df['label'].values

    # Reshape for Conv1D
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Load the saved label encoder
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    label_encoder = joblib.load(le_path)
    y_encoded = label_encoder.transform(y_raw)
    y_categorical = to_categorical(y_encoded)

    # Use same stratified split as during training
    _, X_test, _, y_test = train_test_split(
        X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
    )
    _, _, _, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, stratify=y_categorical, random_state=42
    )

    return X_test, y_test, y_test_raw, label_encoder.classes_

def plot_confusion_matrix(cm, class_labels):
    """
    Plots the confusion matrix as a heatmap.

    Args:
        cm (np.ndarray): Confusion matrix.
        class_labels (list): List of class label names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def evaluate_model():
    """
    Load trained model and evaluate on the test set.
    Show classification report and confusion matrix.
    """
    print("Loading model and test data...")

    model_path = os.path.join(MODEL_DIR, "audio_classifier_model.h5")
    model = load_model(model_path)

    X_test, y_test, y_test_raw, class_labels = load_test_data()

    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("\nGenerating predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_labels)

if __name__ == "__main__":
    evaluate_model()