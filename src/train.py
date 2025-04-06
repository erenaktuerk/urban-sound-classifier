from src.data_preprocessing import preprocess
from src.model import build_model
from src.config import BATCH_SIZE, EPOCHS
import numpy as np

def train_model():
    """
    Function to train the model.
    """
    features, labels = preprocess()  # Preprocess the data to get features and labels
    
    x_train = np.array(features)  # Extracted features
    y_train = np.array(labels)    # Corresponding labels (e.g., classes)
    
    # Optional: Split the data into training and validation sets
    x_val = x_train[:100]   # Example, you could implement a proper split
    y_val = y_train[:100]
    
    model = build_model()
    model.summary()
    
    # Train the model
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val))

if __name__ == "__main__":
    train_model()