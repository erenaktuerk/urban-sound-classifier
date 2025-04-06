from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.config import INPUT_SHAPE, NUM_CLASSES

from keras.models import Sequential
from keras.layers import Conv1D, GRU, Dropout, BatchNormalization, Dense, Activation

def build_model(input_shape, num_classes):
    """
    Build a Keras model for audio classification.
    :param input_shape: Shape of the input data (e.g., (16000, 1))
    :param num_classes: Number of output classes for classification
    :return: Compiled Keras model
    """
    model = Sequential()
    
    # Conv1D Layer
    model.add(Conv1D(64, 5, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # GRU Layer
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(GRU(128))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    # Dense Layer for Classification
    model.add(Dense(num_classes, activation='softmax'))  # For multi-class classification, use softmax
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model