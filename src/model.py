from keras.models import Sequential
from keras.layers import Conv1D, GRU, Dropout, BatchNormalization, Dense, Activation
from src.config import INPUT_SHAPE

def build_model():
    """
    Bauen eines Keras-Modells für die Audio-Klassifikation
    """
    model = Sequential()
    
    # Conv1D Layer
    model.add(Conv1D(64, 5, padding='same', input_shape=INPUT_SHAPE))
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
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model