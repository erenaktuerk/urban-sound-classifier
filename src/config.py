# Configuration file for audio ML project

# Directories
RAW_DATA_DIR = 'data/raw/'               # Path to the raw audio data (fold1, fold2, etc.)
PROCESSED_DATA_DIR = 'data/processed/'   # Path to save processed data (features)
MODEL_DIR = 'models/'                    # Path to save trained models

# Hyperparameters
LEARNING_RATE = 0.001                    # Learning rate for the optimizer
BATCH_SIZE = 32                          # Batch size for training
EPOCHS = 50                              # Number of epochs to train the model
INPUT_SHAPE = (13,)                      # Shape of input features (13 MFCCs per sample)
NUM_CLASSES = 10                         # Number of output classes

# Feature extraction parameters
N_MFCC = 13                              # Number of MFCC features to extract from each audio file

# Class labels
LABELS = [
    'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
    'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
]