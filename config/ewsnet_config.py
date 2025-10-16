import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data & results locations
sequences_directory = os.path.join(BASE_DIR, "data", "DDS_Data_SEU", "data", "csvs", "sequences")
results_root = os.path.join(BASE_DIR, "results", "ewsnet")
model_save_directory = os.path.join(BASE_DIR, "model_weights_ewsnet")

# Training hyperparameters
sequence_length = 20
batch_size = 512
epochs = 20
patience = 200
learning_rate = 1e-3
num_classes = 9
n_splits = 2

# Model-specific options
input_pad_length = 512  # minimum temporal length after padding
prefer_scaled = True
