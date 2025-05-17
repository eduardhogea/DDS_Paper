import os
import sys
import numpy as np

# Adjust BASE_DIR to point to DDS_PAPER, the parent of the current script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths, assuming the structure you described
config_path = os.path.join(BASE_DIR, 'config', 'config_uoc.py')
dataset_path = os.path.join(BASE_DIR, 'data_uoc')
PGB_path = os.path.join(dataset_path, 'PGB', 'PGB')
RGB_path = os.path.join(dataset_path, 'RGB', 'RGB')
csv_file = os.path.join(dataset_path, 'data_robust_uoc.csv')
preprocessor_file = os.path.join(BASE_DIR, 'preprocessor_uoc.joblib')
train_path = os.path.join(dataset_path, 'train_uoc.csv')
val_path = os.path.join(dataset_path, 'val_uoc.csv')
csv_directory = os.path.join(dataset_path)
data_root_folder = PGB_path
sequences_directory = os.path.join(csv_directory, 'output_sequences')
model_save_directory = os.path.join(dataset_path, 'models_uoc')
model_path = os.path.join(model_save_directory, 'model_uoc.h5')
results_path = os.path.join(BASE_DIR, 'results_uoc', 'results.csv')
results_path_ltn = os.path.join(BASE_DIR, 'results_uoc/')
processed_file_tracker = os.path.join(BASE_DIR, 'progress_uoc', 'progress.txt')
# Data Processing
chunk_size = 100000  # Adjust the chunk size according to your memory limitations
sequence_length = 20  # Define your desired sequence length
num_features = 1  # Based on the original number of features before sequencing
processed_bases = set()  # Prepare a list of base names to avoid redundancy

# Model Training Parameters
batch_size = 150
epochs = 500
patience = 200
learning_rate = 0.0001
lr_ltn = 0.0001
n_splits = 2
reg_value = 0.001
num_train_samples = 10
num_test_samples = 10
reg_type = "l1"
n_samples = num_train_samples - 1
num_classes = 9
buffer_size = 200
ltn_batch = 150
S = 1

# lr * 0.3 for seq-10, lr * 0.7 for the rest

# Seed for reproducibility
np.random.seed(42)