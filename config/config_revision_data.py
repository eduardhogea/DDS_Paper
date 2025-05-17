import os
import sys
import numpy as np

# Adjust BASE_DIR to point to DDS_PAPER, the parent of the current script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths, assuming the structure you described
config_path = os.path.join(BASE_DIR, 'config', 'config_revision_data.py')
dataset_path = os.path.join(BASE_DIR, 'data_revision')
kaggle_path = os.path.join(dataset_path, 'kaggle')
csv_file = os.path.join(dataset_path, 'data_robust.csv')
preprocessor_file = os.path.join(BASE_DIR, 'preprocessor_revision.joblib')
train_path = os.path.join(dataset_path, 'train.csv')
val_path = os.path.join(dataset_path, 'val.csv')
csv_directory = os.path.join(dataset_path, 'csvs')
data_root_folder = kaggle_path
sequences_directory = os.path.join(csv_directory, 'sequences')
model_save_directory = os.path.join(dataset_path, 'models')
model_path = os.path.join(model_save_directory, 'model.h5')
results_path = os.path.join(BASE_DIR, 'results_kaggle', 'results.csv')
results_path_ltn = os.path.join(BASE_DIR, 'results_kaggle/')
processed_file_tracker = os.path.join(BASE_DIR, 'progress_revision', 'progress.txt')
# Data Processing
chunk_size = 100000  # Adjust the chunk size according to your memory limitations
sequence_length = 10  # Define your desired sequence length
num_features = 4  # Based on the original number of features before sequencing
processed_bases = set()  # Prepare a list of base names to avoid redundancy

# Model Training Parameters
batch_size = 4000
epochs = 25
patience = 200
learning_rate = 0.0001
lr_ltn = 0.0001
n_splits = 2
reg_value = 0.001
num_train_samples = 5000
num_test_samples = 1000
reg_type = "l1"
n_samples = num_train_samples - 1
num_classes = 2
buffer_size = 200
ltn_batch = 4000
S = 100

# lr * 0.3 for seq-10, lr * 0.7 for the rest

# Seed for reproducibility
np.random.seed(42)