import os
import sys
import numpy as np

# Adjust BASE_DIR to point to DDS_PAPER, the parent of the current script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths, assuming the structure you described
config_path = os.path.join(BASE_DIR, 'config', 'config.py')
dataset_path = os.path.join(BASE_DIR, 'data', 'DDS_Data_SEU')
PGB_path = os.path.join(dataset_path, 'PGB', 'PGB')
RGB_path = os.path.join(dataset_path, 'RGB', 'RGB')
csv_file = os.path.join(dataset_path, 'data_robust.csv')
preprocessor_file = os.path.join(BASE_DIR, 'preprocessor.joblib')
train_path = os.path.join(dataset_path, 'train.csv')
val_path = os.path.join(dataset_path, 'val.csv')
csv_directory = os.path.join(dataset_path, 'data', 'csvs')
data_root_folder = PGB_path
sequences_directory = os.path.join(csv_directory, 'sequences')
model_save_directory = os.path.join(dataset_path, 'models')
model_path = os.path.join(model_save_directory, 'model.h5')

# Data Processing
chunk_size = 100000  # Adjust the chunk size according to your memory limitations
sequence_length = 30  # Define your desired sequence length
num_features = 8  # Based on the original number of features before sequencing
processed_bases = set()  # Prepare a list of base names to avoid redundancy

# Model Training Parameters
batch_size = 512
epochs = 3
patience = 20
learning_rate = 0.0005
n_splits = 4
reg_value = 0.001
num_train_samples = 20000
num_test_samples = 4000
reg_type = "l1"
n_samples = 5000
num_classes = 9
buffer_size = 200
ltn_batch = 1000

# Seed for reproducibility
np.random.seed(45)
