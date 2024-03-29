import numpy as np

# Paths
dataset_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU'
PGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB/PGB'
RGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/RGB/RGB'
csv_file = '/home/ubuntu/dds_paper/DDS_Paper/data/data_robust.csv'
preprocessor_file = 'preprocessor.joblib'
train_path = '/home/ubuntu/dds_paper/DDS_Paper/data/train.csv'
val_path = '/home/ubuntu/dds_paper/DDS_Paper/data/val.csv'
csv_directory = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs'
data_root_folder = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB/PGB'
sequences_directory = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs/sequences"
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/models"
model_path = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/models/model.h5"

# Data Processing
chunk_size = 100000  # Adjust the chunk size according to your memory limitations
sequence_length = 30  # Define your desired sequence length
num_features = 8  # Based on the original number of features before sequencing
processed_bases = set()  # Prepare a list of base names to avoid redundancy

# Model Training Parameters
batch_size = 512
epochs = 10
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
