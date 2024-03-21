import numpy as np

dataset_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU'

PGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB/PGB'
RGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/RGB/RGB'

# Specify the CSV file path
csv_file = '/home/ubuntu/dds_paper/DDS_Paper/data/data_robust.csv'
preprocessor_file = 'preprocessor.joblib'

train_path = '/home/ubuntu/dds_paper/DDS_Paper/data/train.csv'
val_path = '/home/ubuntu/dds_paper/DDS_Paper/data/val.csv'

random = np.random.seed(45)

# Set the chunk size for reading the CSV
chunk_size = 100000  # Adjust the chunk size according to your memory limitations


# Directory containing your scaled CSV files
csv_directory = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs'
# Define your dataset directory
data_root_folder = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB/PGB'

sequence_length = 30  # Example: Define your desired sequence length
sequences_directory = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs/sequences"
num_features = 8  # Based on the original number of features before sequencing

# Prepare a list of base names to avoid redundancy
processed_bases = set()

num_folds = 5
batch_size = 512
epochs = 5
patience = 20
learning_rate = 0.001
n_splits = 4
l2_reg = 0.001
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/models"
