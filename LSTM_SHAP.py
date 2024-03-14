import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from joblib import dump, load
import ltn
import math
import wandb

dataset_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU'

PGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB/PGB'
RGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/RGB/RGB'

# Specify the CSV file path
csv_file = '/home/ubuntu/dds_paper/DDS_Paper/data/data_robust.csv'
preprocessor_file = 'preprocessor.joblib'

train_path = '/home/ubuntu/dds_paper/DDS_Paper/data/train.csv'
val_path = '/home/ubuntu/dds_paper/DDS_Paper/data/val.csv'

np.random.seed(45)

# Set the chunk size for reading the CSV
chunk_size = 100000  # Adjust the chunk size according to your memory limitations



def extract_fault(file_name):
    fault_mapping = {
        '0Health': 'HEA', '1Chipped': 'CTF', '2Miss': 'MTF', 
        '3Root': 'RCF', '4Surface': 'SWF', '5Ball': 'BWF', 
        '6Combination': 'CWF', '7Inner': 'IRF', '8Outer': 'ORF'
    }
    for key, value in fault_mapping.items():
        if key in file_name:
            return value
    return None

def make_csv_writer(csv_file):
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6', 'Channel7', 'Channel8', 'Fault'])
    return csv_writer

def generate_csv(output_directory, root_path, speed, experiment, files):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    filename_suffix = f"{speed}_{experiment}" if experiment else speed
    output_file_path = os.path.join(output_directory, f"PGB_{filename_suffix}.csv")
    rows_written = False  # Initialize a flag to track if any rows are written
    
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = make_csv_writer(csvfile)
        
        for file in tqdm(files, desc=f"Processing {filename_suffix}", unit="file"):
            fault_type = extract_fault(file)
            # Adjust the construction of file_path here
            file_path = os.path.join(root_path, file)  # Use root_path directly
            
            data = pd.read_csv(file_path, sep='\t', header=None, encoding='ISO-8859-1', skiprows=1, nrows=10000)
            if not data.empty:
                rows_written = True  # Set flag to True if we write any data rows
                for index, row in data.iterrows():
                    csv_writer.writerow(row[:8].tolist() + [fault_type])
                    
    # Check if any data rows were written, delete the CSV file if not
    if not rows_written:
        os.remove(output_file_path)
        print(f"No data was written. Deleted the file: {output_file_path}")


def process_pgb_data(data_root_folder, output_directory):
    for root, dirs, files in os.walk(data_root_folder):
        parts = root.split(os.sep)
        if 'Variable_speed' in parts or ('PGB' in parts and files):
            speed = parts[-2] if 'Variable_speed' in parts else parts[-1]
            experiment_dir = parts[-1] if 'Variable_speed' in parts else ''
            generate_csv(output_directory, root, speed, experiment_dir, files)

# Example usage remains the same:
data_root_folder = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB/PGB'
output_directory = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs'
process_pgb_data(data_root_folder, output_directory)
