import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from code.util import extract_speed_from_filename
from code.data_scaling import load_and_scale_data
import config
import joblib

def create_sequences(df, sequence_length):
    sequences = []
    labels = []
    fault_types = df['Fault'].unique()

    for fault in fault_types:
        df_fault = df[df['Fault'] == fault]
        X = df_fault.drop('Fault', axis=1).values
        y = df_fault['Fault'].iloc[0]  # Updated to use iloc for consistency
        
        for i in range(len(df_fault) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
            labels.append(fault)  # Keep the fault type as is
    
    return np.array(sequences), np.array(labels)


def save_sequences(input_directory, output_directory, sequence_length):
    """
    Generates sequences and saves them as NumPy files, one for sequences and one for labels.
    
    Parameters:
    - input_directory: The directory with the original, scaled data files.
    - output_directory: The directory where the NumPy sequence files will be saved.
    - sequence_length: The number of consecutive samples in each sequence.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for file_name in tqdm(os.listdir(input_directory), desc="Generating sequences"):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_directory, file_name))
            sequences, labels = create_sequences(df, sequence_length)
            
            # File names for sequences and labels
            base_name = os.path.splitext(file_name)[0]
            sequences_file_path = os.path.join(output_directory, f"{base_name}_sequences.npy")
            labels_file_path = os.path.join(output_directory, f"{base_name}_labels.npy")
            
            # Save sequences and labels
            np.save(sequences_file_path, sequences)
            np.save(labels_file_path, labels)

def add_speed_feature_and_save(input_directory, output_directory, sequence_length):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for file_name in tqdm(os.listdir(input_directory), desc="Processing files"):
        if file_name.endswith('.csv'):
            speed = extract_speed_from_filename(file_name)
            df = pd.read_csv(os.path.join(input_directory, file_name))
            df['Speed'] = speed  # Add speed as a new column
            
            sequences, labels = create_sequences(df, sequence_length)
            
            base_name = os.path.splitext(file_name)[0]
            sequences_file_path = os.path.join(output_directory, f"{base_name}_sequences.npy")
            labels_file_path = os.path.join(output_directory, f"{base_name}_labels.npy")
            
            np.save(sequences_file_path, sequences)
            np.save(labels_file_path, labels)
            
            
            

# Iterate over your dataset files
for root, dirs, files in os.walk(config.csv_directory):
    for file in sorted(files):
        if file.endswith('.csv') and not file.endswith('_scaled.csv'):  # Process only unscaled .csv files
            csv_path = os.path.join(root, file)
            if 'train' in file:
                # Handle training data
                scaler_path = os.path.join(root, 'scaler_' + file.replace('.csv', '.joblib'))
                scaled_train_df = load_and_scale_data(csv_path, save_scaler_path=scaler_path)
                # Save the scaled training data
                scaled_csv_path = csv_path.replace('.csv', '_scaled.csv')
                scaled_train_df.to_csv(scaled_csv_path, index=False)
            elif 'test' in file:
                # Handle testing data
                scaler_path = os.path.join(root, 'scaler_' + file.replace('_test.csv', '_train.joblib'))
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                scaled_test_df = load_and_scale_data(csv_path, scaler=scaler)
                # Save the scaled testing data
                scaled_csv_path = csv_path.replace('.csv', '_scaled.csv')
                scaled_test_df.to_csv(scaled_csv_path, index=False)

            # Delete the original unscaled .csv file
            os.remove(csv_path)
            
#create sequences
save_sequences(config.csv_directory, config.sequences_directory, config.sequence_length)