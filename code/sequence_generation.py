import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from tqdm import tqdm

from util import extract_speed_from_filename
from data_scaling import load_and_scale_data
import joblib

def load_sequences(sequence_file_path, label_file_path):
    sequences = np.load(sequence_file_path)
    labels = np.load(label_file_path)
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    return sequences, labels_encoded

def create_sequences(df, sequence_length):
    sequences = []
    labels = []
    fault_types = df['Fault'].unique()

    for fault in fault_types:
        df_fault = df[df['Fault'] == fault]
        X = df_fault.drop('Fault', axis=1).values
        y = df_fault['Fault'].iloc[0]
        
        for i in range(len(df_fault) - sequence_length + 1):
            sequences.append(X[i:i+sequence_length])
            labels.append(fault)
    
    return np.array(sequences), np.array(labels)


def save_sequences(input_directory, output_directory, sequence_length):
    """
    Generates sequences and saves them as numpy files, one for sequences and one for labels.
    
    Parameters:
    - input_directory: The directory with the original, scaled data files.
    - output_directory: The directory where the numpy sequence files will be saved.
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
            
            
            