import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm
import os
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from joblib import dump, load
import ltn
import csv
import math
import wandb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import joblib  # For saving the scaler model
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import csv
import pandas as pd
import os
import pandas as pd
from collections import Counter

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


# Directory containing your scaled CSV files
csv_directory = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs'
# Define your dataset directory
data_root_folder = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB/PGB'

sequence_length = 30  # Example: Define your desired sequence length
sequences_directory = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs/sequences"
num_features = 8  # Based on the original number of features before sequencing
input_directory = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs"
output_directory = "/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs/sequences"
# Example call to save_sequences_as_csv (paths and sequence_length need to be defined)
# Example usage
directory = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs'

# def extract_fault(file_name):
#     fault_mapping = {
#         '0Health': 'HEA', '1Chipped': 'CTF', '2Miss': 'MTF', 
#         '3Root': 'RCF', '4Surface': 'SWF', '5Ball': 'BWF', 
#         '6Combination': 'CWF', '7Inner': 'IRF', '8Outer': 'ORF'
#     }
#     for key, value in fault_mapping.items():
#         if key in file_name:
#             return value
#     return None

# def make_csv_writer(csv_file):
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6', 'Channel7', 'Channel8', 'Fault'])
#     return csv_writer

# def generate_csv(output_directory, root_path, speed, experiment, files):
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
    
#     train_filename_suffix = f"{speed}_{experiment}_train" if experiment else f"{speed}_train"
#     test_filename_suffix = f"{speed}_{experiment}_test" if experiment else f"{speed}_test"
    
#     train_output_file_path = os.path.join(output_directory, f"PGB_{train_filename_suffix}.csv")
#     test_output_file_path = os.path.join(output_directory, f"PGB_{test_filename_suffix}.csv")
    
#     with open(train_output_file_path, 'w', newline='', encoding='utf-8') as train_csvfile, \
#         open(test_output_file_path, 'w', newline='', encoding='utf-8') as test_csvfile:
#         train_csv_writer = make_csv_writer(train_csvfile)
#         test_csv_writer = make_csv_writer(test_csvfile)
        
#         for file in tqdm(files, desc=f"Processing {speed} {experiment}", unit="file"):
#             fault_type = extract_fault(file)
#             # Only append 'speed' directory for non-variable speed cases
#             if experiment:
#                 file_path = os.path.join(root_path, file)  # Already includes 'Variable_speed/Experiment#'
#             else:
#                 file_path = os.path.join(root_path, file)  # 'root_path' already includes 'speed' directory
            
#             data = pd.read_csv(file_path, sep='\t', header=None, encoding='ISO-8859-1', skiprows=1, nrows=100000)
#             train_samples, test_samples = data.iloc[:80000, :], data.iloc[80000:100000, :]
            
#             for index, row in train_samples.iterrows():
#                 train_csv_writer.writerow(row[:8].tolist() + [fault_type])
            
#             for index, row in test_samples.iterrows():
#                 test_csv_writer.writerow(row[:8].tolist() + [fault_type])

# def process_pgb_data(data_root_folder, csv_directory):
#     for root, dirs, files in os.walk(data_root_folder):
#         parts = root.split(os.sep)
#         if 'Variable_speed' in parts:
#             speed = "Variable_speed"
#             experiment_dir = parts[-1]  # Get the last part as the experiment name
#             exp_files = [f for f in os.listdir(root) if f.endswith('.txt')]
#             # Pass the 'root' directly without modifying it for variable speed
#             generate_csv(csv_directory, root, speed, experiment_dir, exp_files)
#         elif 'PGB' in parts and files:
#             speed = parts[-1]  # Last part of 'root' is the speed directory
#             # For non-variable speed, pass the 'root' directly
#             generate_csv(csv_directory, root, speed, '', files)



# process_pgb_data(data_root_folder, output_directory)

# import os
# import pandas as pd
# from collections import Counter

# def overview_csv_files(directory):
#     data = []
#     all_faults = set()

#     for file in os.listdir(directory):
#         if file.endswith(".csv"):
#             file_path = os.path.join(directory, file)
#             df = pd.read_csv(file_path)

#             # Check if the CSV is empty (aside from the header)
#             if df.shape[0] == 0:
#                 # Delete the empty CSV file
#                 os.remove(file_path)
#                 print(f"Deleted empty file: {file_path}")
#                 continue  # Skip further processing for this file

#             num_samples = len(df)
#             fault_distribution = Counter(df['Fault'])
#             all_faults.update(fault_distribution.keys())
#             data.append({'File Name': file, 'Number of Samples': num_samples, **fault_distribution})

#     if not data:  # If no data has been gathered, exit the function
#         print("No data found.")
#         return

#     overview_df = pd.DataFrame(data)
#     for fault in all_faults:
#         if fault not in overview_df.columns:
#             overview_df[fault] = 0

#     cols = ['File Name', 'Number of Samples'] + sorted(all_faults)
#     overview_df = overview_df[cols]
#     overview_df.fillna(0, inplace=True)
#     overview_df.loc[:, 'Number of Samples':] = overview_df.loc[:, 'Number of Samples':].astype(int)

#     overview_df = overview_df.sort_values(by='File Name')
#     print(overview_df.to_string(index=False))

# overview_csv_files(directory)



# def load_and_scale_data(csv_path, scaler=None, save_scaler_path=None):
#     """
#     Loads data from a CSV file, scales the features (excluding the 'Fault' column), 
#     and returns the scaled DataFrame. Optionally saves the scaler model.
#     """
#     # Load the data
#     data = pd.read_csv(csv_path)
    
#     # Separate features and target
#     features = data.columns[:-1]  # Assuming the last column is the target
#     X = data[features]
#     y = data['Fault']

#     # Apply scaling
#     if scaler is None:
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#         if save_scaler_path:
#             joblib.dump(scaler, save_scaler_path)
#     else:
#         X_scaled = scaler.transform(X)
    
#     # Combine scaled features with target
#     scaled_df = pd.DataFrame(X_scaled, columns=features)
#     scaled_df['Fault'] = y
    
#     return scaled_df



# # Iterate over your dataset files
# for root, dirs, files in os.walk(csv_directory):
#     for file in sorted(files):
#         if file.endswith('.csv') and not file.endswith('_scaled.csv'):  # Process only unscaled .csv files
#             csv_path = os.path.join(root, file)
#             if 'train' in file:
#                 # Handle training data
#                 scaler_path = os.path.join(root, 'scaler_' + file.replace('.csv', '.joblib'))
#                 scaled_train_df = load_and_scale_data(csv_path, save_scaler_path=scaler_path)
#                 # Save the scaled training data
#                 scaled_csv_path = csv_path.replace('.csv', '_scaled.csv')
#                 scaled_train_df.to_csv(scaled_csv_path, index=False)
#             elif 'test' in file:
#                 # Handle testing data
#                 scaler_path = os.path.join(root, 'scaler_' + file.replace('_test.csv', '_train.joblib'))
#                 scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
#                 scaled_test_df = load_and_scale_data(csv_path, scaler=scaler)
#                 # Save the scaled testing data
#                 scaled_csv_path = csv_path.replace('.csv', '_scaled.csv')
#                 scaled_test_df.to_csv(scaled_csv_path, index=False)

#             # Delete the original unscaled .csv file
#             os.remove(csv_path)




# def create_sequences(df, sequence_length):
#     sequences = []
#     labels = []
#     fault_types = df['Fault'].unique()

#     for fault in fault_types:
#         df_fault = df[df['Fault'] == fault]
#         X = df_fault.drop('Fault', axis=1).values
#         y = df_fault['Fault'].iloc[0]  # Updated to use iloc for consistency
        
#         for i in range(len(df_fault) - sequence_length + 1):
#             sequences.append(X[i:i+sequence_length])
#             labels.append(fault)  # Keep the fault type as is
    
#     return np.array(sequences), np.array(labels)


# def save_sequences(input_directory, output_directory, sequence_length):
#     """
#     Generates sequences and saves them as NumPy files, one for sequences and one for labels.
    
#     Parameters:
#     - input_directory: The directory with the original, scaled data files.
#     - output_directory: The directory where the NumPy sequence files will be saved.
#     - sequence_length: The number of consecutive samples in each sequence.
#     """
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
    
#     for file_name in tqdm(os.listdir(input_directory), desc="Generating sequences"):
#         if file_name.endswith('.csv'):
#             df = pd.read_csv(os.path.join(input_directory, file_name))
#             sequences, labels = create_sequences(df, sequence_length)
            
#             # File names for sequences and labels
#             base_name = os.path.splitext(file_name)[0]
#             sequences_file_path = os.path.join(output_directory, f"{base_name}_sequences.npy")
#             labels_file_path = os.path.join(output_directory, f"{base_name}_labels.npy")
            
#             # Save sequences and labels
#             np.save(sequences_file_path, sequences)
#             np.save(labels_file_path, labels)


# save_sequences(input_directory, output_directory, sequence_length)

import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


def lr_schedule(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.1
    return lr

def load_sequences(sequence_file_path, label_file_path):
    sequences = np.load(sequence_file_path)
    labels = np.load(label_file_path)
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)
    return sequences, labels_onehot

def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(300, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(200),
        Dropout(0.2),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare a list of base names to avoid redundancy
processed_bases = set()

# Iterate through the directory to find matching train and test pairs
for file in sorted(os.listdir(sequences_directory)):
    if "_train_scaled_sequences.npy" in file:
        base_name = file.replace("_train_scaled_sequences.npy", "")
        if base_name in processed_bases:
            continue  # Skip if this set has already been processed

        train_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_sequences.npy")
        train_label_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_labels.npy")
        test_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_sequences.npy")
        test_label_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_labels.npy")

        # Check if corresponding test files exist
        if os.path.exists(test_sequence_file_path) and os.path.exists(test_label_file_path):
            print(f"Processing: {base_name}")
            X_train, y_train = load_sequences(train_sequence_file_path, train_label_file_path)
            X_test, y_test = load_sequences(test_sequence_file_path, test_label_file_path)

            num_classes = y_train.shape[1]
            input_shape = (sequence_length, num_features)
            
            model = create_model(input_shape, num_classes)
            lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)
            early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=2048, callbacks=[early_stopping, lr_scheduler], verbose=2)
            
            processed_bases.add(base_name)  # Mark this set as processed
