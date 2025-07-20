import os
import numpy as np
import pandas as pd
import scipy.io as scio

# Paths to the data files
timedomain_path = 'data_uoc/DataForClassification_TimeDomain.mat'

# Load the .mat file
dict_timedomain = scio.loadmat(timedomain_path)
X2 = dict_timedomain['AccTimeDomain']
time_domain_df = pd.DataFrame(X2)

# Define signal types and parameters
signal_types = ['healthy', 'missing', 'crack', 'spall',
                'chip5a', 'chip4a', 'chip3a', 'chip2a', 'chip1a']
UoC_class_num = len(signal_types)

data_len = 20        # Window size
window_len = 1000     # Step size for sliding window
START_index = [i * 104 for i in range(UoC_class_num)]
labels = [i for i in range(UoC_class_num)]

def process_fault_type(x, label):
    sequences = []
    labels_list = []
    for j in range(x.shape[1]):
        temp_one_column = x.iloc[:, j].copy(deep=True).values
        start_row = 0
        end_row = data_len
        while end_row <= len(temp_one_column):
            temp_x_sample = temp_one_column[start_row:end_row]
            sequences.append(temp_x_sample)
            labels_list.append(label)
            start_row += window_len
            end_row += window_len
    return sequences, labels_list

def collect_all_data():
    all_sequences = []
    all_labels = []
    for i in range(UoC_class_num):
        label = labels[i]
        total_start_column = START_index[label]
        total_end_column = total_start_column + 104
        temp_df = time_domain_df.iloc[:, total_start_column:total_end_column].copy(deep=True)
        sequences, labels_list = process_fault_type(temp_df, label)
        all_sequences.extend(sequences)
        all_labels.extend(labels_list)
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)
    return all_sequences, all_labels

def stratified_split(sequences, labels, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=test_size, random_state=random_state, stratify=labels)
    return X_train, X_test, y_train, y_test

def save_data(X_train, y_train, X_test, y_test, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Reshape sequences to be (num_samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    np.save(os.path.join(output_directory, "train_sequences.npy"), X_train)
    np.save(os.path.join(output_directory, "train_labels.npy"), y_train)
    np.save(os.path.join(output_directory, "test_sequences.npy"), X_test)
    np.save(os.path.join(output_directory, "test_labels.npy"), y_test)
    print(f"Saved {len(y_train)} training sequences and {len(y_test)} test sequences.")
    
if __name__ == '__main__':
    output_directory = 'data_uoc/output_sequences'
    sequences, labels = collect_all_data()
    X_train, X_test, y_train, y_test = stratified_split(sequences, labels)
    save_data(X_train, y_train, X_test, y_test, output_directory)
    
    # Verify the class distribution
    from collections import Counter
    print("Training set label distribution:", Counter(y_train))
    print("Test set label distribution:", Counter(y_test))
    # Verify the shapes of the saved sequences
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
