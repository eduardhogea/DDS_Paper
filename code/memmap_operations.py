import os
import numpy as np
from tqdm import tqdm
import pickle

def memmap_append_and_save(input_directory, output_directory, dataset_type, file_type):
    # Update the output file path to indicate pickle format
    output_file_path = os.path.join(output_directory, f"{dataset_type}_merged_{file_type}.pkl")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    merged_data = None
    current_size = 0

    file_names = [fn for fn in os.listdir(input_directory) if fn.endswith(f'_{file_type}.npy') and dataset_type in fn]
    for file_name in tqdm(file_names, desc=f"Merging {dataset_type} {file_type}"):
        path = os.path.join(input_directory, file_name)
        data = np.load(path)

        # Adjust for both 1D and 2D+ data
        new_shape = (current_size + data.shape[0],) + data.shape[1:] if len(data.shape) > 1 else (current_size + data.shape[0],)
        if merged_data is None:
            # Initially, directly use the loaded data
            merged_data = data.copy()
        else:
            # Concatenate new data
            merged_data = np.concatenate((merged_data, data), axis=0)
        
        current_size += data.shape[0]

    # After processing all files, save the merged data using pickle
    with open(output_file_path, 'wb') as f:
        pickle.dump(merged_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"{dataset_type.capitalize()} {file_type} data merged and saved to {output_file_path} in pickle format")

def merge_npy_files_with_memmap_separated(input_directory, output_directory):
    for dataset_type in ['train', 'test']:
        for file_type in ['sequences', 'labels']:
            memmap_append_and_save(input_directory, output_directory, dataset_type, file_type)
            