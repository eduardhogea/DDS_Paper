import os
import csv
from collections import Counter
import pandas as pd
from tqdm import tqdm
import argparse
import importlib.util

def extract_fault(file_name):
    fault_mapping = {
        '0Health': 'HEA', '1Chipped': 'CTF'
    }
    for key, value in fault_mapping.items():
        if key in file_name:
            return value
    return None

def make_csv_writer(csv_file):
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['a1', 'a2', 'a3', 'a4','Fault'])
    return csv_writer

def generate_csv(output_directory, root_path, speed, experiment, files, train_ratio=0.8):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    train_filename_suffix = f"{speed}_{experiment}_train" if experiment else f"{speed}_train"
    test_filename_suffix = f"{speed}_{experiment}_test" if experiment else f"{speed}_test"
    
    train_output_file_path = os.path.join(output_directory, f"kaggle_{train_filename_suffix}.csv")
    test_output_file_path = os.path.join(output_directory, f"kaggle_{test_filename_suffix}.csv")
    
    with open(train_output_file_path, 'w', newline='', encoding='utf-8') as train_csvfile, \
        open(test_output_file_path, 'w', newline='', encoding='utf-8') as test_csvfile:
        train_csv_writer = make_csv_writer(train_csvfile)
        test_csv_writer = make_csv_writer(test_csvfile)
        
        for file in tqdm(files, desc=f"Processing {speed} {experiment}", unit="file"):
            fault_type = extract_fault(file)
            file_path = os.path.join(root_path, file)
            
            # Load the entire file to get the total number of rows
            data = pd.read_csv(file_path, header=None, skiprows=1)
            total_rows = data.shape[0]
            
            # Calculate train and test sample sizes based on the 80/20 split
            num_train_samples = int(train_ratio * total_rows)
            num_test_samples = total_rows - num_train_samples
            
            # Split the data
            train_samples = data.iloc[:num_train_samples, :]
            test_samples = data.iloc[num_train_samples:, :]
            
            # Write train samples to CSV
            for index, row in train_samples.iterrows():
                train_csv_writer.writerow(row[:4].tolist() + [fault_type])
            
            # Write test samples to CSV
            for index, row in test_samples.iterrows():
                test_csv_writer.writerow(row[:4].tolist() + [fault_type])


def process_pgb_data(data_root_folder, csv_directory, train_ratio=0.7):
    for root, dirs, files in os.walk(data_root_folder):
        parts = root.split(os.sep)
        if 'Variable_speed' in parts:
            speed = "Variable_speed"
            experiment_dir = parts[-1]  # Get the last part as the experiment name
            exp_files = [f for f in os.listdir(root) if f.endswith('.csv')]
            generate_csv(csv_directory, root, speed, experiment_dir, exp_files, train_ratio)
        elif 'kaggle' in parts and files:
            speed = parts[-1]  # Last part of 'root' is the speed directory
            generate_csv(csv_directory, root, speed, '', files, train_ratio)

            
            
def overview_csv_files(directory):
    data = []
    all_faults = set()

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            print(df.head(5))

            # Check if the CSV is empty (aside from the header)
            if df.shape[0] == 0:
                # Delete the empty CSV file
                os.remove(file_path)
                print(f"Deleted empty file: {file_path}")
                continue  # Skip further processing for this file

            num_samples = len(df)
            fault_distribution = Counter(df['Fault'])
            all_faults.update(fault_distribution.keys())
            data.append({'File Name': file, 'Number of Samples': num_samples, **fault_distribution})

    if not data:  # If no data has been gathered, exit the function
        print("No data found.")
        return

    overview_df = pd.DataFrame(data)
    for fault in all_faults:
        if fault not in overview_df.columns:
            overview_df[fault] = 0

    cols = ['File Name', 'Number of Samples'] + sorted(all_faults)
    overview_df = overview_df[cols]
    overview_df.fillna(0, inplace=True)
    overview_df.loc[:, 'Number of Samples':] = overview_df.loc[:, 'Number of Samples':].astype(int)

    overview_df = overview_df.sort_values(by='File Name')
    print(overview_df.to_string(index=False))

def main(config_file):
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    data_root_folder = config_module.data_root_folder
    csv_directory = config_module.csv_directory
    num_train_samples = config_module.num_train_samples
    num_test_samples = config_module.num_test_samples

    process_pgb_data(data_root_folder, csv_directory, num_train_samples, num_test_samples)
    overview_csv_files(csv_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PGB data and generate CSV files")
    parser.add_argument("config_file", type=str, help="Path to the configuration file")

    args = parser.parse_args()

    main(args.config_file)