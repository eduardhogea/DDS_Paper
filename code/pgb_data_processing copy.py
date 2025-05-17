import os
import csv
from collections import Counter
import pandas as pd
from tqdm import tqdm
import argparse
import importlib.util

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

def generate_csv(output_directory, root_path, speed, experiment, files, num_train_samples, num_test_samples):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    train_filename_suffix = f"{speed}_{experiment}_train" if experiment else f"{speed}_train"
    test_filename_suffix = f"{speed}_{experiment}_test" if experiment else f"{speed}_test"
    
    train_output_file_path = os.path.join(output_directory, f"PGB_{train_filename_suffix}.csv")
    test_output_file_path = os.path.join(output_directory, f"PGB_{test_filename_suffix}.csv")
    
    with open(train_output_file_path, 'w', newline='', encoding='utf-8') as train_csvfile, \
        open(test_output_file_path, 'w', newline='', encoding='utf-8') as test_csvfile:
        train_csv_writer = make_csv_writer(train_csvfile)
        test_csv_writer = make_csv_writer(test_csvfile)
        
        for file in tqdm(files, desc=f"Processing {speed} {experiment}", unit="file"):
            fault_type = extract_fault(file)
            file_path = os.path.join(root_path, file)
            
            total_rows = num_train_samples + num_test_samples
            data = pd.read_csv(file_path, sep='\t', header=None, encoding='ISO-8859-1', skiprows=1, nrows=total_rows)
            train_samples, test_samples = data.iloc[:num_train_samples, :], data.iloc[num_train_samples:total_rows, :]
            
            for index, row in train_samples.iterrows():
                train_csv_writer.writerow(row[:8].tolist() + [fault_type])
            
            for index, row in test_samples.iterrows():
                test_csv_writer.writerow(row[:8].tolist() + [fault_type])

def process_pgb_data(data_root_folder, csv_directory, num_train_samples, num_test_samples):
    for root, dirs, files in os.walk(data_root_folder):
        parts = root.split(os.sep)
        if 'Variable_speed' in parts:
            speed = "Variable_speed"
            experiment_dir = parts[-1]  # Get the last part as the experiment name
            exp_files = [f for f in os.listdir(root) if f.endswith('.txt')]
            generate_csv(csv_directory, root, speed, experiment_dir, exp_files, num_train_samples, num_test_samples)
        elif 'PGB' in parts and files:
            speed = parts[-1]  # Last part of 'root' is the speed directory
            generate_csv(csv_directory, root, speed, '', files, num_train_samples, num_test_samples)
            
            
def overview_csv_files(directory):
    data = []
    all_faults = set()

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)

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
