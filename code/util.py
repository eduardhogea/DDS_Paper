import re
import os
import pandas as pd

def extract_speed_from_filename(file_name):
    """
    Extracts the speed from the filename.
    Returns the numeric speed for fixed speeds, or -1 for variable speeds.
    """
    fixed_speed_match = re.search(r"PGB_(\d+)_", file_name)
    if fixed_speed_match:
        return int(fixed_speed_match.group(1))
    variable_speed_match = re.search(r"Variable_speed", file_name)
    if variable_speed_match:
        return -1  # Special value for variable speeds
    return None

def concatenate_and_delete_ltn_csv_files(directory, output_file):
    dfs = []
    files_to_delete = []
    
    for filename in os.listdir(directory):
        if filename.endswith('_ltn.csv'):
            full_path = os.path.join(directory, filename)
            try:
                parts = filename.split("_fold")
                base_name = parts[0]
                if len(parts) > 1:
                    fold_number = parts[1].split('_')[0]
                else:
                    print(f"Skipping file due to unexpected format: {filename}")
                    continue
                
                df = pd.read_csv(full_path)
                df['Fold'] = fold_number
                df['Speed'] = base_name
                
                dfs.append(df)
                files_to_delete.append(full_path)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue
    
    if not dfs:
        print("No files to process.")
        return
    
    combined_df = pd.concat(dfs, ignore_index=True)
    columns_order = ['Fold', 'Speed'] + [col for col in combined_df.columns if col not in ['Fold', 'Speed']]
    combined_df = combined_df[columns_order]
    file_exists = os.path.exists(output_file)
    combined_df.to_csv(output_file, mode='a', index=False, header=not file_exists)
    
    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"Deleted {file_path}")
