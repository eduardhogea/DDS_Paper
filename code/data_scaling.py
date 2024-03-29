import pandas as pd  # Importing pandas for data manipulation
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler for data scaling
import joblib  # Importing joblib for model serialization

def load_and_scale_data(csv_path, scaler=None, save_scaler_path=None):
    """
    Loads data from a CSV file, scales the features (excluding the 'Fault' column), 
    and returns the scaled DataFrame. Optionally saves the scaler model.
    
    Parameters:
    - csv_path (str): Path to the CSV file containing the data.
    - scaler (object): Scaler object for scaling the data. If None, a new StandardScaler is used.
    - save_scaler_path (str): Path to save the scaler model. If provided, the scaler model is saved.
    
    Returns:
    - scaled_df (DataFrame): Scaled DataFrame with features and target column ('Fault').
    """
    # Load the data
    data = pd.read_csv(csv_path)
    
    # Separate features and target
    features = data.columns[:-1]  # Assuming the last column is the target
    X = data[features]
    y = data['Fault']

    # Apply scaling
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if save_scaler_path:
            joblib.dump(scaler, save_scaler_path)
    else:
        X_scaled = scaler.transform(X)
    
    # Combine scaled features with target
    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df['Fault'] = y
    
    return scaled_df
