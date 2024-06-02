
# DDS Paper-LTN Integration

![plot_highdef](https://github.com/eduardhogea/DDS_Paper/assets/72266259/5d4a702e-283d-4e88-8da0-ab22e0ed24f8)

This repository contains the implementation of a hybrid neuro-symbolic model for time-series fault classification in industrial planetary gearboxes. The model integrates Long Short-Term Memory (LSTM) networks with Explainable AI (XAI) techniques and Logic Tensor Networks (LTN), enhancing both predictive accuracy and interpretability.

## System Architecture

The model architecture includes:
- **LSTM Backbone**: Handles initial learning from time-series data to capture temporal dependencies critical for fault detection.
- **XAI Component**: Focuses on permutation feature importance to refine the feature weighting.
- **LTN Framework**: Embeds logical constraints derived from the data to further enhance the model by ensuring logical consistency and interpretability.

## Key Features

1. **Time-Series Data Handling**: The model is specifically designed to optimize the fault classification process for planetary gearboxes using simulated time-series data.
2. **Feature Importance Reweighting**: Utilizes XAI techniques to adjust the weighting of each feature, amplifying the impact of important features and diminishing noise.
3. **Logical Constraints**: LTN framework guides the LSTM model towards solutions that fit the data and adhere to predefined logical rules.
4. **Explainability and Transparency**: Provides insights into the decision-making pathways of the AI model, crucial for deployments in critical environments.

## Dataset

The dataset used in this study is derived from a drivetrain dynamics simulator testbed, featuring various operational conditions for planetary gearboxes. The data includes multichannel vibration signals collected from different sensor channels under various load and speed conditions.

## Implementation Details

### Preprocessing

- **Standardization**: Features are standardized to have zero mean and unit variance.
- **Segmentation**: Data is segmented into training, validation, and test sets, with sequences created using a sliding window technique.
- **Balancing**: The dataset is balanced across fault types and speeds.

### Model Training

1. **Train LSTM**:
   - Initialize and train the LSTM model on the preprocessed data.
   - Use cross-validation and various regularization techniques to optimize performance.

2. **Feature Importance Assessment**:
   - Apply permutation feature importance to evaluate the significance of each feature.
   - Reweight features based on their importance scores.

3. **Train LTN**:
   - Integrate the trained LSTM with the LTN framework.
   - Apply logical constraints and retrain the model to enhance logical consistency.

### Evaluation

- **Performance Metrics**: Accuracy, ROC-AUC, confusion matrix, and satisfiability metrics are used to evaluate the model.
- **Comparative Analysis**: Compare the performance of the original LSTM model with the fine-tuned LTN model.
## Installation

To use this, first, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/DDS_Paper.git
cd DDS_Paper
```

Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run the script with the following commands:

- To only create sequences:
  ```bash
  python code/run.py --create-sequences
  ```

- To only train the model:
  ```bash
  python code/run.py --train-model
  ```

- To perform both tasks:
  ```bash
  python code/run.py --create-sequences --train-model
  ```

## Configuration

The configuration files for this project are located in the `config/` directory. These files allow you to customize various parameters and settings for creating sequences and training models. The primary configuration settings are defined in `config.py`.

### Key Configuration Parameters

- **Paths**
  - `BASE_DIR`: Base directory of the project.
  - `config_path`: Path to the main configuration file.
  - `dataset_path`: Path to the dataset directory.
  - `PGB_path`: Path to the PGB dataset.
  - `RGB_path`: Path to the RGB dataset.
  - `csv_file`: Path to the main CSV data merged files.
  - `preprocessor_file`: Path to the preprocessor file.
  - `train_path`: Path to the training data CSV file.
  - `val_path`: Path to the validation data CSV file.
  - `csv_directory`: Directory containing CSV files.
  - `data_root_folder`: Root folder for data.
  - `sequences_directory`: Directory for sequence data.
  - `model_save_directory`: Directory to save trained models.
  - `model_path`: Path to the saved model file.
  - `results_path`: Path to the results CSV file.
  - `results_path_ltn`: Path to the results directory for LTN.
  - `processed_file_tracker`: Path to the file tracking processed data.

- **Data Processing**
  - `chunk_size`: Size of data chunks for processing.
  - `sequence_length`: Length of sequences to be created.
  - `num_features`: Number of features in the dataset.
  - `processed_bases`: Set of base names to avoid redundancy.

- **Model Training Parameters**
  - `batch_size`: Batch size for training.
  - `epochs`: Number of epochs for training.
  - `patience`: Patience parameter for early stopping.
  - `learning_rate`: Learning rate for the optimizer.
  - `lr_ltn`: Learning rate for LTN.
  - `n_splits`: Number of splits for cross-validation.
  - `reg_value`: Regularization value.
  - `num_train_samples`: Number of training samples per fault type.
  - `num_test_samples`: Number of test samples per fault type.
  - `reg_type`: Type of regularization (e.g., l1, l2).
  - `n_samples`: Number of samples to consider fo feature importance.
  - `num_classes`: Number of output classes.
  - `buffer_size`: Buffer size for data loading.
  - `ltn_batch`: Batch size for LTN.
  - `S`: Number of speeds to consider.

- **Seed for Reproducibility**
  - `np.random.seed(42)`: Seed value for reproducibility.

## Project Structure

- `.github/`: GitHub-related files and workflows.
- `code/`: Contains the main code for running the scripts.
- `config/`: Configuration files for the project.
- `extra_code/`: Additional code that might be useful.
- `plots/`: Directory to store plots generated by the scripts.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: This file.



## Results

The hybrid neuro-symbolic model significantly outperforms the standard LSTM model, particularly in complex fault scenarios. The model demonstrates high accuracy and strong differentiation capabilities across various fault types.

## Conclusion

This study presents a robust approach for enhancing fault classification in industrial settings using a hybrid neuro-symbolic model. The integration of LSTM networks with XAI and LTN techniques provides a powerful tool for predictive maintenance, ensuring both high accuracy and interpretability.


For detailed explanations and further readings, please refer to the full paper included in this repository.
