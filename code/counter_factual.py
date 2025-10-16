import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Standard library imports
import argparse
import csv
import math
import os
import pickle
import random
import re
import sys
import csv
from collections import Counter
import dice_ml



# Append config directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute dir the script is in
sys.path.append(os.path.join(script_dir, '..', 'config'))

# Third-party library imports
import joblib
import ltn
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tqdm import tqdm
from numpy import mean
import time

# Local module imports
import config as config
from model_creation import LSTMModel, lr_schedule
from sequence_generation import load_sequences, save_sequences
from model_evaluation import kfold_cross_validation, normalize_importances, permutation_importance_per_class
from pgb_data_processing import overview_csv_files, process_pgb_data
from data_scaling import load_and_scale_data
from util import concatenate_and_delete_ltn_csv_files
import commons as commons
from tensorflow.keras.callbacks import Callback

import numpy as np
import tensorflow as tf
from alibi.explainers import Counterfactual
from omnixai.data.timeseries import Timeseries
from omnixai.explainers.timeseries import TimeseriesExplainer
# Configurations
results_path_ltn = config.results_path_ltn
model_path = config.model_path
dataset_path = config.dataset_path
PGB_path = config.PGB_path
RGB_path = config.RGB_path
csv_file = config.csv_file
preprocessor_file = config.preprocessor_file
train_path = config.train_path
val_path = config.val_path
chunk_size = config.chunk_size
csv_directory = config.csv_directory
data_root_folder = config.data_root_folder
sequence_length = config.sequence_length
sequences_directory = config.sequences_directory
num_features = config.num_features
processed_bases = config.processed_bases
batch_size = config.batch_size
epochs = config.epochs
patience = config.patience
learning_rate = config.learning_rate
n_splits = config.n_splits
model_save_directory = config.model_save_directory
reg_value = config.reg_value
num_train_samples = config.num_train_samples
num_test_samples = config.num_test_samples
reg_type = config.reg_type
n_samples = config.n_samples
num_classes = config.num_classes
buffer_size = config.buffer_size
ltn_batch = config.ltn_batch
results_path = config.results_path
S = config.S
lr_ltn = config.lr_ltn
processed_file_tracker = config.processed_file_tracker

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

ltn_model_path = ""

# Load the model weights and data
model_path = '/home/ubuntu/dds_paper/DDS_Paper/model_weights/model_PGB_30_0_fold_1.h5'
ltn_model_path = '/home/ubuntu/dds_paper/DDS_Paper/model_weights/ltn_tf_model_PGB_30_0_fold_1.tf'
sequences_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs/sequences/PGB_30_0_test_scaled_sequences.npy'
labels_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/data/csvs/sequences/PGB_30_0_test_scaled_labels.npy'

sequences, labels = load_sequences(sequences_path, labels_path)

input_shape = (sequences.shape[1], sequences.shape[2])  # Adjust input shape from the loaded data

num_classes = len(np.unique(labels))

if ltn_model_path != "":
    model = tf.keras.models.load_model(ltn_model_path)
    importances_path = os.path.join("/home/ubuntu/dds_paper/DDS_Paper/model_weights/normalized_importances_PGB_30_0_fold_1.csv")
    normalized_average_importances = np.loadtxt(importances_path, delimiter=',', skiprows=1)

    sequences = sequences * np.array(normalized_average_importances)
else:
    model = LSTMModel(input_shape=input_shape, num_classes=num_classes, reg_type=reg_type, reg_value=reg_value, return_logits=True)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.build(input_shape=(None, *input_shape))  # None is for batch size
    model.load_weights(model_path)

# Step 1: Make original predictions using the model
original_predictions_logits = model.predict(sequences, verbose=0)
original_predicted_classes = np.argmax(original_predictions_logits, axis=-1)

# Adjust these variables as needed
max_attempts = 500       # Maximum number of perturbation attempts per timestep
perturbation_step = 0.001 # The amount by which to increase the perturbation

# Initialize a dictionary to store perturbation results
perturbation_results_limited = {}
num_sequences_to_analyze = 1  # Analyze the first sequence

# Iterate over the first num_sequences_to_analyze sequences
for seq_index in range(num_sequences_to_analyze):
    original_sequence = sequences[seq_index].copy()
    original_class = original_predicted_classes[seq_index]
    perturbation_results_limited[seq_index] = []
    
    # Iterate over each timestep in the sequence
    for time_index in range(sequences.shape[1]):
        attempts = 0
        perturbation_magnitude = 0
        positive_flip_found = False
        negative_flip_found = False
        positive_result = None
        negative_result = None
        
        while attempts < max_attempts and (not positive_flip_found or not negative_flip_found):
            attempts += 1
            perturbation_magnitude += perturbation_step
            
            # Positive perturbation
            if not positive_flip_found:
                perturbed_sequence_pos = original_sequence.copy()
                perturbed_sequence_pos[time_index] += perturbation_magnitude
                
                perturbed_sequence_expanded_pos = np.expand_dims(perturbed_sequence_pos, axis=0)
                perturbed_prediction_logits_pos = model.predict(perturbed_sequence_expanded_pos, verbose=0)
                perturbed_class_pos = np.argmax(perturbed_prediction_logits_pos, axis=-1)[0]
                
                if perturbed_class_pos != original_class:
                    positive_result = {
                        'time_index': time_index,
                        'original_class': original_class,
                        'perturbed_class': perturbed_class_pos,
                        'perturbation': perturbation_magnitude,
                        'attempts': attempts,
                        'direction': '+'
                    }
                    positive_flip_found = True
                    print(f"Sequence {seq_index}, Time Index {time_index}: Prediction flipped from {original_class} to {perturbed_class_pos} with positive perturbation {perturbation_magnitude} after {attempts} attempts.")
            
            # Negative perturbation
            if not negative_flip_found:
                perturbed_sequence_neg = original_sequence.copy()
                perturbed_sequence_neg[time_index] -= perturbation_magnitude
                
                perturbed_sequence_expanded_neg = np.expand_dims(perturbed_sequence_neg, axis=0)
                perturbed_prediction_logits_neg = model.predict(perturbed_sequence_expanded_neg, verbose=0)
                perturbed_class_neg = np.argmax(perturbed_prediction_logits_neg, axis=-1)[0]
                
                if perturbed_class_neg != original_class:
                    negative_result = {
                        'time_index': time_index,
                        'original_class': original_class,
                        'perturbed_class': perturbed_class_neg,
                        'perturbation': -perturbation_magnitude,
                        'attempts': attempts,
                        'direction': '-'
                    }
                    negative_flip_found = True
                    print(f"Sequence {seq_index}, Time Index {time_index}: Prediction flipped from {original_class} to {perturbed_class_neg} with negative perturbation {-perturbation_magnitude} after {attempts} attempts.")
        
        # After the loop, decide which perturbation to keep
        if positive_result and negative_result:
            # Both flips found, keep the one with smaller absolute perturbation
            if abs(positive_result['perturbation']) < abs(negative_result['perturbation']):
                perturbation_results_limited[seq_index].append(positive_result)
            else:
                perturbation_results_limited[seq_index].append(negative_result)
        elif positive_result:
            perturbation_results_limited[seq_index].append(positive_result)
        elif negative_result:
            perturbation_results_limited[seq_index].append(negative_result)
        else:
            # No flip found within max_attempts
            print(f"Sequence {seq_index}, Time Index {time_index}: No prediction flip found within max attempts.")

# Create a perturbation matrix to hold the minimal perturbation required at each time index
sequence_length = sequences.shape[1]
perturbation_matrix = np.full((num_sequences_to_analyze, sequence_length), np.nan)

for seq_index, flips in perturbation_results_limited.items():
    for flip in flips:
        time_idx = flip['time_index']
        perturbation = flip['perturbation']
        perturbation_matrix[seq_index, time_idx] = perturbation

# Adjusting to plot only the first 10 time steps
time_steps_to_plot = 10

# Calculate the mean of the original sequence for the first 10 time steps
mean_sequence = np.mean(sequences[0, :time_steps_to_plot], axis=1)

# Perturbation values for the first 10 time steps
perturbation_values = perturbation_matrix[0, :time_steps_to_plot]

# Create upper and lower bounds for the error
upper_bound = mean_sequence + perturbation_values  # Adding perturbation
lower_bound = mean_sequence - perturbation_values  # Subtracting perturbation

# Determine y-axis limits based on the data range with some margin
y_min = np.min(lower_bound) - 0.1
y_max = np.max(upper_bound) + 0.1

# Create a plot
plt.figure(figsize=(10, 6))

# Plot the mean sequence (original sequence values) for the first 10 time steps
plt.plot(np.arange(0, time_steps_to_plot), mean_sequence, label='Mean Sequence', linewidth=3, color='blue')

# Shaded area representing the perturbation effect as error
plt.fill_between(np.arange(0, time_steps_to_plot), 
                 lower_bound, 
                 upper_bound, 
                 color='blue', alpha=0.2, label='Perturbation Range')

# Customize the x-ticks to display the first 10 time steps
plt.xticks(ticks=np.arange(0, time_steps_to_plot), labels=np.arange(0, time_steps_to_plot), fontsize=16)

# Set appropriate y-axis limits
plt.ylim([y_min, y_max])

# Set labels and title with larger fonts
plt.title('Mean Sequence with Perturbation Ranges (First 10 Time Steps)', fontsize=20, fontweight='bold')
plt.xlabel('Time Steps', fontsize=18)
plt.ylabel('Sequence Mean Value', fontsize=18)

# Add a legend with larger font
plt.legend(fontsize=16)

# Increase grid line width and font size for a clean journal-style presentation
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("counter-factual.png", dpi = 400)
# Display the plot
plt.show()