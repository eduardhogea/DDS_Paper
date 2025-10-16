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
tf.compat.v1.disable_v2_behavior()
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
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tqdm import tqdm
from numpy import mean
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
tf.compat.v1.disable_v2_behavior()


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
n_classes = config.num_classes
buffer_size = config.buffer_size
ltn_batch = config.ltn_batch
results_path = config.results_path
S = config.S
lr_ltn = config.lr_ltn
processed_file_tracker = config.processed_file_tracker
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/old_model_weights"


for file in sorted(os.listdir(sequences_directory)):
    if "_train_scaled_sequences.npy" in file:
        #if counter >= S:
        #    break
        
        base_name = file.replace("_train_scaled_sequences.npy", "")
        if base_name in processed_bases:
            continue
        
        processed_bases.add(base_name)
        # Load sequences and labels
        train_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_sequences.npy")
        train_label_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_labels.npy")
        X_train, y_train = load_sequences(train_sequence_file_path, train_label_file_path)
        
        test_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_sequences.npy")
        test_label_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_labels.npy")
        X_test, y_test = load_sequences(test_sequence_file_path, test_label_file_path)
        model = tf.keras.models.load_model(os.path.join(model_save_directory, f"ltn_tf_model_{base_name}_fold_{1}.tf"))