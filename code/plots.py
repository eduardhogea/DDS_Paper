
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

class_names = [
    "Healthy",        # 0 - health
    "Chipped Tooth",  # 1 - chipped
    "Missing Tooth",  # 2 - miss
    "Root Crack",     # 3 - root
    "Surface Wear",   # 4 - surface
    "Ball Wear",      # 5 - ball
    "Combo Wear",     # 6 - combination
    "Inner Race",     # 7 - inner
    "Outer Race"      # 8 - outer
]


#!!
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/old_model_weights"

# Setting seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set font sizes
plt.rcParams['font.size'] = 14  # Adjust fontsize globally
plt.rcParams['axes.labelsize'] = 16  # Adjust fontsize for X and Y labels
plt.rcParams['axes.titlesize'] = 18  # Adjust fontsize for Titles
plt.rcParams['xtick.labelsize'] = 14  # Adjust fontsize for X ticks
plt.rcParams['ytick.labelsize'] = 14  # Adjust fontsize for Y ticks
plt.rcParams['legend.fontsize'] = 14  # Adjust fontsize for legends

# Colors cycle for plotting
colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue'])
class MetricsLogger(Callback):
    def __init__(self, csv_path, fold_number, base_name):
        super(MetricsLogger, self).__init__()
        self.csv_path = csv_path
        self.fold_number = fold_number
        self.base_name = base_name
        # Check if file exists to decide whether to write headers
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write the header
                writer.writerow(['Base Name', 'Epoch', 'Fold', 'Loss', 'Accuracy', 'Validation Loss', 'Validation Accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write epoch number, fold number, and desired metrics to the CSV
            writer.writerow([self.base_name, epoch + 1, self.fold_number, logs.get('loss'), logs.get('accuracy'), logs.get('val_loss'), logs.get('val_accuracy')])


# LTN metrics and groundings
metrics_dict = {
    'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
    'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
    'train_accuracy': tf.keras.metrics.CategoricalAccuracy(name="train_accuracy"),
    'test_accuracy': tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
}

kf = KFold(n_splits=2, shuffle=False)
counter = 0
console = Console()
processed_bases = set()

if os.path.exists(processed_file_tracker):
    with open(processed_file_tracker, "r") as file:
        processed_bases = set(file.read().splitlines())

metrics_summary = []

all_probabilities = []  # This will store the probabilities for ROC curve
all_predictions = []    # This will store the class indices for confusion matrix and accuracy
all_labels = []

for file in sorted(os.listdir(sequences_directory)):
    if "_train_scaled_sequences.npy" in file:
        #if counter >= S:
        #    break
        
        base_name = file.replace("_train_scaled_sequences.npy", "")
        if base_name in processed_bases:
            continue
        
        processed_bases.add(base_name)
        counter += 1
        
        # Load sequences and labels
        train_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_sequences.npy")
        train_label_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_labels.npy")
        X_train, y_train = load_sequences(train_sequence_file_path, train_label_file_path)
        
        test_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_sequences.npy")
        test_label_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_labels.npy")
        X_test, y_test = load_sequences(test_sequence_file_path, test_label_file_path)

        # Shuffle the sequences and corresponding labels
        train_indices = np.arange(len(X_train))
        np.random.shuffle(train_indices)
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]

        test_indices = np.arange(len(X_test))
        np.random.shuffle(test_indices)
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Merge for cross-validation
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        #for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        model = tf.keras.models.load_model(os.path.join(model_save_directory, f"ltn_tf_model_{base_name}_fold_{1}.tf"))

        importances_path = os.path.join(model_save_directory, f"normalized_importances_{base_name}_fold_{1}.csv")
        normalized_average_importances = np.loadtxt(importances_path, delimiter=',', skiprows=1)

        X_test_fold_weighted = X_test * np.array(normalized_average_importances)
        ds_test_fold = tf.data.Dataset.from_tensor_slices((X_test_fold_weighted, y_test))
        ds_test_fold = ds_test_fold.batch(batch_size)

        for features, label in ds_test_fold:
            logits = model(features)
            probabilities = tf.nn.softmax(logits).numpy()  # Get probabilities
            predicted_classes = tf.argmax(logits, axis=1).numpy()  # Get predicted classes
            all_probabilities.extend(probabilities)  # Store probabilities for ROC
            all_predictions.extend(predicted_classes)  # Store predicted classes for confusion matrix
            all_labels.extend(label.numpy())
                
all_probabilities = np.array(all_probabilities)
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
#probabilities = tf.nn.softmax(all_predictions).numpy()
# Binarize the labels for multiclass ROC
all_labels_one_hot = label_binarize(all_labels, classes=range(n_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
colors = plt.cm.viridis(np.linspace(0, 1, n_classes))

plt.figure(figsize=(10, 8))
for i, color in zip(range(n_classes), colors):
    fpr[i], tpr[i], _ = roc_curve(all_labels_one_hot[:, i], all_probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(roc_auc[i], f'Class {class_names[i]}')
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Overall ROC Curve for All Speed Scenarios')
plt.legend(loc="lower right")
plt.savefig("plots/overall_roc_uoc.png", format='png', dpi=300, bbox_inches='tight')
plt.show()


print("Distribution of all labels:", Counter(all_labels))
print("Distribution of all predictions:", Counter(all_predictions))
print("Sample labels:", all_labels[:10])
print("Sample predictions:", all_predictions[:10])

# Confusion Matrix computation and visualization
cm = confusion_matrix(all_labels, all_predictions, labels=range(n_classes))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Overall Confusion Matrix for All Speed Scenarios')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("plots/overall_confusion_matrix_uoc.png", format='png', dpi=300, bbox_inches='tight')
plt.show()


