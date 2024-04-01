'''
you can run the script with python script.py --create-sequences to only create sequences, python script.py --train-model to only train the model, or python script.py --create-sequences --train-model to perform both tasks.
'''

import argparse
import csv
import math
import os
import pickle
import random
import re
import sys
sys.path.append('../')

# Third-party library imports
from model_creation import LSTMModel, lr_schedule
from sequence_generation import load_sequences
from model_evaluation import kfold_cross_validation, normalize_importances, permutation_importance_per_class
#from ltn_utils import axioms, test_step, train_step
from pgb_data_processing import overview_csv_files, process_pgb_data
from data_scaling import load_and_scale_data
from sequence_generation import save_sequences
import config.config as config
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers, models, optimizers, callbacks, regularizers, utils
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, LSTM, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model
from keras.utils import to_categorical
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm import tqdm
from numpy import mean
from collections import Counter
import ltn
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import code.commons as commons

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy



# Configurations
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


metrics_dict = {
    'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
    'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
    'train_accuracy': tf.keras.metrics.CategoricalAccuracy(name="train_accuracy"),
    'test_accuracy': tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
}

Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

class_0 = ltn.Constant(0, trainable=False)
class_1 = ltn.Constant(1, trainable=False)
class_2 = ltn.Constant(2, trainable=False)
class_3 = ltn.Constant(3, trainable=False)
class_4 = ltn.Constant(4, trainable=False)
class_5 = ltn.Constant(5, trainable=False)
class_6 = ltn.Constant(6, trainable=False)
class_7 = ltn.Constant(7, trainable=False)
class_8 = ltn.Constant(8, trainable=False)

#p = ltn.Predicate.FromLogits(model, activation_function="softmax", with_class_indexing=True)




def main():
    parser = argparse.ArgumentParser(description="Process data, create sequences, and train the model")
    parser.add_argument("--create-sequences", action="store_true", help="Create sequences from scaled CSV files")
    parser.add_argument("--train-model", action="store_true", help="Train the model using k-fold cross-validation")
    args = parser.parse_args()
    
    if args.create_sequences:
        # Clean the directories
        clean_directory(csv_directory)
        clean_directory(sequences_directory)
        
        process_pgb_data(data_root_folder, csv_directory, num_train_samples, num_test_samples)
        overview_csv_files(csv_directory)
        
        # Iterate over your dataset files
        for root, dirs, files in os.walk(csv_directory):
            for file in sorted(files):
                if file.endswith('.csv') and not file.endswith('_scaled.csv'):  # Process only unscaled .csv files
                    csv_path = os.path.join(root, file)
                    if 'train' in file:
                        # Handle training data
                        scaler_path = os.path.join(root, 'scaler_' + file.replace('.csv', '.joblib'))
                        scaled_train_df = load_and_scale_data(csv_path, save_scaler_path=scaler_path)
                        # Save the scaled training data
                        scaled_csv_path = csv_path.replace('.csv', '_scaled.csv')
                        scaled_train_df.to_csv(scaled_csv_path, index=False)
                    elif 'test' in file:
                        # Handle testing data
                        scaler_path = os.path.join(root, 'scaler_' + file.replace('_test.csv', '_train.joblib'))
                        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                        scaled_test_df = load_and_scale_data(csv_path, scaler=scaler)
                        # Save the scaled testing data
                        scaled_csv_path = csv_path.replace('.csv', '_scaled.csv')
                        scaled_test_df.to_csv(scaled_csv_path, index=False)
                    # Delete the original unscaled .csv file
                    os.remove(csv_path)
                    
        # Create sequences
        save_sequences(csv_directory, sequences_directory, sequence_length)
    
    if args.train_model:
        kf = KFold(n_splits=n_splits, shuffle=False)
        counter = 0
        console = Console()

        # Placeholder for processed base names and metrics
        processed_bases = set()
        metrics_summary = []

        for file in sorted(os.listdir(sequences_directory)):
            if "_train_scaled_sequences.npy" in file:
                base_name = file.replace("_train_scaled_sequences.npy", "")
                if base_name in processed_bases:
                    continue
                counter+=1
                
                
                    
                    
                # Load sequences and labels
                train_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_sequences.npy")
                train_label_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_labels.npy")
                X_train, y_train = load_sequences(train_sequence_file_path, train_label_file_path)

                # Assuming the existence of a test set (adjust if necessary)
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

                input_shape = (sequence_length, num_features)
                fold_metrics = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                    console.print(f"[bold green]Training fold {fold + 1}/{n_splits} for {base_name}[/]")
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                    model = LSTMModel(input_shape=input_shape, num_classes=num_classes, reg_type=reg_type, reg_value=reg_value, return_logits=True)
                    
                    model.compile(optimizer=Adam(learning_rate=0.001),
                        loss=SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
                    
                    lr_scheduler = LearningRateScheduler(lr_schedule)
                    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

                    model_filepath = os.path.join(model_save_directory, f"model_{base_name}_fold_{fold+1}")
                    checkpoint = ModelCheckpoint(model_filepath, save_best_only=True, monitor='val_loss', save_weights_only=False)
                    history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler, checkpoint], verbose=1)
                    
                    # Assuming your model outputs class indices directly
                    y_val_pred_classes = model.predict(X_val_fold)
                    y_val_pred_classes = np.argmax(y_val_pred_classes, axis=1)  # Get predicted classes

                    # Since y_val_fold contains integer labels, there's no need for conversion
                    y_val_true_classes = y_val_fold  # Directly use the integer labels

                    # Calculate and store metrics for this fold
                    accuracy = accuracy_score(y_val_true_classes, y_val_pred_classes)
                    precision = precision_score(y_val_true_classes, y_val_pred_classes, average='macro', zero_division=0)
                    recall = recall_score(y_val_true_classes, y_val_pred_classes, average='macro', zero_division=0)
                    f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='macro')
                    fold_metrics.append((accuracy, precision, recall, f1))


                    console.print(f"Fold {fold+1} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    
                    class_importances = permutation_importance_per_class(model, X_val_fold, y_val_fold, n_repeats=4, n_samples=n_samples)
                    for class_id, importances in class_importances.items():
                        print(f"{class_id} Feature Importances:", importances)
                    
                    
                    p = ltn.Predicate.FromLogits(model, activation_function="softmax", with_class_indexing=True)
                    
                    @tf.function
                    def axioms(features, labels, training=False):
                        x_A = ltn.Variable("x_A", features[labels == 0])
                        x_B = ltn.Variable("x_B", features[labels == 1])
                        x_C = ltn.Variable("x_C", features[labels == 2])
                        x_D = ltn.Variable("x_D", features[labels == 3])
                        x_E = ltn.Variable("x_E", features[labels == 4])
                        x_F = ltn.Variable("x_F", features[labels == 5])
                        x_G = ltn.Variable("x_G", features[labels == 6])
                        x_H = ltn.Variable("x_H", features[labels == 7])
                        x_I = ltn.Variable("x_I", features[labels == 8])
                        axioms = [
                            Forall(x_A, p([x_A, class_0], training=training)),
                            Forall(x_B, p([x_B, class_1], training=training)),
                            Forall(x_C, p([x_C, class_2], training=training)),
                            Forall(x_D, p([x_D, class_3], training=training)),
                            Forall(x_E, p([x_E, class_4], training=training)),
                            Forall(x_F, p([x_F, class_5], training=training)),
                            Forall(x_G, p([x_G, class_6], training=training)),
                            Forall(x_H, p([x_H, class_7], training=training)),
                            Forall(x_I, p([x_I, class_8], training=training))
                        ]
                        sat_level = formula_aggregator(axioms).tensor
                        return sat_level

                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    
                    @tf.function
                    def train_step(features, labels):
                        # sat and update
                        with tf.GradientTape() as tape:
                            sat = axioms(features, labels, training=True)
                            loss = 1.-sat
                        gradients = tape.gradient(loss, p.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, p.trainable_variables))
                        sat = axioms(features, labels) # compute sat without dropout
                        metrics_dict['train_sat_kb'](sat)
                        # accuracy
                        predictions = model([features])
                        metrics_dict['train_accuracy'](tf.one_hot(labels,9),predictions)
                        
                    @tf.function
                    def test_step(features, labels):
                        # sat
                        sat = axioms(features, labels)
                        metrics_dict['test_sat_kb'](sat)
                        # accuracy
                        predictions = model([features])
                        metrics_dict['test_accuracy'](tf.one_hot(labels,9),predictions)
                    
                    normalized_importances = {}

                    # Assuming normalize_importances function logic remains the same and is applicable here.
                    for class_label, importances in class_importances.items():
                        normalized = normalize_importances(importances)
                        normalized_importances[class_label] = normalized
                        
                    ds_train_fold = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))
                    ds_val_fold = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold))
                    
                    
                    # Shuffle the dataset before batching
                    ds_train_fold = ds_train_fold.shuffle(buffer_size).batch(ltn_batch)
                    ds_val_fold = ds_val_fold.shuffle(buffer_size).batch(ltn_batch)

                    #ds_val_fold = tf.data.Dataset.from_tensor_slices((X,y)).batch(small_batch_size)
                    for batch_features, batch_labels in ds_val_fold:
                        
                        # Assuming the axioms function and predicate are prepared for batch processing
                        batch_satisfaction_level = axioms(batch_features, batch_labels, training=False)
                        
                        # Print or aggregate the batch satisfaction levels as needed
                        # .numpy() is used to convert TensorFlow tensors to numpy arrays for printing or further processing
                        print(f"Batch Satisfaction Level: {batch_satisfaction_level.numpy():.4f}")
                        break
                    



                    commons.train(
                        epochs,
                        metrics_dict,
                        ds_train_fold,
                        ds_val_fold,
                        train_step,
                        test_step,
                        csv_path="./results.csv",
                        track_metrics=1
                    )
                    

                    
                        # After processing all folds for the current CSV pair
                if fold_metrics:
                    # Calculate the average of each metric across all folds
                    avg_accuracy = mean([metric[0] for metric in fold_metrics])
                    avg_precision = mean([metric[1] for metric in fold_metrics])
                    avg_recall = mean([metric[2] for metric in fold_metrics])
                    avg_f1 = mean([metric[3] for metric in fold_metrics])

                    # Append averaged metrics to the metrics_summary for overall analysis if needed
                    metrics_summary.append((base_name, avg_accuracy, avg_precision, avg_recall, avg_f1))

                    # Print the averages
                    console.print(f"[bold magenta]Average metrics for {base_name} across {n_splits} folds:[/]")
                    console.print(f"Average Accuracy: {avg_accuracy:.4f}")
                    console.print(f"Average Precision: {avg_precision:.4f}")
                    console.print(f"Average Recall: {avg_recall:.4f}")
                    console.print(f"Average F1: {avg_f1:.4f}\n")
                
                

                
            if counter!=0:
                break


        console.print(f"[bold blue]Model for {base_name} saved.[/]")
        # Optionally, after all file pairs have been processed, print a summary of averages across all file pairs
        console.print("[bold blue]Overall Averages Across All File Pairs:[/]")
        overall_avg_accuracy = mean([metrics[1] for metrics in metrics_summary])
        overall_avg_precision = mean([metrics[2] for metrics in metrics_summary])
        overall_avg_recall = mean([metrics[3] for metrics in metrics_summary])
        overall_avg_f1 = mean([metrics[4] for metrics in metrics_summary])

        console.print(f"Overall Average Accuracy: {overall_avg_accuracy:.4f}")
        console.print(f"Overall Average Precision: {overall_avg_precision:.4f}")
        console.print(f"Overall Average Recall: {overall_avg_recall:.4f}")
        console.print(f"Overall Average F1: {overall_avg_f1:.4f}")

def clean_directory(directory):
    # Remove all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
            
if __name__ == "__main__":


    main()
    
