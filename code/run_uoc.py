'''
you can run the script with python script.py --create-sequences to only create sequences, python script.py --train-model to only train the model, or python script.py --create-sequences --train-model to perform both tasks.
'''

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
import config_uoc as config
from model_creation import LSTMModel, lr_schedule
from sequence_generation import load_sequences, save_sequences
from model_evaluation import kfold_cross_validation, normalize_importances, permutation_importance_per_class
from pgb_data_processing import overview_csv_files, process_pgb_data
from data_scaling import load_and_scale_data
from util import concatenate_and_delete_ltn_csv_files
import commons as commons
from tensorflow.keras.callbacks import Callback

class MetricsLogger(Callback):
    def __init__(self, csv_path, fold_number, base_name=""):
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

#!!
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/model_weights_uoc"


# Setting seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# LTN metrics and groundings
metrics_dict = {
    'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
    'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
    'train_accuracy': tf.keras.metrics.CategoricalAccuracy(name='train_accuracy'),
    'test_accuracy': tf.keras.metrics.CategoricalAccuracy(name='test_accuracy'),
    'test_sat_phi1': tf.keras.metrics.Mean(name='test_sat_phi1'),
    'test_sat_phi2': tf.keras.metrics.Mean(name='test_sat_phi2'),
    'test_sat_phi3': tf.keras.metrics.Mean(name='test_sat_phi3'),
    'test_sat_phi4': tf.keras.metrics.Mean(name='test_sat_phi4'),
    'test_sat_phi5': tf.keras.metrics.Mean(name='test_sat_phi5'),
    'test_sat_phi6': tf.keras.metrics.Mean(name='test_sat_phi6'),
    'test_sat_phi7': tf.keras.metrics.Mean(name='test_sat_phi7'),
    'test_sat_phi8': tf.keras.metrics.Mean(name='test_sat_phi8')
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

# Check if the file exists to decide whether to write headers
if not os.path.exists(results_path):
    with open(results_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(['Speed', 'Epoch', 'Fold', 'Loss', 'Accuracy', 'Validation Loss', 'Validation Accuracy'])




def main():
    parser = argparse.ArgumentParser(description="Process data, create sequences, and train the model")
    parser.add_argument("--create-sequences", action="store_true", help="Create sequences from scaled CSV files")
    parser.add_argument("--train-model", action="store_true", help="Train the model using k-fold cross-validation")
    args = parser.parse_args()
    
    if args.create_sequences:
        # Clean the directories when creating new sequences
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
                    # Delete the original unscaled .csv file for memory preservation
                    os.remove(csv_path)
                    
        # Create sequences
        save_sequences(csv_directory, sequences_directory, sequence_length)
    
    if args.train_model:
        
        kf = KFold(n_splits=n_splits, shuffle=False)
        counter = 0
        console = Console()
        processed_bases = set()
        
        if os.path.exists(processed_file_tracker):
            with open(processed_file_tracker, "r") as file:
                processed_bases = set(file.read().splitlines())
        
        metrics_summary = []
        
            
            
        # Load sequences and labels
        train_sequence_file_path = os.path.join(sequences_directory, "train_sequences.npy")
        train_label_file_path = os.path.join(sequences_directory, "train_labels.npy")
        X_train, y_train = load_sequences(train_sequence_file_path, train_label_file_path)
        
        test_sequence_file_path = os.path.join(sequences_directory, "test_sequences.npy")
        test_label_file_path = os.path.join(sequences_directory, "test_labels.npy")
        X_test, y_test = load_sequences(test_sequence_file_path, test_label_file_path)

        # Shuffle the sequences and corresponding labels. Before this they were kept ordered.
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
            print(f"Class distribution in train fold {fold+1}:", Counter(y[train_idx]))
            print(f"Class distribution in validation fold {fold+1}:", Counter(y[val_idx]))
            metrics_logger = MetricsLogger(results_path, fold_number=fold+1)
            console.print(f"[bold green]Training fold {fold + 1}/{n_splits}")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = LSTMModel(input_shape=input_shape, num_classes=num_classes, reg_type=reg_type, reg_value=reg_value, return_logits=True)
            
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
            
            lr_scheduler = LearningRateScheduler(lr_schedule)
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

            model_filepath = os.path.join(model_save_directory, f"model_fold_{fold+1}.h5")
            checkpoint = ModelCheckpoint(filepath=model_filepath, save_best_only=True, monitor='val_accuracy', save_weights_only=True, mode='max')
            history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                                epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, lr_scheduler, checkpoint, metrics_logger], verbose=1)
            
            
            model.load_weights(model_filepath)
            
            y_val_pred_classes = model.predict(X_val_fold, batch_size = batch_size)
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
            
            importances_matrix = np.array(list(class_importances.values()))
            # Calculate the average importance across all classes
            average_importances = np.mean(importances_matrix, axis=0)
            
            normalized_average_importances = normalize_importances(average_importances)

            print("Normalized Average Feature Importances:", normalized_average_importances)
            importances_path = os.path.join(model_save_directory, f"normalized_importances_fold_{fold+1}.csv")

            # Save normalized average feature importances to a CSV file
            np.savetxt(importances_path, normalized_average_importances, delimiter=',', header='Feature Importances', comments='')

            # Inform the user where the importances have been saved
            print(f"Saved normalized feature importances to {importances_path}")

            p = ltn.Predicate.FromLogits(model, activation_function="softmax", with_class_indexing=True)
            @tf.function
            def sat_phi1(features):
                x = ltn.Variable("x", features)
                phi1 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_1]))), p=5)
                return phi1.tensor

            @tf.function
            def sat_phi2(features):
                x = ltn.Variable("x", features)
                phi2 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_2]))), p=5)
                return phi2.tensor

            @tf.function
            def sat_phi3(features):
                x = ltn.Variable("x", features)
                phi3 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_3]))), p=5)
                return phi3.tensor

            @tf.function
            def sat_phi4(features):
                x = ltn.Variable("x", features)
                phi4 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_4]))), p=5)
                return phi4.tensor

            @tf.function
            def sat_phi5(features):
                x = ltn.Variable("x", features)
                phi5 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_5]))), p=5)
                return phi5.tensor

            @tf.function
            def sat_phi6(features):
                x = ltn.Variable("x", features)
                phi6 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_6]))), p=5)
                return phi6.tensor

            @tf.function
            def sat_phi7(features):
                x = ltn.Variable("x", features)
                phi7 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_7]))), p=5)
                return phi7.tensor

            @tf.function
            def sat_phi8(features):
                x = ltn.Variable("x", features)
                phi8 = Forall(x, Implies(p([x, class_0]), Not(p([x, class_8]))), p=5)
                return phi8.tensor

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

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
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
                # Satisfaction levels for knowledge base axioms
                sat = axioms(features, labels)
                metrics_dict['test_sat_kb'](sat)

                # Satisfaction levels for individual phi queries
                metrics_dict['test_sat_phi1'](sat_phi1(features))
                metrics_dict['test_sat_phi2'](sat_phi2(features))
                metrics_dict['test_sat_phi3'](sat_phi3(features))
                metrics_dict['test_sat_phi4'](sat_phi4(features))
                metrics_dict['test_sat_phi5'](sat_phi5(features))
                metrics_dict['test_sat_phi6'](sat_phi6(features))
                metrics_dict['test_sat_phi7'](sat_phi7(features))
                metrics_dict['test_sat_phi8'](sat_phi8(features))

                # Accuracy
                predictions = model([features])
                metrics_dict['test_accuracy'](tf.one_hot(labels, 9), predictions)
            
            
            # Print overall class distribution before batching
            train_class_distribution = Counter(y_train_fold)
            val_class_distribution = Counter(y_val_fold)
            print(f"Training fold class distribution: {train_class_distribution}")
            print(f"Validation fold class distribution: {val_class_distribution}")
            
            X_train_fold_weighted = X_train_fold * np.array(normalized_average_importances)
            X_val_fold_weighted = X_val_fold * np.array(normalized_average_importances)
                
            ds_train_fold = tf.data.Dataset.from_tensor_slices((X_train_fold_weighted, y_train_fold))
            ds_val_fold = tf.data.Dataset.from_tensor_slices((X_val_fold_weighted, y_val_fold))
            
            ds_train_fold = ds_train_fold.batch(batch_size)
            ds_val_fold = ds_val_fold.batch(batch_size)

            for batch_features, batch_labels in ds_val_fold:
                batch_satisfaction_level = axioms(batch_features, batch_labels, training=False)
                print(f"Batch Satisfaction Level: {batch_satisfaction_level.numpy():.4f}")
                break
            

            
            results_path_ltn_fold = results_path_ltn + "_fold" + str(fold+1) + '_ltn.csv'
            
            start_time = time.time()
            commons.train(
                epochs,
                metrics_dict,
                ds_train_fold,
                ds_val_fold,
                train_step,
                test_step,
                csv_path=results_path_ltn_fold,
                track_metrics=1
            )
            end_time = time.time()
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")

            model.save_weights(os.path.join(model_save_directory, f"ltn_model_fold_{fold+1}_weights.h5"))
            model.save(os.path.join(model_save_directory, f"ltn_tf_model_fold_{fold+1}.tf"))
            
            
        
            
        if fold_metrics:
            # Calculate the average of each metric across all folds
            avg_accuracy = mean([metric[0] for metric in fold_metrics])
            avg_precision = mean([metric[1] for metric in fold_metrics])
            avg_recall = mean([metric[2] for metric in fold_metrics])
            avg_f1 = mean([metric[3] for metric in fold_metrics])

            # Append averaged metrics to the metrics_summary for overall analysis if needed
            metrics_summary.append((avg_accuracy, avg_precision, avg_recall, avg_f1))

            # Print the averages
            console.print(f"[bold magenta]Average metrics for across {n_splits} folds:[/]")
            console.print(f"Average Accuracy: {avg_accuracy:.4f}")
            console.print(f"Average Precision: {avg_precision:.4f}")
            console.print(f"Average Recall: {avg_recall:.4f}")
            console.print(f"Average F1: {avg_f1:.4f}\n")



                
                # if counter>2:
                #     break
            
        concatenate_and_delete_ltn_csv_files(results_path_ltn, "results_uoc/results_ltn.csv")

        console.print(f"[bold blue]Model for saved.[/]")
        console.print("[bold blue]Overall Averages Across All File Pairs:[/]")
        overall_avg_accuracy = mean([metrics[1] for metrics in metrics_summary])
        overall_avg_precision = mean([metrics[2] for metrics in metrics_summary])
        overall_avg_recall = mean([metrics[3] for metrics in metrics_summary])
        overall_avg_f1 = mean([metrics[4] for metrics in metrics_summary])

        console.print(f"Overall Average Accuracy: {overall_avg_accuracy:.4f}")
        console.print(f"Overall Average Precision: {overall_avg_precision:.4f}")
        console.print(f"Overall Average Recall: {overall_avg_recall:.4f}")
        console.print(f"Overall Average F1: {overall_avg_f1:.4f}")
        
        # After the loop, save the updated processed bases to file
        with open(processed_file_tracker, "w") as file:
            for base in processed_bases:
                file.write(base + "\n")
        
        

        
                
        

        

def clean_directory(directory):
    # Remove all files in the directory, needed when creating new sequences
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
            
if __name__ == "__main__":
    main()
