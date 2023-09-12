#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm
import os
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from joblib import dump, load
import ltn
import csv
import math
import wandb

wandb.init(project="LCNC", name="CNN_backbone")


dataset_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU'

PGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/PGB'
RGB_path = '/home/ubuntu/dds_paper/DDS_Paper/data/DDS_Data_SEU/RGB'

# Specify the CSV file path
csv_file = '/home/ubuntu/dds_paper/DDS_Paper/data/data_robust.csv'
preprocessor_file = 'preprocessor.joblib'

train_path = '/home/ubuntu/dds_paper/DDS_Paper/data/train.csv'
val_path = '/home/ubuntu/dds_paper/DDS_Paper/data/val.csv'

np.random.seed(45)

# Set the chunk size for reading the CSV
chunk_size = 100000  # Adjust the chunk size according to your memory limitations


# In[ ]:


def extract_fault(file_name):
    match = re.search(r'\d+', file_name)
    if match:
        return int(match.group(0)[0])  # Extract the first digit
    else:
        return None

def process_files_to_csv(data_folders, output_file):
    # Check if the file already exists
    if os.path.isfile(output_file):
        print(f"File {output_file} already exists. Skipping processing.")
        return

    total_files = sum([len(files) for data_folder in data_folders for r, d, files in os.walk(data_folder)])

    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6', 'Channel7', 'Channel8', 'Speed', 'Type', 'Fault'])

        with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
            for data_folder in data_folders:
                for root, dirs, files in os.walk(data_folder):
                    if '.ipynb_checkpoints' in root:
                        continue  # Skip .ipynb_checkpoints folders
                    for file in files:
                        if file.endswith('.txt'):
                            file_path = os.path.join(root, file)
                            path_parts = file_path.split('\\')
                            variable_speed = 'Variable_speed' in file_path
                            type_index = -4 if variable_speed else -3
                            type = path_parts[type_index] if path_parts[type_index] in ['PGB', 'RGB'] else None
                            if type is not None:
                                speed_index = -3 if variable_speed else -2
                                speed = path_parts[speed_index]
                                fault = extract_fault(file)

                                data = pd.read_csv(file_path, sep='\t', encoding='ISO-8859-1')
                                reshaped_data = data.values[:, :]

                                for row_data in tqdm(reshaped_data, desc="Processing rows", unit="row", leave=False):
                                    row = row_data.tolist() + [speed, type, fault]
                                    csv_writer.writerow(row)
                            pbar.update()


# In[ ]:


# Example usage
process_files_to_csv([PGB_path, RGB_path], csv_file)


# In[ ]:


def split_and_sample_data(file_path, output_train, output_val, train_ratio=0.8, sample_fraction=0.01):
    # Check if the output files already exist
    if os.path.isfile(output_train) and os.path.isfile(output_val):
        print(f"Files {output_train} and {output_val} already exist. Skipping processing.")
        return
    
    chunksize = 100000
    total_lines = sum([80740352, 80740352, 80740352, 80740352, 80740352, 80740352, 80740352, 80740352, 80740352])
    total_chunks = math.ceil(total_lines / chunksize)

    reader = pd.read_csv(file_path, chunksize=chunksize)

    with open(output_train, 'w', newline='') as train_file, open(output_val, 'w', newline='') as val_file:
        train_writer = csv.writer(train_file)
        val_writer = csv.writer(val_file)
        
        for i, chunk in tqdm(enumerate(reader), total=total_chunks, desc="Processing chunks", unit="chunk"):
            chunk_sample = chunk.sample(frac=sample_fraction, random_state=1)
            if i == 0:
                train_writer.writerow(chunk_sample.columns.values)
                val_writer.writerow(chunk_sample.columns.values)

            train_data = chunk_sample.iloc[:int(train_ratio*len(chunk_sample))].values
            val_data = chunk_sample.iloc[int(train_ratio*len(chunk_sample)):].values

            train_writer.writerows(train_data)
            val_writer.writerows(val_data)

split_and_sample_data(csv_file, train_path, val_path)


# # Data preprocessing

# In[ ]:



# Initialize a dictionary to store the fault counts
fault_counts = {}

# Iterate through the CSV file using chunksize
with tqdm(total=1, unit='chunk', desc='Processing CSV') as pbar:
    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        
        #print(chunk)
        # Assuming there is a column named 'fault' in the CSV representing the fault type
        fault_column = 'Fault'

        # Count the occurrences of each fault in the current chunk
        fault_chunk_counts = chunk[fault_column].value_counts()

        # Aggregate the counts with the overall fault_counts dictionary
        for fault, count in fault_chunk_counts.items():
            fault_counts[fault] = fault_counts.get(fault, 0) + count

        pbar.update()

# Print the fault counts
for fault, count in fault_counts.items():
    print(f"Fault: {fault}, Count: {count}")


# In[ ]:


# Initialize dictionaries to store the counts
speed_counts = {}
type_counts = {}

# Iterate through the CSV file using chunksize
with tqdm(total=1, unit='chunk', desc='Processing CSV') as pbar:
    for chunk in pd.read_csv(train_path, chunksize=chunk_size):
        # Assuming there is a column named 'Speed' in the CSV representing the speed values
        
        speed_column = 'Speed'

        # Count the occurrences of each speed value in the current chunk
        speed_chunk_counts = chunk[speed_column].value_counts()

        # Aggregate the counts with the overall speed_counts dictionary
        for speed, count in speed_chunk_counts.items():
            speed_counts[speed] = speed_counts.get(speed, 0) + count

        # Assuming there is a column named 'Type' in the CSV representing the types
        type_column = 'Type'

        # Count the occurrences of each type in the current chunk
        type_chunk_counts = chunk[type_column].value_counts()

        # Aggregate the counts with the overall type_counts dictionary
        for typ, count in type_chunk_counts.items():
            type_counts[typ] = type_counts.get(typ, 0) + count

        pbar.update()

# Print the speed counts
print("Speed Counts:")
for speed, count in speed_counts.items():
    print(f"Speed: {speed}, Count: {count}")

# Print the type counts
print("Type Counts:")
for typ, count in type_counts.items():
    print(f"Type: {typ}, Count: {count}")


# In[ ]:


def data_generator(batch_size, data = csv_file):
    chunksize = batch_size
    for chunk in pd.read_csv(data, chunksize=chunksize):
        # One-hot encode the categorical features
        categorical_features = chunk[categorical_features_columns]
        categorical_features = one_hot_encoder.transform(categorical_features).toarray()
        
        # Concatenate with numerical features
        numerical_features = chunk[numerical_features_columns]
        X = np.concatenate([numerical_features, categorical_features], axis=1)

        sample_size = X.shape[0]
        #print(sample_size)
        time_steps = X.shape[1]
        #print(time_steps)
        input_dimensions = 1

        X_reshaped = X.reshape(sample_size,time_steps,input_dimensions)

        # Extract the labels
        y = chunk['Fault'].values

        yield X_reshaped, y

def data_generator_all(batch_size, data = csv_file):
    # Read all the data
    df = pd.read_csv(data)
    
    # Shuffle the DataFrame rows 
    df = df.sample(frac=1)

    # Calculate the number of batches
    num_batches = len(df) // batch_size

    for i in range(num_batches):
        batch = df.iloc[i*batch_size:(i+1)*batch_size]

        # One-hot encode the categorical features
        categorical_features = batch[categorical_features_columns]
        categorical_features = one_hot_encoder.transform(categorical_features).toarray()

        # Concatenate with numerical features
        numerical_features = batch[numerical_features_columns].values
        X = np.concatenate([numerical_features, categorical_features], axis=1)

        sample_size = X.shape[0]
        time_steps = X.shape[1]
        input_dimensions = 1

        X_reshaped = X.reshape(sample_size, time_steps, input_dimensions)

        # Extract the labels
        y = batch['Fault'].values

        yield X_reshaped, y


# Define the categories
# Define the categories
speed_categories = ['20_0','30_0','30_1','30_2', '30_3','30_4','30_5','40_0','50_0', 'Variable_speed']
type_categories = ['PGB', 'RGB']

# Initialize the OneHotEncoder
one_hot_encoder = OneHotEncoder(categories=[speed_categories, type_categories])

# Create a dummy dataset
dummy_df = pd.DataFrame(data=[['20_0', 'PGB']], columns=['Speed', 'Type'])

# Fit the encoder
one_hot_encoder.fit(dummy_df)

# Define your feature column names
categorical_features_columns = ['Speed', 'Type']
numerical_features_columns = ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6', 'Channel7', 'Channel8']

# batch_size = 2  # Adjust as necessary
# dataset = tf.data.Dataset.from_generator(data_generator, args=[batch_size], output_signature=(
#     tf.TensorSpec(shape=(batch_size, 1, 20), dtype=tf.float32),  # Update this to match the shape of X
#     tf.TensorSpec(shape=(batch_size,), dtype=tf.int32)
# ))


# In[ ]:


generator = data_generator_all(batch_size=256, data = train_path)
last_X, last_y = None, None

for sample_X, sample_y in generator:
    last_X, last_y = sample_X, sample_y
    print(last_X.shape)
    break

print(last_X)
print(last_y)


# ## CNN as backbone

# In[ ]:


import keras

def build_cnn_backbone():
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(last_X.shape[1],last_X.shape[2])))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=4, activation='elu', name="Conv1D_1"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='elu', name="Conv1D_2"))
    model.add(keras.layers.Conv1D(filters=16, kernel_size=2, activation='elu', name="Conv1D_3"))
    model.add(keras.layers.MaxPooling1D(pool_size=2, name="MaxPooling1D"))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(8, activation='relu', name="Dense_1"))
    model.add(keras.layers.Dense(9, name="Dense_2"))
    return model

# Build the CNN backbone model
model = build_cnn_backbone()

# Wrap the model in an LTN predicate
p = ltn.Predicate.FromLogits(model, activation_function="softmax", with_class_indexing=True)


# In[ ]:


# Constants to index/iterate on the classes
HEALTHY = ltn.Constant(0, trainable=False)
CTF = ltn.Constant(1, trainable=False)
MTF = ltn.Constant(2, trainable=False)
RCF = ltn.Constant(3, trainable=False)
SWF = ltn.Constant(4, trainable=False)
BWF = ltn.Constant(5, trainable=False)
CWF = ltn.Constant(6, trainable=False)
IRF = ltn.Constant(7, trainable=False)
ORF = ltn.Constant(8, trainable=False)


# In[ ]:


#operators
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")


# In[ ]:


# Define the formula aggregator
formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

@tf.function
def axioms(features, labels, training=False):
    # Variables for each class
    x_healthy = ltn.Variable("x_healthy", features[labels == 0])
    x_ctf = ltn.Variable("x_ctf", features[labels == 1])
    x_mtf = ltn.Variable("x_mtf", features[labels == 2])
    x_rcf = ltn.Variable("x_rcf", features[labels == 3])
    x_swf = ltn.Variable("x_swf", features[labels == 4])
    x_bwf = ltn.Variable("x_bwf", features[labels == 5])
    x_cwf = ltn.Variable("x_cwf", features[labels == 6])
    x_irf = ltn.Variable("x_irf", features[labels == 7])
    x_orf = ltn.Variable("x_orf", features[labels == 8])

    # Fault list for mutual exclusivity axioms
    faults = [HEALTHY, CTF, MTF, RCF, SWF, BWF, CWF, IRF, ORF]
    fault_vars = [x_healthy, x_ctf, x_mtf, x_rcf, x_swf, x_bwf, x_cwf, x_irf, x_orf]

    axioms = []
    for i, fault in enumerate(faults):
        # Add the axiom that for all x of a certain fault, the probability of that fault should be high
        axioms.append(Forall(fault_vars[i], p([fault_vars[i], fault], training=training)))

        # Add the axioms for mutual exclusivity
        for j, other_fault in enumerate(faults):
            if i != j:
                axioms.append(Forall(fault_vars[i], Not(p([fault_vars[i], other_fault], training=training))))

    sat_level = formula_aggregator(axioms).tensor
    return sat_level


# In[ ]:



# Print the initial satisfaction level for each batch of the test dataset
for sample_X, sample_y in generator:
    #sample_X_reshaped = tf.reshape(sample_X, (sample_X.shape[0], sample_X.shape[1], 1))
    #print(sample_X)
    print("Initial sat level %.5f" % axioms(sample_X, sample_y))
    break


# In[ ]:


metrics_dict = {
    'train_sat_kb': tf.keras.metrics.Mean(name='train_sat_kb'),
    'test_sat_kb': tf.keras.metrics.Mean(name='test_sat_kb'),
    'train_accuracy': tf.keras.metrics.CategoricalAccuracy(name="train_accuracy"),
    'test_accuracy': tf.keras.metrics.CategoricalAccuracy(name="test_accuracy")
}


# In[ ]:


@tf.function
def train_step(features, labels, optimizer):
    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(features, labels, training=True)
        loss = 1. - sat
    gradients = tape.gradient(loss, p.trainable_variables)
    optimizer.apply_gradients(zip(gradients, p.trainable_variables))
    sat = axioms(features, labels)  # compute sat without dropout
    metrics_dict['train_sat_kb'](sat)
    # accuracy
    predictions = model([features])
    metrics_dict['train_accuracy'](labels, predictions)

@tf.function
def test_step(features, labels, optimizer):
    # sat
    sat = axioms(features, labels)
    metrics_dict['test_sat_kb'](sat)
    # accuracy
    predictions = model([features])
    metrics_dict['test_accuracy'](labels, predictions)


# In[ ]:


# Sweep configuration
sweep_config = {
    "name": "dds-sweep",
    "method": "random",
    "metric": {
        "name": "test_sat_kb",
        "goal": "maximize"
    },
    "parameters": {
        "batch_size": {
            "values": [32, 64, 128, 256, 1024, 4096]
        },
        "epochs": {
            "values": [5]
        },
        "learning_rate": {
            "min": 0.0001,
            "max": 0.1
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="dds-paper")


# In[ ]:


from collections import defaultdict
import time
from tqdm import tqdm
import wandb  # Import wandb

def train(epochs, metrics_dict, train_generator_func, test_generator_func, train_step, test_step,
          num_train_steps, num_test_steps, track_metrics=1, csv_path=None, scheduled_parameters=defaultdict(lambda: {}), optimizer=None):
 
    template = "Epoch {}"
    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label

    if csv_path is not None:
        csv_file = open(csv_path, "w+")
        headers = ",".join(["Epoch"] + list(metrics_dict.keys()))
        csv_template = ",".join(["{}" for _ in range(len(metrics_dict) + 1)])
        csv_file.write(headers + "\n")

    epoch_times = []
    start_time = time.time()

    for epoch in range(epochs):
        # Reset metrics
        for metrics in metrics_dict.values():
            metrics.reset_states()

        # Training loop
        train_generator = train_generator_func()
        pbar = tqdm(total=num_train_steps)
        for batch_elements in train_generator:
            train_step(*batch_elements,optimizer=optimizer, **scheduled_parameters[epoch])

            # Log training metrics to wandb
            wandb.log({"Training Satisfaction": metrics_dict['train_sat_kb'].result().numpy(),
                       "Training Accuracy": metrics_dict['train_accuracy'].result().numpy()})
            
            pbar.update()
        pbar.close()

        # Validation loop
        test_generator = test_generator_func()
        pbar = tqdm(total=num_test_steps)
        for batch_elements in test_generator:
            test_step(*batch_elements, optimizer=optimizer, **scheduled_parameters[epoch])

            # Log testing metrics to wandb
            wandb.log({"Validation Satisfaction": metrics_dict['test_sat_kb'].result().numpy(),
                       "Validation Accuracy": metrics_dict['test_accuracy'].result().numpy()})

            pbar.update()
        pbar.close()

        # Additional logging
        if csv_path is not None:
            metrics_results = [metrics.result() for metrics in metrics_dict.values()]
            csv_file.write(csv_template.format(epoch, *metrics_results) + "\n")
            csv_file.flush()

        end_time = time.time()
        epoch_times.append(end_time - start_time)
        start_time = end_time

    if csv_path is not None:
        csv_file.close()

    return epoch_times


# In[ ]:


def train_wrapper():
    # Initialize Wandb for this sweep run
    run = wandb.init()

    # Use hyperparameters from wandb.config
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Your existing setup code
    num_train_steps = sum(1 for _ in data_generator(batch_size, data=train_path))
    num_test_steps = sum(1 for _ in data_generator(batch_size, data=val_path))

    def train_generator_func():
        return data_generator_all(batch_size=batch_size, data=train_path)

    def val_generator_func():
        return data_generator_all(batch_size=batch_size, data=val_path)

    # Call your existing train function
    epoch_times = train(
        epochs,
        metrics_dict,
        train_generator_func,
        val_generator_func,
        train_step,
        test_step,
        csv_path="/home/ubuntu/dds_paper/DDS_Paper/data/final.csv",
        track_metrics=1,
        num_train_steps=num_train_steps,
        num_test_steps=num_test_steps,
        optimizer=optimizer 
    )


# In[ ]:


wandb.agent(sweep_id, function=train_wrapper)


# In[ ]:


# batch_size = 256

# EPOCHS = 5

# # Get the number of training and testing steps per epoch
# num_train_steps = sum(1 for _ in data_generator(batch_size, data='train.csv'))
# num_test_steps = sum(1 for _ in data_generator(batch_size, data='val.csv'))

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# # Define your generator functions
# def train_generator_func():
#     return data_generator_all(batch_size=batch_size, data='train.csv')

# def val_generator_func():
#     return data_generator_all(batch_size=batch_size, data='val.csv')


# In[ ]:



# training = train(
#     EPOCHS,
#     metrics_dict,
#     train_generator_func,
#     val_generator_func,
#     train_step,
#     test_step,
#     csv_path="final.csv",
#     track_metrics=1,
#     num_train_steps=num_train_steps,
#     num_test_steps=num_test_steps
# )

