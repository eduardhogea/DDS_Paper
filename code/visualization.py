import numpy as np


def display_samples(sequences_file_path, labels_file_path, num_samples=1):
    """
    Displays a specified number of samples from the sequences and labels .npy files.
    
    Parameters:
    - sequences_file_path: Path to the .npy file containing sequences.
    - labels_file_path: Path to the .npy file containing labels.
    - num_samples: Number of samples to display. Default is 5.
    """
    # Load the sequences and labels
    sequences = np.load(sequences_file_path)
    labels = np.load(labels_file_path)
    
    # Determine the number of samples to display (cannot exceed the length of the data)
    num_samples = min(num_samples, len(sequences))
    
    # Display the specified number of samples
    for i in range(num_samples):
        print(f"Sample {i+1}:")
        print("Sequence:")
        print(sequences[i])
        print("Label:")
        print(labels[i])
        print("-" * 50)  # Separator for readability