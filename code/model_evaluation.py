import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from model_creation import create_model

def normalize_importances(importances):
    """
    Normalize a list of feature importances so that they sum to 1,
    ensuring all values are positive. This function now expects
    a list (or numpy array) of numerical importances as its input.
    """
    # Convert the importances to a numpy array to ensure compatibility with numpy operations
    importances = np.array(importances)
    abs_importances = np.abs(importances)  # Take the absolute values to ensure all values are positive
    total_importance = np.sum(abs_importances)
    
    # Avoid division by zero by checking if the total_importance is not zero
    if total_importance != 0:
        normalized_importances = abs_importances / total_importance
    else:
        # If the total is 0 (e.g., all importances are 0), return the original importances
        normalized_importances = abs_importances
    
    # Return the normalized importances, ensuring it's converted back to a list if needed
    return normalized_importances.tolist()


def calculate_class_accuracies(predictions, true_labels):
    # Assuming predictions are probabilities, get the predicted class indices
    pred_labels = np.argmax(predictions, axis=1)
    
    # true_labels are already integer labels, no need for np.argmax

    # Initialize a dictionary to store accuracy for each class present in true_labels
    class_accuracies = {}
    
    for class_index in np.unique(true_labels):  # Loop only through classes present in true_labels
        class_mask = true_labels == class_index
        
        # Calculate accuracy for the current class
        class_accuracies[class_index] = accuracy_score(true_labels[class_mask], pred_labels[class_mask])
    
    return class_accuracies


def permutation_importance_per_class(model, X_val, y_val, n_repeats=10, n_samples=None):
    n_samples = n_samples if n_samples is not None else X_val.shape[0]
    random_indices = np.random.choice(X_val.shape[0], size=n_samples, replace=False)
    X_val_subset = X_val[random_indices]
    y_val_subset = y_val[random_indices]
    
    # Get baseline class-specific accuracies
    baseline_predictions = model.predict(X_val_subset, verbose = 0)
    baseline_class_accuracies = calculate_class_accuracies(baseline_predictions, y_val_subset)
    
    # Prepare storage for importances, using a dictionary to accommodate variable class presence
    feature_importances = {class_index: np.zeros((X_val.shape[2], n_repeats)) for class_index in baseline_class_accuracies.keys()}
    
    for feature_index in tqdm(range(X_val.shape[2]), desc='Calculating Feature Importance'):
        for n in range(n_repeats):
            saved_feature = X_val_subset[:, :, feature_index].copy()
            np.random.shuffle(X_val_subset[:, :, feature_index])
            
            permuted_predictions = model.predict(X_val_subset, verbose = 0)
            permuted_class_accuracies = calculate_class_accuracies(permuted_predictions, y_val_subset)
            
            for class_index in baseline_class_accuracies.keys():
                feature_importances[class_index][feature_index, n] = baseline_class_accuracies[class_index] - permuted_class_accuracies.get(class_index, 0)
            
            X_val_subset[:, :, feature_index] = saved_feature
    
    # Average the importance scores across repeats and prepare formatted output
    average_importances = {class_index: importances.mean(axis=1) for class_index, importances in feature_importances.items()}
    
    # Format the output
    formatted_importances = {f"Class {class_index}": importance.tolist() for class_index, importance in average_importances.items()}
    return formatted_importances


            
def kfold_cross_validation(X, y, num_folds=5):
    input_shape = X.shape[1:]  # Assuming X is (num_samples, time_steps, features)
    num_classes = y.shape[1]
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_no = 1
    for train, test in kfold.split(X, y):
        print(f"Training on fold {fold_no}...")
        
        model = create_model(input_shape, num_classes, l2_reg=config.reg_value)
        lr_scheduler = LearningRateScheduler(config.lr_schedule, verbose=0)
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        
        model.fit(X[train], y[train], validation_data=(X[test], y[test]),
                epochs=config.epochs, batch_size=config.batch_size, callbacks=[early_stopping, lr_scheduler], verbose=1)
        
        fold_no += 1