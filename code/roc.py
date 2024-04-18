import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize

def plot_roc_curves(model, X_test, y_test, class_labels):
    # Load the trained model
    
    # Predict probabilities for each class
    y_score = model.predict(X_test)
    
    # Binarize the labels for each class
    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_labels)))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, label in enumerate(class_labels):
        fpr[label], tpr[label], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    # Plot all ROC curves
    plt.figure()
    for label in class_labels:
        plt.plot(fpr[label], tpr[label],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(label, roc_auc[label]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-Class')
    plt.legend(loc="lower right")
    plt.show()
