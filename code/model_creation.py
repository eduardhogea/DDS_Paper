import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers

def lr_schedule(epoch, lr):
    if epoch % 50 == 0 and epoch > 0:
        return lr * 0.9
    return lr

def create_model(input_shape, num_classes, reg_type='l2', reg_value=0.001, return_logits=False):
    if reg_type == 'l2':
        regularizer = regularizers.l2(reg_value)
    elif reg_type == 'l1':
        regularizer = regularizers.l1(reg_value)
    else:
        raise ValueError("Invalid regularizer type. Choose 'l1' or 'l2'.")

    model = Sequential([
        Input(shape=input_shape),
        LSTM(4098, return_sequences=True, kernel_regularizer=regularizer),
        Dropout(0.3),
        LSTM(2056, return_sequences=True, kernel_regularizer=regularizer),
        Dropout(0.3),
        LSTM(1024, kernel_regularizer=regularizer),
        Dropout(0.3),
        Dense(512, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=regularizer),
        Dropout(0.3)
    ])
    
    if return_logits:
        model.add(Dense(num_classes))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    
    if return_logits:
        model.compile(optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class LSTMModel(Model):
    """LSTM-based model for sequence processing."""
    def __init__(self, input_shape, num_classes, reg_type='l2', reg_value=0.001, return_logits=False):
        super(LSTMModel, self).__init__()
        
        if reg_type == 'l2':
            regularizer = regularizers.l2(reg_value)
        elif reg_type == 'l1':
            regularizer = regularizers.l1(reg_value)
        else:
            raise ValueError("Invalid regularizer type. Choose 'l1' or 'l2'.")
        
        self.sequence = [
            LSTM(256, return_sequences=True, kernel_regularizer=regularizer, input_shape=input_shape),
            Dropout(0.3),
            LSTM(128, return_sequences=True, kernel_regularizer=regularizer),
            Dropout(0.3),
            LSTM(64, kernel_regularizer=regularizer),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=regularizer),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_regularizer=regularizer),
            Dropout(0.3)
        ]
        
        self.return_logits = return_logits
        if return_logits:
            self.logits_layer = Dense(num_classes)
        else:
            self.probabilities_layer = Dense(num_classes, activation='softmax')
        
        #self.compile_model()
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.sequence:
            x = layer(x, training=training)
        if self.return_logits:
            x = self.logits_layer(x)
        else:
            x = self.probabilities_layer(x)
        return x
    
    def compile_model(self):
        if self.return_logits:
            self.compile(optimizer=Adam(learning_rate=0.001),
                        loss=SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        else:
            self.compile(optimizer=Adam(learning_rate=0.001),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])