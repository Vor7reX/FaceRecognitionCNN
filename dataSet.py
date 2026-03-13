
"""
Vor7reX Data Management Module.
Handles image tensor acquisition, partitioning (Train/Test split), 
and grayscale normalization for CNN ingestion.
"""

from read_data import read_file
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import random

class DataSet(object):
    """
    Automated pipeline for dataset preprocessing. 
    Standardizes raw imagery into normalized tensors ready for neural training.
    """
    
    def __init__(self, path):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.img_size = 128
        
        # Trigger automated data extraction pipeline upon instantiation
        self.extract_data(path)

    def extract_data(self, path):
        """
        Executes data ingestion, stochastic partitioning, and feature scaling.
        """
        
        # Load raw matrices and identity labels from local repository
        imgs, labels, counter = read_file(path)

        # Stochastic Partitioning: 80% allocated to Training, 20% to Validation/Test
        X_train, X_test, y_train, y_test = train_test_split(
            imgs, labels, 
            test_size=0.2, 
            random_state=random.randint(0, 100)
        )

        # Tensor Reshaping: Formatting for 1-channel (grayscale) CNN input
        # Feature Scaling: Pixel values normalized from [0-255] to [0.0-1.0] range
        X_train = X_train.reshape(X_train.shape[0], self.img_size, self.img_size, 1).astype('float32') / 255.0
        X_test = X_test.reshape(X_test.shape[0], self.img_size, self.img_size, 1).astype('float32') / 255.0

        # One-Hot Encoding: Converting integer labels to categorical probability matrices
        Y_train = to_categorical(y_train, num_classes=counter)
        Y_test = to_categorical(y_test, num_classes=counter)
        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

    def check(self):
        """
        Dataset diagnostic utility for structural integrity verification.
        """
        print(f'--- VOR7REX DATASET METRICS ---')
        print(f'Training Set - Shape: {self.X_train.shape}')
        print(f'Test Set     - Shape: {self.X_test.shape}')
        print(f'Total Classes Detected: {self.num_classes}')