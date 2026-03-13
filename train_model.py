"""
Vor7reX Convolutional Neural Network (CNN) Training Suite.
Implements a deep architecture for multi-class facial recognition with 
automated performance telemetry and high-resolution diagnostic plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataSet import DataSet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

class Model(object):
    """
    Main architectural class for the Vor7reX inference engine.
    Handles model construction, training orchestration, and asset generation.
    """
    FILE_PATH = os.path.join('.', 'model_vor7rex.h5') 
    IMAGE_SIZE = 128

    def __init__(self):
        self.model = None

    def read_trainData(self, dataset):
        """Injects normalized tensor data into the model object."""
        self.dataset = dataset

    def build_model(self):
        """Deep CNN Topology: Hierarchical feature extraction stack."""
        self.model = Sequential()
        
        # Block 1
        self.model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        # Block 2
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.3))

        # Block 3
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.4))

        # Classifier
        self.model.add(Flatten())
        self.model.add(Dense(1024)) 
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        
        self.model.summary()

    def train_model(self):
        """Executes the training pipeline and exports diagnostic assets."""
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        print(f"VOR7REX TRAINING SESSION: Processing {self.dataset.num_classes} identity classes...")
        
        # Capture training telemetry in history object
        history = self.model.fit(
            datagen.flow(self.dataset.X_train, self.dataset.Y_train, batch_size=32),
            validation_data=(self.dataset.X_test, self.dataset.Y_test),
            epochs=150
        )

        # Automated performance plotting for README documentation
        self._generate_performance_plot(history)

    def _generate_performance_plot(self, history):
        """Renders and saves high-resolution metrics after training session."""
        print("VOR7REX VISUALIZER: Exporting README performance assets...")
        
        plt.figure(figsize=(16, 6))
        
        # Accuracy Convergence
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='teal', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='coral', linestyle='--')
        plt.title('Vor7reX Model Convergence', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Error Decay
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', color='cyan', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', color='gray', linestyle='--')
        plt.title('Vor7reX Error Decay', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('vor7rex_performance.png', dpi=150)
        print("DIAGNOSTICS COMPLETE: Asset 'vor7rex_performance.png' generated successfully.")

    def evaluate_model(self):
        """Performance validation against non-training telemetry."""
        print('\n--- VOR7REX FINAL EVALUATION ---')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)
        print(f'Final Validation Accuracy: {accuracy*100:.2f}%')

    def predict(self, image):
        """Real-time inference entry point."""
        image = image.reshape(1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1).astype('float32') / 255.0
        result = self.model.predict(image)
        max_index = np.argmax(result)
        return max_index, result[0][max_index]

    def save_model(self, file_path=FILE_PATH):
        """Serializes neural weights."""
        self.model.save(file_path)
        print(f"MODEL PERSISTENCE: Data exported to {file_path}")

    def load_model(self, file_path=FILE_PATH):
        """Restores model state."""
        self.model = load_model(file_path)
        print(f"MODEL RESTORATION: {file_path} loaded successfully.")

if __name__ == '__main__':
    path = os.path.join('.', 'pictures', 'dataset')
    data = DataSet(path)
    if data.num_classes is not None:
        face_ia = Model()
        face_ia.read_trainData(data)
        face_ia.build_model()
        face_ia.train_model()
        face_ia.evaluate_model()
        face_ia.save_model()