"""
Vor7reX Performance Analytics & Diagnostics Suite.
Generates advanced telemetry visualizations, including Confusion Matrices 
and Classification Reports to identify inter-class misidentification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataSet import DataSet
from train_model import Model
from read_data import read_name_list

class PerformanceAnalyzer:
    """
    Diagnostic engine for the Vor7reX framework.
    Translates neural output into actionable visual intelligence.
    """

    def __init__(self):
        self.model_wrapper = Model()
        self.model_wrapper.load_model()
        self.img_size = 128
        
        # Load dataset metadata
        dataset_path = os.path.join('.', 'pictures', 'dataset')
        self.name_list = read_name_list(dataset_path)

    def generate_report(self, dataset):
        """
        Executes a full validation sweep and generates diagnostic plots.
        """
        print("VOR7REX ANALYTICS: Initializing validation sweep...")
        
        # Extract ground truth and execute predictions
        y_true = np.argmax(dataset.Y_test, axis=1)
        
        # Predicting in batches for memory efficiency
        predictions = self.model_wrapper.model.predict(dataset.X_test)
        y_pred = np.argmax(predictions, axis=1)

        # 1. Generate Textual Classification Report
        print("\n--- CLASSIFICATION METRICS ---")
        print(classification_report(y_true, y_pred, target_names=self.name_list))

        # 2. Compute Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # 3. Visualization Pipeline
        self._plot_confusion_matrix(cm)

    def _plot_confusion_matrix(self, cm):
        """
        Renders a high-resolution heatmap for error analysis.
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.name_list, 
                    yticklabels=self.name_list)
        
        plt.title('Vor7reX Neural Confusion Matrix', fontsize=16)
        plt.ylabel('Actual Identity', fontsize=12)
        plt.xlabel('Predicted Identity', fontsize=12)
        
        # Save output to disk
        output_file = 'performance_telemetry.png'
        plt.savefig(output_file)
        print(f"DIAGNOSTICS COMPLETE: Telemetry saved to {output_file}")

if __name__ == '__main__':
    # Initialize Data Pipeline
    path = os.path.join('.', 'pictures', 'dataset')
    data = DataSet(path)
    
    # Run Analytics
    analyzer = PerformanceAnalyzer()
    analyzer.generate_report(data)