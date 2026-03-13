"""
Vor7reX Model Validation & Benchmarking Suite.
Standardized utility for evaluating inference performance on single-sample 
inputs and batch validation datasets.
"""

from read_data import read_name_list, read_file
from train_model import Model
import cv2
import os

def test_onePicture(path):
    """
    Executes a single-point inference test on a target image.
    Validates the full pre-processing pipeline and model prediction accuracy.
    """
    model = Model()
    model.load_model() # Load pre-trained Vor7reX weights
    
    if not os.path.exists(path):
        print(f"I/O ERROR: Target sample {path} not found.")
        return

    # Ingesting raw sample
    img = cv2.imread(path)
    
    # Pre-processing pipeline: spatial scaling to 128px and grayscale conversion
    img_size = 128 
    img_resized = cv2.resize(img, (img_size, img_size))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Execute neural inference
    picType, prob = model.predict(img_gray)
    
    if picType != -1:
        dataset_path = os.path.join('.', 'pictures', 'dataset')
        name_list = read_name_list(dataset_path)
        print(f"INFERENCE RESULT: [{name_list[picType]}] - Confidence: {prob*100:.2f}%")
    else:
        print("INFERENCE RESULT: Identity unverified (Stranger)")

def test_onBatch(path):
    """
    Automated batch validation against a labeled ground-truth dataset.
    Calculates global accuracy metrics and identifies classification errors.
    """
    model = Model()
    model.load_model()
    
    success_count = 0
    img_list, label_list, counter = read_file(path)
    
    if len(img_list) == 0:
        print("VALIDATION ERROR: No valid tensors found in target path.")
        return 0

    dataset_path = os.path.join('.', 'pictures', 'dataset')
    name_list = read_name_list(dataset_path)

    print(f"--- VOR7REX BATCH VALIDATION: Processing {len(img_list)} samples ---")
    
    for i, img in enumerate(img_list):
        picType, prob = model.predict(img)
        actual_label = label_list[i]
        
        if picType != -1:
            predicted_name = name_list[picType]
            actual_name = name_list[actual_label]
            
            if picType == actual_label:
                success_count += 1
                print(f"SAMPLE [{i}] - STATUS: MATCH - Identity: {predicted_name} ({prob*100:.1f}%)")
            else:
                print(f"SAMPLE [{i}] - STATUS: MISMATCH - Predicted: {predicted_name} | Actual: {actual_name}")
        else:
            print(f"SAMPLE [{i}] - STATUS: UNKNOWN - Confidence below threshold.")

    # Calculate global performance metrics
    accuracy = (success_count / len(img_list)) * 100
    print(f"--- VALIDATION COMPLETE ---")
    print(f"Global System Accuracy: {accuracy:.2f}% ({success_count}/{len(img_list)})")
    return success_count

if __name__ == '__main__':
    # Internal benchmark configuration
    path_to_test = os.path.join('.', 'pictures', 'dataset')
    test_onBatch(path_to_test)
    
    # Default: Single-sample diagnostic test
    #test_image = os.path.join('.', 'pictures', 'test_sample.jpg')
    #test_onePicture(test_image)