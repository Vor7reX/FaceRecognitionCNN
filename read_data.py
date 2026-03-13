"""
Vor7reX Data Ingestion & Indexing Module.
Provides low-level I/O operations to translate directory structures into 
structured training tensors and identity mappings.
"""

import os
import cv2
import numpy as np
from read_img import endwith

def read_file(path):
    """
    Scans the repository: each subdirectory represents a unique identity class.
    Returns: image_tensor (numpy ndarray), label_vector (list), class_count (int).
    """
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128

    # Path integrity verification to prevent runtime exceptions
    if not os.path.exists(path):
        print(f"I/O ERROR: Target path {path} is unreachable.")
        return np.array([]), [], 0

    # Directory traversal: iterating through unique identity containers
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        
        # Filtering non-directory artifacts
        if not os.path.isdir(child_path):
            continue

        print(f"BUFFERING: Class ID {dir_counter} identified as [{child_dir}]")

        for dir_image in os.listdir(child_path):
            # Format validation: enforcing JPEG/PNG compliance
            if endwith(dir_image, 'jpg') or endwith(dir_image, 'png') or endwith(dir_image, 'JPG'):
                img_path = os.path.join(child_path, dir_image)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Pre-processing pipeline: spatial scaling and grayscale transformation
                    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                    
                    img_list.append(recolored_img)
                    label_list.append(dir_counter)
                else:
                    print(f"FILE WARNING: Header corruption in {img_path}")

        dir_counter += 1

    # Final matrix transformation for CNN ingestion
    img_list = np.array(img_list)
    
    return img_list, label_list, dir_counter

def read_name_list(path):
    """
    Maps folder identifiers to human-readable identity strings.
    Used for UI/Overlay labels during real-time inference.
    """
    name_list = []
    if os.path.exists(path):
        for child_dir in os.listdir(path):
            if os.path.isdir(os.path.join(path, child_dir)):
                name_list.append(child_dir)
    return name_list

if __name__ == '__main__':
    # Local integration test for data integrity verification
    dataset_path = os.path.join('.', 'pictures', 'dataset')
    imgs, labels, counter = read_file(dataset_path)
    print(f"--- VOR7REX DATA INGESTION STATUS ---")
    print(f"Payload size: {len(imgs)} samples")
    print(f"Class distribution: {counter} identities indexed")