# -*- coding: utf-8 -*-
"""
Vor7reX Feature Extraction Engine.
Automates facial localization, segmentation, and grayscale standardization 
to generate high-fidelity training datasets from raw imagery.
"""

import os
import cv2
import time
from read_img import readAllImg

def readPicSaveFace(sourcePath, objectPath, *suffix):
    """
    Parses a source directory, executes Haar Cascade detection, 
    and exports standardized 128x128 grayscale ROI tensors.
    """
    count = 1
    
    # Initialize the frontal face classifier via OpenCV's hardware abstraction
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"KERNEL ERROR: Resource not found at {cascade_path}")
        return 0

    try:
        # Batch load images from source directory using the Vor7reX I/O module
        resultArray = readAllImg(sourcePath, *suffix)
        
        faces_found_in_folder = 0
        for i in resultArray:
            if not isinstance(i, str):
                # Conversion to 1-channel grayscale: Essential for Vor7reX model architecture
                gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                
                # Multi-scale detection: optimized with 1.2 scale factor and 50px floor limit
                faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
                
                for (x, y, w, h) in faces:
                    # Generate unique timestamped identifiers to prevent collisions
                    fileName = f"face_{int(time.time())}_{count}.jpg"
                    
                    # ROI Segmentation & Standardization: Force 128x128 resolution for model parity
                    face_img = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
                    
                    save_path = os.path.join(objectPath, fileName)
                    cv2.imwrite(save_path, face_img)
                    
                    count += 1
                    faces_found_in_folder += 1
        return faces_found_in_folder

    except Exception as e:
        print(f"RUNTIME ERROR in {sourcePath}: {e}")
        return 0

if __name__ == '__main__':
    # Configuration of the Vor7reX local storage pipeline
    raw_base_dir = os.path.join('.', 'pictures', 'raw')
    dataset_base_dir = os.path.join('.', 'pictures', 'dataset')

    if not os.path.exists(raw_base_dir):
        print(f"CRITICAL ERROR: Source path {raw_base_dir} is unreachable.")
    else:
        # Directory traversal: Identify target identity sub-folders
        folders = [f for f in os.listdir(raw_base_dir) if os.path.isdir(os.path.join(raw_base_dir, f))]
        print(f"VOR7REX PIPELINE: Processing {len(folders)} identity classes...")

        for person_name in folders:
            source_dir = os.path.join(raw_base_dir, person_name)
            dest_dir = os.path.join(dataset_base_dir, person_name)

            # Ensure recursive directory integrity
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            print(f" Parsing [{person_name}]...", end=" ", flush=True)
            num = readPicSaveFace(source_dir, dest_dir, '.jpg', '.JPG', '.png', '.PNG')
            print(f"Success! {num} samples extracted.")

        print("\n--- VOR7REX DATASET GENERATION COMPLETE ---")