"""
Vor7reX Real-Time Inference & Computer Vision Suite.
Integrates OpenCV video streams with deep learning models to perform
live face detection, preprocessing (CLAHE), and neural classification.
"""

import cv2
import os
import numpy as np
from train_model import Model
from read_data import read_name_list

class Camera_reader(object):
    """
    Hardware interface and inference orchestrator.
    Handles camera I/O, frame processing, and neural network injection.
    """
    
    def __init__(self):
        """Initializes the Vor7reX engine and restores trained weights."""
        self.model_wrapper = Model()
        self.model_wrapper.load_model() 
        self.img_size = 128

    def build_camera(self):
        """
        Main execution loop for tactical face recognition.
        Implements real-time CLAHE equalization and adaptive thresholding.
        """
        # Load Haar Cascade: Pre-trained spatial feature detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load Identity Mapping: Links neural indices to human names
        dataset_path = os.path.join('.', 'pictures', 'dataset')
        name_list = read_name_list(dataset_path)

        # Hardware Abstraction Layer: Camera Initialization
        cameraCapture = cv2.VideoCapture(0)
        cameraCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cameraCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Mitigates lighting inconsistencies and screen glare from devices like iPads
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        print("VOR7REX MISSION CONTROL: System Live. Threshold: 0.5. Press 'Q' to abort.")

        frame_count = 0
        faces = []

        while True:
            success, frame = cameraCapture.read()
            if not success: 
                print("TELEMETRY ERROR: Failed to grab frame.")
                break

            # Optimization: Skip face detection every other frame to maintain high FPS
            if frame_count % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply contrast enhancement for superior feature definition
                gray = clahe.apply(gray)
                # Detect objects of different sizes in the input image
                faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                # Region of Interest (ROI) extraction
                ROI = gray[y:y + h, x:x + w]
                
                try:
                    # 1. Spatial Normalization: Match training dimensions (128x128)
                    ROI = cv2.resize(ROI, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                    
                    # 2. Intensity Normalization: Scale pixel values to [0, 1] range
                    # CRITICAL: Ensures parity with training-time data distribution
                    ROI = ROI.astype("float32") / 255.0
                    
                    # 3. Tensor Reshaping: Add Batch and Channel dimensions for Keras (1, 128, 128, 1)
                    ROI = np.expand_dims(ROI, axis=0)
                    ROI = np.expand_dims(ROI, axis=-1)
                    
                    # 4. Neural Inference: Execute forward pass and extract confidence scores
                    predictions = self.model_wrapper.model.predict(ROI, verbose=0)
                    label = np.argmax(predictions)
                    prob = np.max(predictions)
                    
                    # 5. Decision Logic & Visualization
                    # Balanced Threshold: 0.50 (Mitigates False Positives)
                    if prob >= 0.50:
                        show_name = name_list[label]
                        # UI Feedback: Green for high confidence (>75%), Yellow for low
                        color = (0, 255, 0) if prob > 0.75 else (0, 255, 255)
                    else:
                        show_name = 'Stranger'
                        color = (0, 0, 255) # Red for unauthorized/unknown
                        
                    # Overlay UI Elements on the frame
                    text = f"{show_name} {prob*100:.1f}%"
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                except Exception as e:
                    # Generic error handling to prevent runtime crashes during inference
                    continue

            # Display real-time telemetry feed
            cv2.imshow("Vor7reX Tactical Monitor", frame)
            frame_count += 1

            # System Abort Key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("VOR7REX MISSION CONTROL: User Abort Signal Received.")
                break

        # Resource Cleanup
        cameraCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize and execute the Tactical Reader
    camera = Camera_reader()
    camera.build_camera()