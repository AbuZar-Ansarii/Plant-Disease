import cv2 
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle

# Load class names
try:
    with open("D:\pandas\class_names (1).pkl", "rb") as f:
        class_names = pickle.load(f)
    print(f"Loaded {len(class_names)} class names successfully.")
except FileNotFoundError:
    print("Error: class_names.pkl file not found.")
    exit()
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

# Load the pre-trained model
try:
    model = load_model('plant_model.h5')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: plant_model.h5 file not found.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize video capture
url = 'http://192.168.31.93:8080/video'
cap = cv2.VideoCapture(url)

# Check if video capture is successfully opened
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set desired frame dimensions (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        # Preprocess the frame for prediction
        resized_frame = cv2.resize(frame, (256, 256))
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.reshape(normalized_frame, (1, 256, 256, 3))

        # Make predictions
        predictions = model.predict(reshaped_frame, verbose=0)  # Disable prediction logging
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class name if available, otherwise use index
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = str(predicted_class_idx)

        # Display the prediction on the frame
        cv2.putText(frame, f'Disease: {predicted_class}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw rectangle around detection area
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        cv2.putText(frame, "Prediction Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Plant Disease Detection', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()