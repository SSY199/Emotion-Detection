# import cv2
# import numpy as np
# from keras.models import load_model

# # Don’t compile on load – we won't retrain it
# model = load_model("emotion_model.h5", compile=False)
# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# # Emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for x, y, w, h in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_gray = cv2.resize(roi_gray, (48, 48))
#         roi_gray = roi_gray.astype("float") / 255.0
#         roi_gray = np.expand_dims(roi_gray, axis=0)
#         roi_gray = np.expand_dims(roi_gray, axis=-1)

#         prediction = model.predict(roi_gray, verbose=0)
#         max_index = int(np.argmax(prediction))
#         emotion = emotion_labels[max_index]

#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow("Emotion Detector", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and Haar cascade
try:
    model = load_model("emotion_model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error loading Haar cascade file.")
    exit()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    # If no frame is captured, exit the loop
    if not ret:
        print("Error: Failed to capture image. Exiting.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize the region of interest (ROI) to 64x64, not 48x48
        roi_gray = cv2.resize(roi_gray, (64, 64))
        
        # Normalize the pixel values to [0, 1]
        roi_gray = roi_gray.astype("float") / 255.0
        
        # Expand the dimensions to match the model's expected input shape
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Add batch dimension
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension

        # Make prediction
        prediction = model.predict(roi_gray, verbose=0)
        
        # Get the emotion label with the highest prediction score
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]

        # Draw rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with emotion detection
    cv2.imshow("Emotion Detector", frame)

    # Exit if 'q' is pressed or the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

