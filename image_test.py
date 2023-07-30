import cv2
import numpy as np
from keras.models import load_model

# Load the emotion recognition model
emotion_model = load_model('model.h5')

# Define the emotion labels and their corresponding colors
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 255, 255), (0, 0, 0), (255, 0, 255)]

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the test image
test_image_path = 'test.jpg'  # Replace with the actual path to the test image
test_image = cv2.imread(test_image_path)
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)  # Convert the test image to grayscale

# Perform face detection using the Haar cascade classifier
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Process each detected face
for (x, y, w, h) in faces:
    face_image = gray[y:y+h, x:x+w]  # Extract the face region from the grayscale frame

    # Preprocess the face image (e.g., resize, convert to grayscale, normalize, etc.)
    # ...
    resized_face_image = cv2.resize(face_image, (48, 48))
    normalized_face_image = resized_face_image / 255.0
    preprocessed_face_image = np.expand_dims(normalized_face_image, axis=0)

    # Perform emotion recognition using your trained model
    emotion_prediction = emotion_model.predict(preprocessed_face_image)
    emotion_index = np.argmax(emotion_prediction)
    emotion_label = emotion_labels[emotion_index]

    # Draw a rectangle around the face and label it with the predicted emotion
    cv2.rectangle(test_image, (x, y), (x+w, y+h), emotion_colors[emotion_index], 2)
    cv2.putText(test_image, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_colors[emotion_index], 2)

cv2.imshow('Face Emotion Recognition', test_image)  # Display the test image with face emotion recognition
cv2.waitKey(0)
cv2.destroyAllWindows()
