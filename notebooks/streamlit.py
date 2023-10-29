import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the pre-trained model
model_json_file = "Emotion-model.json"
model_weights_file = "FacialExpression_weights.hdf5"

with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the list of emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Open a connection to the webcam (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize the face region to match the input size of your model
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize the pixel values to be between 0 and 1
        face_roi = face_roi / 255.0
        
        # Reshape the face region to match the model's input shape
        face_roi = face_roi.reshape(1, 48, 48, 1)
        
        # Use the model to predict the emotion
        emotion_probabilities = model.predict(face_roi)
        
        # Get the index of the predicted emotion
        predicted_emotion_index = np.argmax(emotion_probabilities)
        
        # Get the label of the predicted emotion
        predicted_emotion_label = emotion_labels[predicted_emotion_index]
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Display the predicted emotion label near the face
        cv2.putText(frame, predicted_emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame with detected emotions
    cv2.imshow('Facial Expression Recognition', frame)
    
    # Check for the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
