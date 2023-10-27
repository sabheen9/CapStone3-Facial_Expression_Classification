import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

# Configuration for GPU memory usage
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)

# Load the facial expression model
class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

# Create a Streamlit web app
st.title("Facial Expression Recognition with Streamlit")

# Load the pre-trained model
model = FacialExpressionModel("Emotion-model.json", "FacialExpression_weights.hdf5")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facec.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        roi = cv2.resize(face, (48, 48))
        roi = np.array(roi, dtype="float32")
        roi = np.expand_dims(roi, axis=0)
        emotion = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        
        # Display the detected emotion
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the video stream in Streamlit
    st.image(frame, channels="BGR", use_column_width=True)

# Release the webcam and close the Streamlit app on user exit
cap.release()
st.stop()


