import streamlit as st

import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageOps

# Load the pre-trained model
model_json_file = "Emotion-model.json"
model_weights_file = "FacialExpression_weights.hdf5"

with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
model.load_weights(model_weights_file)

# Create a Streamlit web app
st.title("Facial Expression Recognition with Streamlit")

st.write(""" 
# Facial Expression Recognition
""")

file = st.file_uploader("Upload an image", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (48, 48)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)

    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    detected_emotion = emotion_labels[np.argmax(predictions)]
    st.write("Detected Emotion:", detected_emotion)

st.sidebar.image('mg.png', use_column_width=True)
st.sidebar.title("Mangifera Healthika")
st.sidebar.subheader("Accurate detection of emotions in facial expressions.")