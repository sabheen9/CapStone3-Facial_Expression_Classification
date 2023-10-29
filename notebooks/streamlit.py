import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import cv2

st.set_page_config(
    page_title="Potato Disease Classification"
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

with st.sidebar:
        st.header("Facial Recognition App")
        st.title("Facial Expression Recognition")
        st.subheader("Facial expression recognition enables more natural and intuitive interactions between humans and computer systems, enhancing user experience and engagement.")



img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    st.write(cv2_img.shape)
