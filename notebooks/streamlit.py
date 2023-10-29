import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
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
        st.header("POTATO RIGHT")
        st.title("Potato Leaf Disease Early Prediction")
        st.subheader("Early detection of diseases present in the leaf. This helps an user to easily detect the disease and identify it's cause.")