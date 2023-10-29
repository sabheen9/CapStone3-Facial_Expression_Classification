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
        st.header("Facial Recognition App")
        st.title("Facial Expression Recognition")
        st.subheader("Facial expression recognition enables more natural and intuitive interactions between humans and computer systems, enhancing user experience and engagement.")