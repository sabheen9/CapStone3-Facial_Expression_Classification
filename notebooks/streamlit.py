import streamlit as st

cam=st.selectbox(
  'Choose a Cam', 
  [
    '',
    'pitriverbridge/pitriverbridge.jpg',
    'johnsongrade/johnsongrade.jpg',
    'perez/perez.jpg',
    'mthebron/mthebron.jpg',
    'eurekaway/eurekaway.jpg',
    'sr70us395/sr70us395.jpg',
    'bogard/bogard.jpg',
    'eastriverside/eastriverside.jpg',
  ]
)
if cam:
  st.image('https://cwwp2.dot.ca.gov/data/d2/cctv/image/' + cam)

