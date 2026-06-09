import streamlit as st
import mediapipe as mp

st.write("MediaPipe Version:", mp.__version__)

try:
    st.write("Solutions Found:", hasattr(mp, "solutions"))
except Exception as e:
    st.error(str(e))
