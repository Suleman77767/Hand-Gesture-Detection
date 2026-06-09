import streamlit as st
import cv2
import mediapipe as mp

st.title("Hand Gesture Detection App")

st.write("OpenCV:", cv2.__version__)
st.write("MediaPipe:", mp.__version__)
st.write("Status: OK")
