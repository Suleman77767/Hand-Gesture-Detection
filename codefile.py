import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Hand Gesture Detection")

st.title("🤚 Hand Gesture Detection")
st.write("If you see this page, all libraries are installed correctly.")

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

st.success("OpenCV and MediaPipe loaded successfully!")

# Create a blank image
image = np.zeros((400, 600, 3), dtype=np.uint8)

# Add text on image
cv2.putText(
    image,
    "Hand Gesture App Ready",
    (50, 200),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

st.image(image, channels="BGR")
