import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Hand Gesture Detection", layout="centered")

st.title("🤚 Real-Time Hand Gesture Detection")

run = st.checkbox("Start Camera")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Function to detect basic gesture
def detect_gesture(hand_landmarks):
    tips = [8, 12, 16, 20]   # finger tips
    fingers = []

    # Thumb (simple check)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = fingers.count(1)

    if total == 5:
        return "Open Palm ✋"
    elif total == 0:
        return "Fist ✊"
    elif total == 2:
        return "Victory ✌️"
    elif total == 1:
        return "Pointing ☝️"
    else:
        return "Unknown Gesture"

frame_placeholder = st.empty()

cap = None

if run:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while run:
            success, frame = cap.read()
            if not success:
                st.error("Camera not detected")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            gesture_text = "No Hand Detected"

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        handLms,
                        mp_hands.HAND_CONNECTIONS
                    )

                    gesture_text = detect_gesture(handLms)

            cv2.putText(
                frame,
                gesture_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            frame_placeholder.image(frame, channels="BGR")

    if cap:
        cap.release()
