import cv2
import numpy as np
import streamlit as st
import time

st.title("🧙 Invisibility Cloak - Streamlit Edition")
st.markdown("Put on a red cloth and watch the magic happen!")

start_button = st.button("Start Cloak Effect")
frame_placeholder = st.empty()

if start_button:
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    # Capture background
    st.info("Capturing background... Hold still for 3 seconds")
    for i in range(30):
        ret, background = cap.read()
    background = np.flip(background, axis=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.flip(frame, axis=1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red cloak range
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Create mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        cloak_mask = mask1 + mask2

        # Remove noise
        cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        cloak_mask = cv2.dilate(cloak_mask, np.ones((3, 3), np.uint8), iterations=1)

        # Invert mask
        inverse_mask = cv2.bitwise_not(cloak_mask)

        # Extract cloak & non-cloak areas
        cloak_area = cv2.bitwise_and(background, background, mask=cloak_mask)
        non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)

        # Combine
        final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

        # Convert BGR → RGB for Streamlit
        final_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(final_rgb, channels="RGB")

    cap.release()

