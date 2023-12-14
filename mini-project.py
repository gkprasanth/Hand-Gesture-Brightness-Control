import numpy as np
import mediapipe as mp
import cv2
import screen_brightness_control as sbc
import os
import shutil
import schedule
import time
from cloudinary.uploader import upload
import cloudinary

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

# Initialize Cloudinary
cloudinary.config(
    cloud_name="dv4lqj6mi",
    api_key="136933691664477",
    api_secret="ZoOxEWPdfXEHTTNGGnm7v3zGUoA"
)

# Initialize Hand Tracking
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils



# File Movement and Cloud Upload
def move_files(source_directory, destination_directory):
    file_names = os.listdir(source_directory)
    for file_name in file_names:
        shutil.copy(os.path.join(source_directory, file_name), destination_directory)

def job():
    source_directory = "E:/Data/batch"
    destination_directory = "E:/Data/backup"
    move_files(source_directory, destination_directory)

    base_directory = "E:/Data/batch"
    file_extensions = [".jpg", ".png", ".pdf"]
    file_paths = [f"{base_directory}/{file_name}" for file_name in os.listdir(base_directory) if
                  file_name.endswith(tuple(file_extensions))]

    for file_path in file_paths:
        result = upload(file_path, resource_type="raw")
        file_url = result['url']
        print(f"File uploaded successfully. URL: {file_url}")

# Schedule the job every day at a specific time
schedule.every().day.at("02:25").do(job)

# Continuous scheduling loop
while True:
    schedule.run_pending()
    # time.sleep(1)
    a, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape  # Use shape[0] for height, shape[1] for width
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

        # Hand Gesture Detection and Brightness Adjustment
        if lmList:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = np.hypot(x2 - x1, y2 - y1)
            bright = np.interp(length, [15, 220], [0, 100])
            sbc.set_brightness(int(bright))

    # Display Image
    cv2.imshow('Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR for display
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Cleanup after Hand Tracking
cap.release()
cv2.destroyAllWindows()


