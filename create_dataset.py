# import os
# import pickle
# import mediapipe as mp
# import cv2
# import matplotlib.pyplot as plt
#
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#
# DATA_DIR = './data'
#
# data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#
#         x_ = []
#         y_ = []
#
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#
#                     x_.append(x)
#                     y_.append(y)
#
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))
#
#             data.append(data_aux)
#             labels.append(dir_)
#
# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Set image dimensions
img_height, img_width = 224, 224  # Kích thước ảnh đầu vào cho CNN
channels = 3  # Số kênh màu (RGB)

# Create empty lists to store image tensors and labels
x_data = []
y_data = []

# Loop through the data and preprocess it
for i, landmark_data in enumerate(data):
    # Convert the landmark data into an image
    img = np.zeros((img_height, img_width, channels), dtype=np.uint8)
    for j in range(0, len(landmark_data), 2):
        x = int(landmark_data[j] * img_width)
        y = int(landmark_data[j+1] * img_height)
        cv2.circle(img, (x, y), 2, (255, 255, 255), -1)

    # Append the image tensor and its label to the lists
    x_data.append(img)
    y_data.append(labels[i])

# Convert the lists to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

f = open('data.pickle', 'wb')
pickle.dump({'data': x_data, 'labels': y_data}, f)
f.close()