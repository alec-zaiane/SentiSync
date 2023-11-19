from tensorflow.keras import models
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import dlib
import socket, struct, pickle

USE_WEBCAM = True # true to use webcam, false to use raspi server

if not USE_WEBCAM:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('10.42.0.1', 5555))
    payload_size = struct.calcsize("=L")

    data = b''

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Instantiate the EmotionCNN model
emotion_model = models.load_model('C:/Users/ksekh/OneDrive/Desktop/VGG Testing/emotion-detection-in-real-time/datasets/trained_vggface.h5', compile=False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}

# Open a video capture object (0 represents the default camera, or you can specify a file path)
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables for skipping frames
# skip_frames = 10
# frame_count = 0

while True:
    # Read every 'skip_frames' frames
    # frame_count += 1
    # if frame_count % skip_frames != 0:
    #     continue

    if USE_WEBCAM:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Could not read frame.")
            break

    else:
    # Receive the size of the serialized frame
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        packed_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("=L", packed_size)[0]

        # Receive the rest of serialized frame
        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame_data = pickle.loads(frame_data)
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)


        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert the frame to gqrayscale (dlib works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate through detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Extract the face region from the frame
        face_img = frame[y:y+h, x:x+w]

        # Check if the face image is not empty
        if not face_img.size == 0:
            # Resize pixels to the model size
            cropped_img = cv2.resize(face_img, (96, 96))
            cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
            cropped_img_float = cropped_img_expanded.astype(float)

            # Model prediction
            prediction = emotion_model.predict(cropped_img_float, verbose = None)
            maxindex = int(np.argmax(prediction))

            # Display the emotion label
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            
            # Draw rectangles around the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        else:
            print("Empty face image detected.")

    # Display the output
    cv2.imshow('Real-Time Emotion Classification with dlib Face Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
