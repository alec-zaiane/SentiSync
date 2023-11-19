import dlib
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from CNN import EmotionCNN, DeeperEmotionCNN
import socket, struct, pickle
import numpy as np


# Load the pre-trained face detection model from dlib
USE_WEBCAM = True # true to use webcam, false to use raspi server

if not USE_WEBCAM:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.137.85', 5555))
    payload_size = struct.calcsize("=L")

    data = b''


detector = dlib.get_frontal_face_detector()

# Instantiate the EmotionCNN model
emotion_model = DeeperEmotionCNN()

# Load the trained parameters
emotion_model.load_state_dict(torch.load('models/emotion_cnn_model11.pth', map_location=torch.device('cpu')))

# Define a function to preprocess the face for the emotion recognition model
def preprocess_face(face_img):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(face_img).unsqueeze(0)

# Open a video capture object (0 represents the default camera, or you can specify a file path)


if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
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

    # Convert the frame to grayscale (dlib works with grayscale images)
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
            pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            # Continue with the rest of the code
        else:
            print("Empty face image detected.")

        # Preprocess the face for emotion recognition model
        input_data = preprocess_face(pil_image)

        # Perform inference with the emotion recognition model
        with torch.no_grad():
            emotion_model.eval()
            output = emotion_model(input_data)
            predicted_emotion = torch.argmax(output, dim=1).item()

        # Define the emotions (you might need to adjust these based on your model's output)
        # emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        emotions = ["Happy", "Neutral", "Sad"]
        emotion_text = emotions[predicted_emotion]

        # Draw rectangles around the detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted emotion text
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Real-Time Face Detection with Emotion Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()