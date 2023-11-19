from tensorflow.keras import models
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import dlib
import socket, struct, pickle
import queue
import threading
import math



USE_WEBCAM = False # true to use webcam, false to use raspi server
connection_ip = "192.168.1.99"
video_queue = queue.Queue()


if not USE_WEBCAM:
    def receive_video():
        global connection_ip
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((connection_ip, 5555))
        payload_size = struct.calcsize("=L")

        data = b''

        while True:
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
            frame_array_0 = np.frombuffer(frame_data[0], dtype=np.uint8)
            frame_array_1 = np.frombuffer(frame_data[1], dtype=np.uint8)
            output_frame_0 = cv2.imdecode(frame_array_0, flags=cv2.IMREAD_COLOR)
            output_frame_1 = cv2.imdecode(frame_array_1, flags=cv2.IMREAD_COLOR)
            video_queue.put((output_frame_0, output_frame_1))
    
    def find_best_face(faces, frame):
        if len(faces) == 0:
            return (0,0,0,0)
        if len(faces) == 1: #shortcut
            return faces[0]
        frame_center = (frame.shape[0]/2, frame.shape[1]/2)
        best_found_face = faces[0]
        x,y,w,h = faces[0]
        best_found_ctrpoint = (x+w/2, y+h/2)
        best_found_dist = math.dist(best_found_ctrpoint, frame_center)
        for (x,y,w,h) in faces[1:]:
            current_ctr = (x+w/2, y+h/2)
            current_dist = math.dist(current_ctr, frame_center)
            if current_dist < best_found_dist:
                best_found_face = (x,y,w,h)
                best_found_ctrpoint = current_ctr
                best_found_dist = current_dist
        return best_found_face
    
    send_data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_data_socket.connect((connection_ip, 5557))
    def send_data(speaker_em, user_em):
        global send_data_socket
        data = (speaker_em, user_em)
        s = pickle.dumps(data)
        send_data_socket.sendall(struct.pack("=L", len(s)) + s)

    ewma_user = None
    ewma_speaker = None
    alpha = 0.3

    

# Load the pre-trained face detection model from dlib
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("loaded face detector")

# Instantiate the EmotionCNN model
emotion_model = models.load_model('C:/Users/ksekh/OneDrive/Desktop/VGG Testing/emotion-detection-in-real-time/datasets/trained_vggface.h5', compile=False)
print("model loaded")
# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}

# Open a video capture object (0 represents the default camera, or you can specify a file path)
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    video_thread = threading.Thread(target=receive_video)
    video_thread.start()
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
        )
        for (x, y, w, h) in face:
            face_img = frame[y:y+h, x:x+w]
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

    else: # we are getting video from the network
        while video_queue.qsize() > 1:
             #throw out any older frames we haven't had time to process
            video_queue.get()
        frames = video_queue.get()
        speakerframe, userframe = frames # speaker is the person being spoken to, user is the one hosting the call
        speakerfaces = detector.detectMultiScale(
            speakerframe, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
        )
        userfaces = detector.detectMultiScale(
            userframe, scaleFactor=1.1, minNeighbors=5, minSize=(20,20)
        )
        # find the closest faces to the midpoint of their images
        speakerface = find_best_face(speakerfaces, speakerframe)
        userface = find_best_face(userfaces, userframe)

        emotions = ['','']
        for i,((x,y,w,h), img) in enumerate(((speakerface, speakerframe), (userface, userframe))):
            face_img = img[y:y+h, x:x+w]
            if not face_img.size==0:
                resized = cv2.resize(face_img, (96,96))
                resized = np.expand_dims(resized, axis=0)
                resized = resized.astype(float)
                prediction = emotion_model.predict(resized, verbose=None)
                if i == 0:
                    if ewma_speaker is None:
                        ewma_speaker = prediction
                    else:
                        ewma_speaker = alpha*prediction + (1-alpha)*ewma_speaker
                else:
                    if ewma_user is None:
                        ewma_user = prediction
                    else:
                        ewma_user = alpha*prediction + (1-alpha)*ewma_user
                
                maxindex = int(np.argmax((ewma_speaker, ewma_user)[i]))
                emotions[i] = emotion_dict[maxindex]

                cv2.putText(img, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                emotions[i] = 'None'
        
        send_data(emotions[0], emotions[1])
        cv2.imshow("user",userframe)
        cv2.imshow("speaker",speakerframe)


        



    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
