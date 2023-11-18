import dlib
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Load the VGG16 model pre-trained on ImageNet
vgg_model = VGG16(weights='imagenet', include_top=True)

# Define the emotions
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Open a video capture object (0 represents the default camera, or you can specify a file path)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

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
            resized_img = cv2.resize(face_img, (224, 224))  # Resize to VGG input size
            img_array = keras_image.img_to_array(resized_img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Perform inference with VGG model for emotion recognition
            predictions = vgg_model.predict(img_array)
            predicted_class = np.argmax(predictions)
            emotion_text = emotions[predicted_class]

            # Draw rectangles around the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the predicted emotion text
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print("Empty face image detected.")

    # Display the output
    cv2.imshow('Real-Time Face Detection with Emotion Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()