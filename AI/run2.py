import cv2
import dlib
from fer import FER

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Initialize the FER model for emotion recognition
emotion_detector = FER()

# Open a video capture object (0 represents the default camera, or you can specify a file path)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale (dlib works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate through detected faces and perform emotion recognition
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Extract the face region for emotion recognition
        face_region = frame[y:y+h, x:x+w]

        # Perform emotion recognition
        emotions = emotion_detector.detect_emotions(face_region)

        # Display the detected emotions
        if emotions:
            emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
            cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw rectangles around the detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the output
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()