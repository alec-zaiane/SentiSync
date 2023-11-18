import dlib
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from CNN import EmotionCNN

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Instantiate the EmotionCNN model
emotion_model = EmotionCNN()

# Load the trained parameters
emotion_model.load_state_dict(torch.load('emotion_cnn_model.pth', map_location=torch.device('cpu')))

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
            pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            # Continue with the rest of the code
        else:
            print("Empty face image detected.")

        # Preprocess the face for emotion recognition model
        input_data = preprocess_face(pil_image)

        # Perform inference with the emotion recognition model
        with torch.no_grad():
            output = emotion_model(input_data)
            predicted_emotion = torch.argmax(output, dim=1).item()

        # Define the emotions (you might need to adjust these based on your model's output)
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
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