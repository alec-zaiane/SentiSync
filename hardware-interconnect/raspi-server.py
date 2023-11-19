import cv2
import socket
import pickle
import struct
import threading
import RPi.GPIO as GPIO

# Create a socket connection
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bindip is the IP address of the computer running the server program, get it using ipconfig
bindip = "192.168.1.100"
server_socket.bind((bindip, 5555))  # Change the IP and port as needed
server_socket.listen(0)
connection, client_address = server_socket.accept()
print("Connection from", client_address)

# Open the camera
cap = cv2.VideoCapture("/dev/video0")  # Use 0 for the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 405)
print("camera opened")



# setup GPIO
# 17 is beep pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.LOW)
print("GPIO setup")


# Function to receive data from the separate computer, use this to receive commands for haptic feedback
def receive_data():
    while True:
        data = connection.recv(1024)  # Adjust buffer size as needed
        if not data:
            break
        # Process the received data as needed
        print("Received data:", data.decode("utf-8"))

# Start a separate thread for receiving data
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()
print("receive thread started")

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        #frame = "Test hello :) yippee"
        result, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if result:

            # Serialize the frame
            data = pickle.dumps(frame)
            # Pack the serialized frame and send it
            size = struct.pack("=L", len(data))
            connection.sendall(size + data)

finally:
    cap.release()
    connection.close()
    server_socket.close()

