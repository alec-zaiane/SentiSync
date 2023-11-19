import cv2
import socket
import pickle
import struct
import threading
import RPi.GPIO as GPIO
import time

# Create a socket connection
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bindip is the IP address of the computer running the server program, get it using ipconfig
bindip = "192.168.1.100"
server_socket.bind((bindip, 5555))  # Change the IP and port as needed
server_socket.listen(0)
connection, client_address = server_socket.accept()
print("Connection from", client_address)

# Open the webcam
cap = cv2.VideoCapture("/dev/video0")  # Use 0 for the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 405)

# setup GPIO
# 17 is beep pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.LOW)


# Function to receive data from the separate computer
def receive_data():
    while True:
        data = connection.recv(1024)  # Adjust buffer size as needed
        if not data:
            break
        # Process the received data as needed
        print("Received data:", data.decode("utf-8"))
        data = data.decode("utf-8")
        if data == "beep":
            print("beep")
            GPIO.output(17, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(17, GPIO.LOW)
        elif data == "beep beep":
            print("beep beep")
            GPIO.output(17, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(17, GPIO.LOW)
            time.sleep(0.1)
            GPIO.output(17, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(17, GPIO.LOW)
        elif data == "hbd":
            BEEPER_PIN = 17

            # Define the notes for the "Happy Birthday" song
            notes = [
                ('C4', 0.5), ('C4', 0.5), ('D4', 1.0), ('C4', 1.0), ('F4', 1.0), ('E4', 2.0),
                ('C4', 0.5), ('C4', 0.5), ('D4', 1.0), ('C4', 1.0), ('G4', 1.0), ('F4', 2.0),
                ('C4', 0.5), ('C4', 0.5), ('C5', 1.0), ('A4', 1.0), ('F4', 1.0), ('E4', 1.0), ('D4', 2.0),
                ('Bb4', 0.5), ('Bb4', 0.5), ('A4', 1.0), ('F4', 1.0), ('G4', 1.0), ('F4', 2.0)
            ]

            # Define the frequencies for each note
            frequencies = {
                'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'Bb4': 466.16, 'B4': 493.88, 'C5': 523.25
            }

            pwm = GPIO.PWM(BEEPER_PIN, 100)
            for note, duration in notes:
                pwm.ChangeFrequency(frequencies[note])
                pwm.start(50)
                time.sleep(duration)
                pwm.stop()
                time.sleep(0.05)

# Start a separate thread for receiving data
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        #frame = "Test hello :) yippee"
        result, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if result:

            # Serialize the frame
            data = pickle.dumps(frame)
            #print(data)
            #print("=====")
            # Pack the serialized frame and send it
            size = struct.pack("=L", len(data))
            #print(type(size))
            #print(size)
            #print(struct.unpack("=L", size))
            connection.sendall(size + data)

finally:
    cap.release()
    connection.close()
    server_socket.close()

