import cv2
import socket
import pickle
import struct
import numpy as np
import threading

# Create a socket connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.42.0.1', 5555))
payload_size = struct.calcsize("=L")
data = b""

# Function to send data to the Raspberry Pi
def send_data():
    while True:
        message = input("Enter a message to send: ")
        client_socket.sendall(message.encode("utf-8"))

# Start a separate thread for sending data
send_thread = threading.Thread(target=send_data)
send_thread.start()

try:
    ctr = 0
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
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)
        
        #print(frame)
        cv2.imshow('Received Video', frame)
        cv2.waitKey(1)

finally:
    client_socket.close()
