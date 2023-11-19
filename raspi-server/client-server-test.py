import cv2
import socket
import pickle
import struct
import numpy as np
import threading
import queue

# Create a socket connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.100', 5555))
payload_size = struct.calcsize("=L")

frame_queue = queue.Queue()

# Function to send data to the Raspberry Pi
def send_data():
    while True:
        message = input("Enter a message to send: ")
        client_socket.sendall(message.encode("utf-8"))

# Start a separate thread for sending data
send_thread = threading.Thread(target=send_data)
send_thread.start()

def receive_frame():
    data = b""
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
        output_frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)
        frame_queue.put(output_frame)

# Start a separate thread for receiving data
receive_thread = threading.Thread(target=receive_frame)
receive_thread.start()

try:
    ctr = 0
    while True:
        # get the next frame from the queue (skip a few frames if needed)
        while frame_queue.qsize() > 1:
            frame_queue.get()
        frame = frame_queue.get()
        
        cv2.imshow('Received Video', frame)
        cv2.waitKey(1)

finally:
    client_socket.close()
