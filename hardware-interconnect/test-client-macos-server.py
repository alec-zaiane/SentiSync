import cv2
import socket
import pickle
import struct
import numpy as np
import threading
from queue import Queue
from rich import print
import pyaudio

connection_ip = '192.168.1.99'
assert connection_ip != '', 'Please set the IP address of the server.'

video_queue = Queue()
audio_queue = Queue()

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
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        output_frame = cv2.imdecode(frame_array, flags=cv2.IMREAD_COLOR)
        video_queue.put(output_frame)

def receive_audio():
    global connection_ip
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((connection_ip, 5556))
    payload_size = struct.calcsize("=L")

    data = b''

    while True:
        data = client_socket.recv(1024)
        if not data:
            break

        audio_queue.put(data)

def send_data():
    global connection_ip
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((connection_ip, 5557))

    while True:
        message = input("Enter a message to send: ")
        client_socket.sendall(message.encode("utf-8"))

def main():
    global video_queue
    global audio_queue

    video_thread = threading.Thread(target=receive_video)
    audio_thread = threading.Thread(target=receive_audio)
    send_thread = threading.Thread(target=send_data)

    video_thread.start()
    audio_thread.start()
    send_thread.start()

    while True:
        # get the next frame from the queue (skip a few frames if needed)
        while video_queue.qsize() > 1:
            video_queue.get()
        frame = video_queue.get()
        
        cv2.imshow('Received Video', frame)
        cv2.waitKey(1)
    
        while audio_queue.qsize() > 44: # 44000 is about 1 second of audio, only keep 1 second of audio in the queue to keep memory usage low
            audio_queue.get()


if __name__ == '__main__':
    main()