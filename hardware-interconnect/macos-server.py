import asyncio, socket, struct, threading
import pyautogui
import numpy as np
import cv2
import pickle
import pyaudio
from rich import print

local_ip = socket.gethostbyname(socket.gethostname())
print(f"[yellow]Local IP address: {local_ip}[/yellow]")

assert local_ip != "", "Please set the local IP address of the server."


def serve_facetime():
    global local_ip
    ft_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ft_socket.bind((local_ip, 5555))
    ft_socket.listen(0)
    print(f"Waiting for FaceTime connection... (local IP: {local_ip}, port: 5555)")
    ft_connection, ft_client_address = ft_socket.accept()
    print(f"FaceTime connection established with {ft_client_address}.")
    while True:
        # capture whole screen, TODO crop to just the FaceTime window
        img = pyautogui.screenshot()
        frame = np.array(img)
        # rescale to 720p and compress before sending it
        frame = cv2.resize(frame, (1280, 720))
        result, frame = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if result:
            # Serialize the frame
            data = pickle.dumps(frame)
            # Pack the serialized frame and send it
            size = struct.pack("=L", len(data))
            ft_connection.sendall(size + data)
        else:
            print("failed to encode frame, trying again (ctrl-c to quit)")
    ft_connection.close()


def serve_audio():
    global local_ip
    audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    audio_socket.bind((local_ip, 5556))
    audio_socket.listen(0)
    print(f"Waiting for audio connection... (local IP: {local_ip}, port: 5556)")
    audio_connection, audio_client_address = audio_socket.accept()
    print(f"Audio connection established with {audio_client_address}.")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Set audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    # Open audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK)
        # Send audio data over the connection
        audio_connection.sendall(data)

    # Close the audio stream and connection
    stream.stop_stream()
    stream.close()
    audio_connection.close()
    p.terminate()

def receive_data():
    global local_ip
    data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data_socket.bind((local_ip, 5557))
    data_socket.listen(0)
    print(f"Waiting for data connection... (local IP: {local_ip}, port: 5557)")
    data_connection, data_client_address = data_socket.accept()
    print(f"Data connection established with {data_client_address}.")
    while True:
        data = data_connection.recv(1024)  # Adjust buffer size as needed
        if not data:
            print("data connection closed")
            break
        # Process the received data as needed
        print("Received data:", data.decode("utf-8"))

    data_connection.close()
    
def main():
    print("Starting server...")
    facetime_svr = threading.Thread(target=serve_facetime)
    facetime_svr.start()
    audio_svr = threading.Thread(target=serve_audio)
    audio_svr.start()
    data_svr = threading.Thread(target=receive_data)
    data_svr.start()
    print("[green]Server started.[/green]")
    # wait for all threads to finish
    facetime_svr.join()
    audio_svr.join()
    data_svr.join()
        
    print("[green]Server finished.[/green]")


if __name__ == "__main__":
    main()