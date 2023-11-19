import asyncio, socket, struct, threading
import pyautogui
import numpy as np
import cv2
import pickle
import pyaudio
from rich import print
import time

local_ip = ""
local_ip = "192.168.1.99"
print(f"[yellow]Local IP address: {local_ip}[/yellow]")

assert local_ip != "", "Please set the local IP address of the server."

kill_threads = False


def process_screen(frame):
    # process the frame here,
    # INPUT: black and white image
    # OUTPUT: cropped, resized, processed image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cropin1 = (144, 104, -142, -148) # distance from left, top, right bottom
    framecrop1 = frame[cropin1[1]:cropin1[3], cropin1[0]:cropin1[2]]
    cropin2 = (1589, 747, -40, -168)
    framecrop2 = frame[cropin2[1]:cropin2[3], cropin2[0]:cropin2[2]]
    # now rescale framecrop1
    framecrop1 = cv2.resize(framecrop1, (0,0), fx=0.7, fy=0.7)

    result1, framecrop1 = cv2.imencode(".jpg", framecrop1, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    result2, framecrop2 = cv2.imencode(".jpg", framecrop2, [int(cv2.IMWRITE_JPEG_QUALITY), 90]) # higher quality for tiny image
    return (result1 and result2), (framecrop1, framecrop2)

def serve_screen():
    global local_ip
    ft_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ft_socket.bind((local_ip, 5555))
    ft_socket.listen(0)
    print(f"Waiting for FaceTime connection... (local IP: {local_ip}, port: 5555)")
    ft_connection, ft_client_address = ft_socket.accept()
    print(f"FaceTime connection established with {ft_client_address}.")
    while not kill_threads:
        img = pyautogui.screenshot()
        frame = np.array(img)
        # process then send the frames
        result, frames = process_screen(frame)
        if result:
            # Serialize the frame
            data = pickle.dumps(frames)
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

    while not kill_threads:
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
    while not kill_threads:
        data = b''
        while len(data) < struct.calcsize("=L"):
            data = data_connection.recv(1024)  # Adjust buffer size as needed
        # Receive the rest of serialized frame
        while len(data) < struct.unpack("=L", data[:struct.calcsize("=L")])[0]:
            data += data_connection.recv(1024)
        data = data[struct.calcsize("=L"):]
        data = pickle.loads(data)
        # Process the received data as needed

        user_em, speaker_em = data[0], data[1]
        img = np.zeros((200,500))
        img = cv2.putText(img, f"user: {user_em}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        img = cv2.putText(img, f"speaker: {speaker_em}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("data", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    data_connection.close()
    
def main():
    print("Starting server...")
    time.sleep(0.1)
    facetime_svr = threading.Thread(target=serve_screen)
    facetime_svr.start()
    time.sleep(0.1)
    audio_svr = threading.Thread(target=serve_audio)
    audio_svr.start()
    time.sleep(0.1)
    data_svr = threading.Thread(target=receive_data)
    data_svr.start()
    time.sleep(0.1)
    print("[green]Server started.[/green]")
    print("Press ctrl-c to stop the server.")
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping server...")
            kill_threads = True
            break
    # wait for all threads to finish
    facetime_svr.join()
    audio_svr.join()
    data_svr.join()
        
    print("[green]Server finished.[/green]")


if __name__ == "__main__":
    main()