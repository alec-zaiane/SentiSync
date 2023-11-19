import socket
import threading
import RPi.GPIO as GPIO
import time

# Create a socket connection
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bindip is the IP address of the computer running the server program, get it using ipconfig
bindip = "192.168.1.100"
server_socket.bind((bindip, 5555))  # Change the IP and port as needed
server_socket.listen(0)
print(f"Waiting for connection on {bindip}:5555...")
connection, client_address = server_socket.accept()
print("Connection from", client_address)



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
        data = data.decode("utf-8")
        if data == "beep":
            GPIO.output(17, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(17, GPIO.LOW)
        elif data == "quit":
            break

# Start a separate thread for receiving data
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()
print("receive thread started")

try:
    while True:
        pass

finally:
    connection.close()
    server_socket.close()

