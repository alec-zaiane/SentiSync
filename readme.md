## instructions for running:

connect a raspberry pi, client PC and server PC to the same network, connect a vibration motor to GPIO17 on the raspberry pi.

make sure all IP addresses are correct in:
- raspi-server.py (bindip is its own IP address)
- user_server_host.py (local_ip is the client PC IP address, raspi_ip is the raspi's IP)
- emotion_webcam_demo.py (connection_ip is the client PC's IP address)


Now, run the python files in order
1. raspi-server.py on the raspberry pi
2. user_server_host.py on the client PC
3. emotion_webcam_demo.py on the server PC

finally, open a google meets meeting on the client PC and bring the python window to the foreground.

