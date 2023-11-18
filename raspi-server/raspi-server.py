# yoinked from https://github.com/NakulLakhotia/Live-Streaming-using-OpenCV-Flask/blob/main/app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
import cv2

app = Flask(__name__)

print("opening camera")
video_device = "/dev/video0"
camera = cv2.VideoCapture(video_device) 
print("camera opened")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("camera res set")

# for local webcam use cv2.VideoCapture(0)  
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            print("failed to read frame!!!!!")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


# @app.route('/notify', methods=['POST'])
# def notify():
#     data = request.get_json()
#     notify_value = data.get('notify')

#     if notify_value is None:
#         return jsonify(error='Invalid JSON payload'), 400

#     # Match the value of "notify" using a match statement
#     match notify_value:
#         case 'x':
#             # Handle case 'x'
#             # ...
#             return jsonify(message='Matched case x')
#         case 'y':
#             # Handle case 'y'
#             # ...
#             return jsonify(message='Matched case y')
#         case 'z':
#             # Handle case 'z'
#             # ...
#             return jsonify(message='Matched case z')
#         case _:
#             return jsonify(error='Unmatched case'), 400

if __name__ == '__main__':
    app.run(debug=True)