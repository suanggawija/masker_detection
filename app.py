from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load your trained YOLO v8 model
model = YOLO('best.pt')

def gen_frames():
    camera = cv2.VideoCapture(0)  # 0 means the default webcam
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Convert the frame to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform mask detection using YOLO
            results = model(img_rgb)

            # Draw the bounding boxes on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert the frame back to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
