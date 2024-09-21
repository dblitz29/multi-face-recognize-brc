import os
import cv2
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import base64
import io
from PIL import Image
from deepface import DeepFace

app = Flask(__name__)

model = YOLO('yolov8n.pt')

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

video_path = None
img1_path = None
img2_path = None
names = {"img1": "Face 1", "img2": "Face 2"}
interval = 1
interval_unit = "seconds"

# Fungsi Return Waktu
def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle video and image uploads along with face names
@app.route('/upload', methods=['POST'])
def upload_files():
    global video_path, img1_path, img2_path, names, interval, interval_unit
    
    if 'video' not in request.files or 'img1' not in request.files:
        return "Please upload both video and reference image 1.", 400
    
    # Save video and reference images
    video_file = request.files['video']
    img1_file = request.files['img1']
    img2_file = request.files.get('img2')  # Optional Wajah 2
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.png')
    
    video_file.save(video_path)
    img1_file.save(img1_path)

    if img2_file:
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.png')
        img2_file.save(img2_path)
    else:
        img2_path = None

    # Nama wajah 
    names["img1"] = request.form.get('name1', 'Face 1')
    names["img2"] = request.form.get('name2', 'Face 2')

    # Hitungan interval unit untuk fps
    interval = int(request.form.get('interval', 1))
    interval_unit = request.form.get('interval_unit', 'seconds')
    return "Files and names uploaded successfully!"

############# Endpoint Proses Gambar ################
@app.route('/process', methods=['GET'])
def process_video_stream():
    def generate():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Ambil fps per secomd
        # Masukin jenis frame per (detik/menit)
        if interval_unit == "minutes":
            skip_frames = int(fps * 60 * interval)
        else:
            skip_frames = int(fps * interval)  
        frame_count = 0
        with open(img1_path, "rb") as img1_file:
            base64_img1 = base64.b64encode(img1_file.read()).decode('utf-8')
        base64_img2 = None
        if img2_path and os.path.exists(img2_path):
            with open(img2_path, "rb") as img2_file:
                base64_img2 = base64.b64encode(img2_file.read()).decode('utf-8')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                yield "data: Processing finished\n\n"
                break
            if frame_count % skip_frames == 0:
                results = model(frame)
                for detection in results:
                    boxes = detection.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        class_id = int(box.cls[0])
                        if class_id == 0:
                            face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                            face_img_resized = cv2.resize(face_img, (400, 400))
                            detected_face_path = os.path.join(UPLOAD_FOLDER, 'detected_face.jpg')
                            cv2.imwrite(detected_face_path, face_img_resized)
                            with open(detected_face_path, "rb") as detected_face_file:
                                base64_detected_face = base64.b64encode(detected_face_file.read()).decode('utf-8')
                            result1 = DeepFace.verify(img1_path=img1_path, img2_path=detected_face_path, enforce_detection=False)
                            name = None
                            ref_image_base64 = base64_img1
                            if result1['verified']:
                                name = names["img1"]
                            if img2_path and os.path.exists(img2_path):
                                result2 = DeepFace.verify(img1_path=img2_path, img2_path=detected_face_path, enforce_detection=False)
                                if result2['verified']:
                                    name = names["img2"]
                                    ref_image_base64 = base64_img2
                            if name:
                                timestamp_seconds = frame_count / fps
                                timestamp_hms = seconds_to_hms(timestamp_seconds)
                                yield f"data: {timestamp_hms}||{name}||{ref_image_base64}||{base64_detected_face}\n\n"

            frame_count += 1

        cap.release()

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True)
