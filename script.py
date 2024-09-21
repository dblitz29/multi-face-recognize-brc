import logging
import cv2
from ultralytics import YOLO
from deepface import DeepFace

logging.getLogger('ultralytics').setLevel(logging.WARNING)
model = YOLO('yolov8n.pt')

video_path = input("Masukkan Direktori Video: ")
reference_img1_path = input("Masukkan Direktori Foto Wajah 1: ")
reference_img2_path = input("Masukkan Direktori Foto Wajah 2:")

def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

skip_frames = int(fps)

if not cap.isOpened():
    print("Error: Video tidak bisa dibuka.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Video selesai.")
        break

    if frame_count % skip_frames == 0:
        results = model(frame)

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                confidence = box.conf[0].cpu().numpy()
                class_id = box.cls[0].cpu().numpy()

                if class_id == 0 and confidence > 0.5:
                    face_img = frame[int(y1):int(y2), int(x1):int(x2)]                    
                    face_img = cv2.resize(face_img, (224, 224))
                    
                    detected_face_path = 'detected_face.jpg'
                    cv2.imwrite(detected_face_path, face_img)
                    
                    result1 = DeepFace.verify(img1_path=reference_img1_path, img2_path=detected_face_path, enforce_detection=False)
                    
                    label = None
                    if result1['verified']:
                        label = "Wajah 1"
                    if reference_img2_path:
                        result2 = DeepFace.verify(img1_path=reference_img2_path, img2_path=detected_face_path, enforce_detection=False)
                        if result2['verified']:
                            label = "Wajah 2"
                    
                    if label:
                        timestamp_seconds = frame_count / fps
                        timestamp_hms = seconds_to_hms(timestamp_seconds)
                        print(f"{label} ditemukan pada waktu: {timestamp_hms}")
                        
                        cv2.putText(frame, f"{label} pada: {timestamp_hms}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Wajah Cocok Ditemukan", frame)
    cv2.imshow("Video Processing", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Pengguna keluar dari video.")
        break

cap.release()
cv2.destroyAllWindows()

print("Program selesai.")