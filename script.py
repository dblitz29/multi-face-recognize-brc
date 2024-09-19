import cv2
import numpy as np
import tensorflow as tf
import torch
from sklearn.metrics.pairwise import cosine_similarity

model = torch.hub.load('ultralytics/yolov8', 'yolov8n')

efficientnet_b0 = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

def extract_face_features(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = tf.keras.applications.efficientnet.preprocess_input(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    features = efficientnet_b0.predict(face_img)
    return features

def compare_faces(reference_features, detected_features, threshold=0.7):
    similarity = cosine_similarity(reference_features, detected_features)
    return similarity[0][0] > threshold

reference_img = cv2.imread('foto.png')
reference_features = extract_face_features(reference_img)

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy() 
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if class_id == 0:
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]
            detected_features = extract_face_features(face_img)
            
            if compare_faces(reference_features, detected_features):
                timestamp = frame_count / fps
                print(f"Matching face found at: {int(timestamp // 60)}:{int(timestamp % 60)}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
