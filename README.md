
# 🖼️ YOLOv8 & DeepFace Flask App for Face Recognition

Welcome to the **YOLOv8 & DeepFace Flask App**! This project integrates **Flask**, **YOLOv8**, and **DeepFace** for real-time face recognition in video streams. You can run it directly using **Python** or inside a **Docker** container for a more controlled environment.

## 📁 Project Structure

```bash
.
├── app.py                    # Main Flask application
├── script.py                 # Python script to run detection without Flask
├── scriptv5.py               # Another script using YOLOv5 without Flask
├── Dockerfile                # Dockerfile for containerizing the app
├── requirements.txt          # Python dependencies (if you want to run without Docker)
├── templates/
│── index.html            # HTML template for web interface   
└── uploads/                  # Folder for video/image uploads (generated dynamically)
├── yolov5s.pt                 # YOLOv5 Model
└── yolov8n.pt                 # YOLOv8 Model
```

---

## 🚀 Features

- **YOLOv8-based** object detection
- **DeepFace** integration for face verification
- Real-time video processing and timestamping of detected faces
- Web interface for uploading videos and reference images

---

## 🔧 Installation

### Running with Python (without Docker)

You can run the application using Python if you prefer not to use Docker.

### 1. Clone the Repository

First, clone this repository to your local machine:
```bash
git https://github.com/dblitz29/multi-face-recognize-brc
cd multi-face-recognize-brc
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the Python dependencies via `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

To run the Flask app:
```bash
python app.py
```

### 5. Access the App

Open your browser and go to:
```
http://localhost:5000
```

---

### Running Python Without Flask

If you want to run the detection without Flask, you can use the following two scripts:

- **script.py**: To run face detection without Flask.
  ```bash
  python script.py
  ```

- **scriptv5.py**: To run face detection using YOLOv5.
  ```bash
  python scriptv5.py
  ```

---

## 🐳 Running with Docker

If you prefer to run the application inside Docker, follow these steps.

### 1. Build the Docker Image

```bash
docker build -t flask-yolo-app .
```

### 2. Run the Docker Container

Run the container and expose port 5000:
```bash
docker run -p 5000:5000 flask-yolo-app
```

### 3. Access the App

Open your browser and go to:
```
http://localhost:5000
```

---

## 🛠️ How to Use the App

### 1. Upload Video and Reference Images

- Upload a video (in `.mp4` format) and two reference face images (e.g., `Face 1`, `Face 2`) via the web interface.
- Optionally, you can assign a **name** to each reference face (defaults to **Face 1** and **Face 2**).

### 2. Real-Time Face Detection

- Once uploaded, the app processes the video, detecting faces every second (or as configured).
- Matched faces are shown in real-time, along with the **timestamp** and **reference image**.

### 3. Results

- The web interface will display a scrollable list of detected faces with timestamps and names.
- Optionally, the application can save the results and detected faces in the `uploads/` folder.

---

## 📝 Example Usage

1. **Upload Video**: Select a video file (`video.mp4`).
2. **Upload Reference Images**: Upload face images (`img1.png` and `img2.png`).
3. **Results**: The app detects and compares faces with reference images. Matches are displayed with the timestamp on the web interface.

---

## 🐍 Python Dependencies

In case you're running without Docker, here are the key dependencies:

```txt
Flask==3.0.3
torch==2.4.1
torchvision==0.15.2
torchaudio==2.0.2
DeepFace==0.0.79
opencv-python-headless==4.7.0.72
ultralytics==8.0.89
```

Install them by running:
```bash
pip install -r requirements.txt
```

---

## 📦 Docker Setup

### Dockerfile Overview

The **Dockerfile** handles setting up the environment for the app. Here’s a brief overview:

- **Base Image**: We use `python:3.9-slim` for a lightweight yet functional base.
- **System Dependencies**: Installs essential packages like `ffmpeg` and OpenCV dependencies.
- **Python Packages**: Installs necessary libraries, including **PyTorch**, **Flask**, **DeepFace**, and **YOLOv8**.
- **Runs the App**: The Flask app runs on port 5000 by default.

---

## 🛑 Troubleshooting

1. **Face Not Detected**: Ensure the reference image is clear and contains only one face. Adjust the reference images if necessary.
2. **Docker Build Takes Too Long**: Consider using pre-built Docker images (like `pytorch/pytorch`) or improve caching in your Docker setup.
3. **Running Out of Memory**: If you're processing large videos, ensure your system has enough memory or limit the number of frames processed.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🤝 Contributions

Feel free to contribute to the project by submitting a pull request or reporting issues!

---

## 🌟 Acknowledgments

- Thanks to **Ultralytics** for YOLOv8
- **DeepFace** for easy facial recognition
- Flask for the backend framework
