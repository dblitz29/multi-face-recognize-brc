FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install flask torch torchvision torchaudio deepface opencv-python-headless ultralytics tf-keras
COPY . /app
EXPOSE 5000
CMD ["python", "app.py"]
