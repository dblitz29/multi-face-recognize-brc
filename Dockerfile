# Use a lightweight Python image based on Debian
FROM python:3.9-slim

# Set environment variable to not buffer output
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install dependencies required for OpenCV and other system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install flask torch torchvision torchaudio deepface opencv-python-headless ultralytics

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port Flask is running on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]
