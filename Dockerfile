FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
