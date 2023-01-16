FROM tensorflow/tensorflow:2.4.1-gpu-jupyter
RUN rm /etc/apt/sources.list.d/cuda.list  
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN curl https://developer.download.nvidia.com/compute/cuda/repos/debian10/x86_64/cuda-keyring_1.0-1_all.deb -o cuda-keyring_1.0-1_all.deb
RUN apt-key del 7fa2af80
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt-get install ffmpeg
RUN apt-get install libsm6
RUN apt-get install libxext6  -y
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
