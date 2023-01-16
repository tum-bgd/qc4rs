FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN rm /etc/apt/sources.list.d/cuda.list  
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
COPY test_requirements.txt /tmp/test_requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/test_requirements.txt