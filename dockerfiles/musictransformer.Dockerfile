FROM tensorflow/tensorflow:2.2.0-gpu
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

######### Set up system with Python #########
RUN apt-get update && apt-get install -y \
  libgirepository1.0-dev gcc libcairo2-dev \
  libjpeg-dev libgif-dev libpango1.0-dev libssl-dev \
  pkg-config gir1.2-gtk-3.0 software-properties-common \
  git curl wget

WORKDIR /src

RUN git clone https://github.com/thekevinscott/MusicTransformer-tensorflow2.0.git /src
RUN git clone https://github.com/jason9693/midi-neural-processor.git /src/midi_processor
RUN pip3 install -r /src/requirements.txt --ignore-installed
