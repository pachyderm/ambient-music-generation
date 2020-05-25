####################################
#
# Music Transformer
#
####################################
FROM tensorflow/tensorflow:2.2.0-gpu-jupyter
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
  libgirepository1.0-dev gcc libcairo2-dev \
  pkg-config gir1.2-gtk-3.0 \
  git curl wget vim \
  libjpeg-dev libgif-dev libpango1.0-dev libssl-dev
RUN mkdir -p /src
# RUN git clone https://github.com/jason9693/midi-neural-processor.git \
#   && mv midi-neural-processor /src/midi_processor
COPY MusicTransformer-tensorflow2.0/requirements.txt /src/requirements.txt
RUN pip3 install -r /src/requirements.txt --ignore-installed
COPY MusicTransformer-tensorflow2.0 /src
WORKDIR /src
CMD ["bash", "-c", "/src/musictransformer.sh"]
