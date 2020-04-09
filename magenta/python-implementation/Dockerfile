FROM alpine as model

# Get Magenta pre-trained model
RUN apk update && apk add wget zip \
    && mkdir /models \
    && cd /models \
    && wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip \
    && unzip maestro_checkpoint.zip \
    && rm maestro_checkpoint.zip

# FROM ubuntu:latest as magenta

# RUN apt-get update \
#   && apt-get install -y git

# RUN git clone https://github.com/tensorflow/magenta.git

FROM ubuntu:latest
# COPY --from=magenta /magenta /code/magenta

WORKDIR /code

# Copy Magenta pre-trained model
COPY --from=model /models /models

# Install sound libraries and system files
RUN apt-get update \
  && apt-get -y install build-essential libasound2-dev libjack-dev portaudio19-dev sndfile-tools sox

# Install python 3 and magenta
RUN apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && pip3 install magenta

RUN mkdir /checkpoints
