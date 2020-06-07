FROM tensorflow/tensorflow:2.2.0-gpu
RUN apt-get update \
    && apt-get install wget curl zip git vim -y

RUN apt-get update && \
    apt-get install -y libfluidsynth1 \
    fluid-soundfont-gm build-essential libasound2-dev \
    libjack-dev ffmpeg sox
RUN python3 -m pip install --upgrade pip
RUN pip3 install pyfluidsynth pretty_midi
RUN pip3 install -qU magenta
RUN apt autoremove -y
RUN apt-get install --fix-broken
RUN pip3 install librosa

WORKDIR /src

RUN mkdir -p /src/onsets-frames/maestro
RUN wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip
RUN unzip maestro_checkpoint.zip -d /src/onsets-frames/maestro
RUN rm maestro_checkpoint.zip

ADD https://api.github.com/repos/thekevinscott/onsets-and-frames-transcription/git/refs/heads/master /version.json
RUN git clone https://github.com/thekevinscott/onsets-and-frames-transcription.git /src/transcription
RUN rm /version.json
