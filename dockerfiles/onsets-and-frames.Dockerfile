FROM ubuntu:18.04
RUN apt-get clean \
  && apt-get update \
  && apt-get install -y \
     python3-pip python3-dev wget zip git \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN pip3 install tensorflow==1.15.2
RUN apt-get update -qq && apt-get install -qq libfluidsynth1 \
    fluid-soundfont-gm build-essential libasound2-dev libjack-dev ffmpeg sox
RUN pip3 install pyfluidsynth pretty_midi
RUN pip3 install -qU magenta==1.3.1
RUN apt-get autoremove -y && apt-get install --fix-broken
RUN pip3 install librosa

RUN mkdir -p /data
RUN wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip
RUN unzip maestro_checkpoint.zip -d /data/maestro
RUN rm maestro_checkpoint.zip

ADD https://api.github.com/repos/thekevinscott/onsets-and-frames-transcription/git/refs/heads/master version.json
RUN git clone https://github.com/thekevinscott/onsets-and-frames-transcription.git /src
RUN rm version.json
