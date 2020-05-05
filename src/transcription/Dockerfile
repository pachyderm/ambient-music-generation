FROM ubuntu:latest
RUN apt-get clean \
  && apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN apt-get install wget curl zip -y

RUN pip3 install tensorflow==1.15.2
RUN apt-get update -qq && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev ffmpeg 
RUN pip3 install pyfluidsynth pretty_midi
RUN pip3 install -qU magenta
RUN apt autoremove -y
RUN apt-get install --fix-broken
RUN pip3 install librosa

######### Jupyter #########
RUN mkdir /notebooks && chmod a+rwx /notebooks

RUN mkdir -p /notebooks/onsets-frames
RUN mkdir /notebooks/onsets-frames/maestro
RUN wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip
RUN unzip maestro_checkpoint.zip -d /notebooks/onsets-frames/maestro
RUN rm maestro_checkpoint.zip


RUN mkdir /logs && chmod 777 /logs
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
EXPOSE 8888

COPY src /notebooks/src

CMD ["python3", "/notebooks/src/transcribe.py"]


