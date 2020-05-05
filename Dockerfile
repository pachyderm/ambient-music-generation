####################################
#
# Maestro
#
####################################
FROM alpine as maestro
# Get Magenta pre-trained model
RUN apk update && apk add wget zip \
    && mkdir /data \
    && wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip \
    && unzip maestro_checkpoint.zip -d /data/maestro \
    && rm maestro_checkpoint.zip

####################################
#
# Music Transformer
#
####################################
FROM ubuntu:latest as music_transformer
RUN apt-get update && apt-get install ca-certificates curl apt-transport-https ca-certificates gnupg -y && \
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
  apt-get update && apt-get install google-cloud-sdk -y
RUN mkdir -p /data/transformer && \
  gsutil -q -m cp -r gs://magentadata/models/music_transformer/* /data/transformer && \
  gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /data/transformer

####################################
#
# Jupyter
#
####################################
FROM ubuntu:latest

######### Magenta Data #########
# Copy Maestro
COPY --from=maestro /data /data/maestro
# Copy Transformer
COPY --from=music_transformer /data /data

######### Install Python 3 #########
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && python3 -m pip install --upgrade pip \
  && pip3 install --upgrade pip

######### Install System Packages #########
RUN apt-get update && apt-get install wget curl zip libcairo2-dev -y

######### Install Google Cloud SDK #########
RUN apt-get update && apt-get install ca-certificates curl apt-transport-https ca-certificates gnupg -y && \
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
  apt-get update && apt-get install google-cloud-sdk -y

######### Install Magenta #########
RUN pip3 install tensorflow==1.15.2
RUN apt-get update -qq && apt-get install -qq libfluidsynth1 fluid-soundfont-gm build-essential libasound2-dev libjack-dev ffmpeg
RUN pip3 install pyfluidsynth pretty_midi && pip3 install -qU magenta
RUN apt autoremove -y && apt-get install --fix-broken
RUN pip3 install librosa

######### Install Jupyter and Notebook requirements #########
RUN pip3 install google-colab
RUN pip3 install jupyter
# https://github.com/jupyter/jupyter_console/issues/163#issuecomment-418392676
RUN pip3 install --upgrade ipykernel
RUN pip3 install tensor2tensor prompt-toolkit
# Magenta runs an old version of bokeh
RUN pip uninstall bokeh -y
RUN pip3 install bokeh==1.4.0

######### Set up file system #########
RUN mkdir /notebooks && chmod a+rwx /notebooks
RUN mkdir /logs && chmod 777 /logs
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
EXPOSE 8888
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

######### Run Jupyter notebook #########
CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --allow-root --no-browser"]
