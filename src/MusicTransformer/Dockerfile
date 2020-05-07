####################################
#
# Music Transformer
#
####################################
FROM ubuntu:latest as music_transformer_data
RUN apt-get update && apt-get install ca-certificates curl apt-transport-https ca-certificates gnupg -y && \
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
  apt-get update && apt-get install google-cloud-sdk -y
RUN mkdir -p /data/transformer && \
  gsutil -q -m cp -r gs://magentadata/models/music_transformer/* /data/transformer && \
  gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /data/transformer

# ####################################
# #
# # Jupyter
# #
# ####################################
FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive

######### Magenta Data #########
# Copy Transformer
COPY --from=music_transformer_data /data /data

######### Install Python 3 and system packages#########
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
  libgirepository1.0-dev gcc libcairo2-dev \
  pkg-config gir1.2-gtk-3.0 software-properties-common \
  git curl wget \
  && add-apt-repository ppa:deadsnakes/ppa -y \
  && apt-get install python3.6 python3-pip -y

# ######### Set up Jason #########
RUN apt-get install libcairo2-dev libjpeg-dev libgif-dev libpango1.0-dev libssl-dev -y
RUN git clone https://github.com/hitheory/MusicTransformer-tensorflow2.0.git /MusicTransformer-tensorflow2.0
RUN pip3 install -r /MusicTransformer-tensorflow2.0/requirements.txt --ignore-installed

######### Install Jupyter and Notebook requirements #########
RUN pip3 install google-colab
RUN pip3 install jupyter
# https://github.com/jupyter/jupyter_console/issues/163#issuecomment-418392676
RUN pip3 install --upgrade ipykernel
RUN pip3 install tensor2tensor prompt-toolkit

######### Set up file system #########
RUN mkdir -p /notebooks && chmod a+rwx /notebooks
RUN mkdir -p /logs && chmod 777 /logs
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
EXPOSE 8888
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

######### Install jason #########
WORKDIR /MusicTransformer-tensorflow2.0
RUN git clone https://github.com/jason9693/midi-neural-processor.git \
  && mv midi-neural-processor midi_processor
RUN sh /MusicTransformer-tensorflow2.0/dataset/scripts/classic_piano_downloader.sh /data/classic_piano
RUN python3 preprocess.py /data/classic_piano /data/classic_piano_preprocessed

######### Run Jupyter notebook #########
CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --allow-root --no-browser"]
