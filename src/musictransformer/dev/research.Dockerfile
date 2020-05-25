####################################
#
# Jupyter
#
####################################
FROM ubuntu:latest
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

######### Magenta Data #########

RUN apt-get update && apt-get install -y \
  libgirepository1.0-dev gcc libcairo2-dev \
  pkg-config gir1.2-gtk-3.0 software-properties-common \
  git curl wget \
  && add-apt-repository ppa:deadsnakes/ppa -y \
  && apt-get install python3.6 python3-pip -y

######### NVIDIA #########
RUN apt-get install nvidia-modprobe nvidia-utils-440 -y

######### Set up Jason #########
RUN apt-get install libcairo2-dev libjpeg-dev libgif-dev libpango1.0-dev libssl-dev -y
COPY MusicTransformer-tensorflow2.0/requirements.txt /notebooks/MusicTransformer-tensorflow2.0/requirements.txt
RUN pip3 install -r /notebooks/MusicTransformer-tensorflow2.0/requirements.txt --ignore-installed

######### Install Jupyter and Notebook requirements #########
RUN pip3 install google-colab
# https://github.com/jupyter/jupyter_console/issues/163#issuecomment-418392676
RUN pip3 install tensor2tensor prompt-toolkit
RUN pip3 install jupyter
RUN pip3 install --upgrade ipykernel
RUN pip3 uninstall notebook -y
RUN pip3 install --ignore-installed --no-cache-dir --upgrade notebook
RUN pip3 install tensorboard
# Magenta runs an old version of bokeh
# RUN pip uninstall bokeh -y
# RUN pip3 install bokeh==1.4.0

COPY MusicTransformer-tensorflow2.0 /notebooks/MusicTransformer-tensorflow2.0

######### Set up file system #########
RUN mkdir -p /notebooks && chmod a+rwx /notebooks
RUN mkdir -p /logs && chmod 777 /logs
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
RUN chmod -R 777 /notebooks
EXPOSE 8888
COPY musictransformer/dev/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

######### Run Jupyter notebook #########
COPY musictransformer/dev/bash_scripts /bash_scripts/
CMD ["bash", "-c", "/bash_scripts/dev.start.sh"]
