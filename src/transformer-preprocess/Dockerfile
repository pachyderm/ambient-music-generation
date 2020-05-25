FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
  libgirepository1.0-dev gcc libcairo2-dev \
  libjpeg-dev libgif-dev libpango1.0-dev libssl-dev \
  pkg-config gir1.2-gtk-3.0 software-properties-common \
  git curl wget \
  && add-apt-repository ppa:deadsnakes/ppa -y \
  && apt-get install python3.6 python3-pip -y

# ######### Set up Jason #########
COPY MusicTransformer-tensorflow2.0/requirements.txt /src/requirements.txt
RUN pip3 install -r /src/requirements.txt --ignore-installed

COPY MusicTransformer-tensorflow2.0 /src
