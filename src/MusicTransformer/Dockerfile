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
# RUN git clone https://github.com/jason9693/MusicTransformer-tensorflow2.0.git /notebooks/MusicTransformer-tensorflow2.0
RUN pip3 install requests
RUN pip3 install pycairo
RUN pip3 install pygobject --ignore-installed
# Install heavy dependencies so they're cached
RUN pip3 install pandas==0.24.2 numpy==1.18.4
RUN apt-get install libcairo2-dev libjpeg-dev libgif-dev libpango1.0-dev libssl-dev -y
RUN pip3 install grpcio==1.28.1
RUN pip3 install tensorflow-gpu==2.2.0rc4
RUN pip3 install pygobject==3.36.0
RUN pip3 install six==1.12.0 boto3==1.9.115 docker==4.0.2 dvc==0.50.1 grandalf==0.6 Flask==1.0.3
RUN pip3 install cryptography==2.1.4
RUN pip3 install Pillow==6.2.0 SQLAlchemy==1.3.5 sqlparse==0.3.0 tf-estimator-nightly==1.14.0.dev2019060501 tfp-nightly==0.8.0.dev20190807
RUN pip3 install SecretStorage==2.3.1 shortuuid==0.5.0 simplejson==3.16.0 pycrypto==2.6.1 pretty-midi==0.2.8
RUN pip3 install shortuuid==0.5.0 simplejson==3.16.0
RUN pip3 install absl-py==0.7.1 alembic==1.0.11 appdirs==1.4.3 asciimatics==1.11.0 asn1crypto==0.24.0
RUN pip3 install astor==0.8.0 boto3==1.9.115 botocore==1.12.180 certifi==2018.1.18
RUN pip3 install chardet==3.0.4 Click==7.0 cloudpickle==1.1.1 colorama==0.4.1
RUN pip3 install config==0.4.2 configobj==5.0.6 configparser==3.7.4 contextlib2==0.5.5
RUN pip3 install cryptography==2.1.4 databricks-cli==0.8.7 decorator==4.4.0 distro==1.4.0
RUN pip3 install docker==4.0.2 docutils==0.14 dvc==0.50.1 entrypoints==0.3
RUN pip3 install Flask==1.0.3 funcy==1.12 future==0.17.1 gast==0.3.3
RUN pip3 install git-url-parse==1.2.2 gitdb2==2.0.5 GitPython==2.1.11 
RUN pip3 install google-pasta==0.1.8
RUN pip3 install grandalf==0.6 gunicorn==19.9.0 h5py==2.10.0
RUN pip3 install humanize==0.5.1 idna==2.6 inflect==2.1.0 itsdangerous==1.1.0
RUN pip3 install Jinja2==2.10.1 jmespath==0.9.4 jsonpath-ng==1.4.3 Keras-Applications==1.0.8
RUN pip3 install Keras-Preprocessing==1.1.0 keyring==10.6.0 keyrings.alt==3.0 Mako==1.0.12
RUN pip3 install Markdown==3.1.1 MarkupSafe==1.1.1 mido==1.2.9 mlflow==1.0.0
RUN pip3 install nanotime==0.5.2 networkx==2.3 
RUN pip3 install pathspec==0.5.9 pbr==5.3.1 Pillow==6.2.0 ply==3.11
RUN pip3 install pretty-midi==0.2.8 progress==1.5 protobuf==3.8.0 psutil==5.6.2
RUN pip3 install pyasn1==0.4.6
RUN pip3 install pycrypto==2.6.1 pyfiglet==0.8.post1 pygobject==3.36.0
RUN pip3 install pyparsing==2.4.0 python-dateutil==2.8.0 python-editor==1.0.4 pytz==2019.1
RUN pip3 install pyxdg==0.25 PyYAML==5.1.1 querystring-parser==1.2.3 requests==2.22.0
RUN pip3 install s3transfer==0.2.1 schema==0.7.0 SecretStorage==2.3.1
RUN pip3 install shortuuid==0.5.0 simplejson==3.16.0 six==1.12.0 smmap2==2.0.5
RUN pip3 install SQLAlchemy==1.3.5 sqlparse==0.3.0 ssh-import-id==5.7 tabulate==0.8.3
RUN pip3 install tb-nightly==1.14.0a20190603 termcolor==1.1.0 tf-estimator-nightly==1.14.0.dev2019060501
RUN pip3 install tfp-nightly==0.8.0.dev20190807 treelib==1.5.5 urllib3==1.22 wcwidth==0.1.7
RUN pip3 install websocket-client==0.56.0 Werkzeug==0.15.4 wrapt==1.11.1 zc.lockfile==1.4
RUN pip3 install ruamel.yaml==0.16.10


COPY MusicTransformer-tensorflow2.0 /notebooks/MusicTransformer-tensorflow2.0

RUN sed 's/^pygobject==.*/pygobject==3.36.0/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^tensorflow-gpu==.*/tensorflow-gpu==2.2.0rc4/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^grpcio==.*/grpcio==1.28.1/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^h5py==.*/h5py==2.10.0/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^google-pasta==.*/gast==0.1.8/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^ruamel.yaml==.*/ruamel.yaml==0.16.10/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^gast==.*/gast==0.3.3/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^numpy==.*/numpy==0.18.4/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed 's/^pyasn1==.*/pyasn1==0.4.6/g' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed '72d' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN sed '7d' /notebooks/MusicTransformer-tensorflow2.0/requirements.txt -i
RUN pip3 install -r /notebooks/MusicTransformer-tensorflow2.0/requirements.txt --ignore-installed





# ######### Install System Packages #########
# RUN apt-get update && apt-get install wget curl zip libcairo2-dev libgirepository1.0-dev -y
# RUN apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0 -y
# # RUN apt-get update && apt-get install wget curl zip -y

# ######### Set up Jason #########
# COPY MusicTransformer-pytorch /notebooks/MusicTransformer-pytorch
# COPY MusicTransformer-tensorflow2.0 /notebooks/MusicTransformer-tensorflow2.0
# RUN pip3 install requests
# RUN pip3 install pycairo
# RUN pip3 install PyGObject
# # RUN pip3 install pygobject==3.27.0 --user
# RUN pip3 install -r /notebooks/MusicTransformer-tensorflow2.0/requirements.txt --ignore-installed

######### Install Jupyter and Notebook requirements #########
RUN pip3 install google-colab
RUN pip3 install jupyter
# https://github.com/jupyter/jupyter_console/issues/163#issuecomment-418392676
RUN pip3 install --upgrade ipykernel
RUN pip3 install tensor2tensor prompt-toolkit
# Magenta runs an old version of bokeh
# RUN pip uninstall bokeh -y
# RUN pip3 install bokeh==1.4.0


######### Set up file system #########
RUN mkdir -p /notebooks && chmod a+rwx /notebooks
RUN mkdir -p /logs && chmod 777 /logs
RUN mkdir /.local && chmod a+rwx /.local
WORKDIR /notebooks
EXPOSE 8888
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

######### Install jason #########
WORKDIR /notebooks/MusicTransformer-tensorflow2.0
RUN git clone https://github.com/jason9693/midi-neural-processor.git \
  && mv midi-neural-processor midi_processor
RUN sh dataset/scripts/classic_piano_downloader.sh ./classic_piano
RUN python3 preprocess.py ./classic_piano ./dataset_out

######### Run Jupyter notebook #########
CMD ["bash", "-c", "jupyter notebook --ip 0.0.0.0 --allow-root --no-browser"]
