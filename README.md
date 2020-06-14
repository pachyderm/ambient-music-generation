# Ambient Music Generation

## Transcription

This stage of the pipeline transcribes incoming `.wav` files into `.midi` files.

The container installs and sets up Magenta's [Onsets & Frames](https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription#training). It clones [this repo](https://github.com/thekevinscott/onsets-and-frames-transcription) to orchestrate the reading of files and transcription.

*Container:* [`hitheory/onsets-and-frames:v1.0`](https://hub.docker.com/repository/docker/hitheory/musictransformer)
*Dockerfile:* [`dockerfiles/onsets-and-frames.Dockerfile`](dockerfiles/onsets-and-frames.Dockerfile)
*Pipeline JSON:* [`pipelines/midi.json`](pipelines/midi.json)
*Entry script:* [`/src/transcribe.py`](https://github.com/thekevinscott/onsets-and-frames-transcription/blob/master/transcribe.py)
*Arguments:*
* `--input` - the directory containing the input files
* `--output` - the directory in which to write the MIDI files

Example:

```
python3 /src/transcribe.py --input /pfs/audio-processed-wav --output /pfs/out
```

## Preprocessing

Before training the model, the data must be pre-processed. Specifically, the MIDI files need to be transformed into a TFRecord format before being used for training.

We rely on [a fork of an implementation](https://github.com/thekevinscott/MusicTransformer-tensorflow2.0) of MusicTransformer by [@jason9693](https://github.com/jason9693).

*Container:* [`hitheory/musictranformer:v1.0.0`](https://hub.docker.com/repository/docker/hitheory/musictransformer)
*Dockerfile:* [`dockerfiles/musictransformer.Dockerfile`](dockerfiles/musictransformer.Dockerfile)
*Pipeline JSON:* [`pipelines/transformer-preprocess.json`](pipelines/transformer-preprocess.json)
*Entry script:* [`/src/preprocess.py`](https://github.com/thekevinscott/MusicTransformer-tensorflow2.0/blob/master/preprocess.py)
*Arguments:*
* `<first arg>` - the directory containing the input files to pre-process
* `<second arg>` - the directory in which to write the pre-processed files

Example:

```
python3 /src/preprocess.py /pfs/midi /pfs/out
```

## Training

Finally, we train on the TFRecord files.

We rely on [the same fork of an implementation](https://github.com/thekevinscott/MusicTransformer-tensorflow2.0) of MusicTransformer by [@jason9693](https://github.com/jason9693), and use the same container and Dockerfile.

*Container:* [`hitheory/musictranformer:v1.0.0`](https://hub.docker.com/repository/docker/hitheory/musictransformer)
*Dockerfile:* [`dockerfiles/musictransformer.Dockerfile`](dockerfiles/musictransformer.Dockerfile)
*Pipeline JSON:* [`pipelines/musictransformer.json`](pipelines/musictransformer.json)
*Entry script:* [`/src/preprocess.py`](https://github.com/thekevinscott/MusicTransformer-tensorflow2.0/blob/master/train.py)
*Arguments:*
* `--l_r` - The learning rate to use. If `None`, [a custom learning rate is used, as defined in the original repo](https://github.com/thekevinscott/MusicTransformer-tensorflow2.0#hyper-parameter).
* `--batch_size` - The batch size to use
* `--max_seq` - The sequence length to use ([more information in the paper](https://arxiv.org/pdf/1809.04281.pdf)).
* `--epochs` - The number of epochs to use
* `--input_path` - The directory containing the files to use for training
* `--save_path` - The directory in which to write the trained model
* `--num_layers` - [The number of layers to use](https://github.com/thekevinscott/MusicTransformer-tensorflow2.0/blob/master/model.py#L15).


Example:

```
python3 /src/train.py --epochs 500 --save_path /pfs/out --input_path /pfs/transformer-preprecess --batch_size 2 --max_seq 2048
```
