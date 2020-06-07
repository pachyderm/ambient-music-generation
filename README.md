# Ambient Music Generation

## Pipelines

The overall structure looks like this:

```
  audio-unprocessed - the original, unprocessed audio files ()
  audio-processed-wav - the processed audio files (transformed into .wav)
  midi - the wav files transformed into midi files
  transformer-preprocess - the midi files muxed into a format appropriate for training
  musictranformer - the training pipeline
```
