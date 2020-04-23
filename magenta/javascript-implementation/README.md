# Javascript implementation

Run with the bash script:

```
./run.sh
```

The bash script will first build the `Dockerfile`, and will then run it. It will automatically mount the directory `../audio/samples` - this can be changed directly in the `run.sh` script to specify a different directory.

The puppeteer Javascript code lives under `src/index.js`, and will read the mounted sample files and transcribe them.

## Changing Directory Paths

You can change the directory names and paths that get mounted.

First, look at `run.sh`, and you'll see two `-v` mount paths: [one for inputs (samples)](https://github.com/thekevinscott/ambient-music-generation/blob/master/magenta/javascript-implementation/run.sh#L12), and [one for outputs (MIDI transcriptions)](https://github.com/thekevinscott/ambient-music-generation/blob/master/magenta/javascript-implementation/run.sh#L13).

To change the internal directory names, you'll also need to modify the Javascript file. Specifically, [you'll want to modify `src/index.js`, lines 4 and 5](https://github.com/thekevinscott/ambient-music-generation/blob/master/magenta/javascript-implementation/src/index.js#L4), to point to the internally mounted directories.

## Current Issues

Puppeteer currently does not see the GPU in headless mode, so headless transcription (within the Docker container) is slower than expected.
