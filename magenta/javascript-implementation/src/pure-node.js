// These hacks below are needed because the library uses performance and fetch which
// exist in browsers but not in node. We are working on simplifying this!
global.performance = Date;
global.fetch = require('node-fetch');

const fs = require('fs');
const FileReader = require('filereader');
const fileReader = new FileReader()
const Window = require('window');
global.window = new Window();
var AudioContext = require('web-audio-api').AudioContext
  , context = new AudioContext
  , Speaker = require('speaker')

context.outStream = new Speaker({
  channels: context.format.numberOfChannels,
  bitDepth: context.format.bitDepth,
  sampleRate: context.sampleRate
})

global.AudioContext = context;
class OfflineAudioContext {
  constructor(_, SAMPLE_RATE_1, SAMPLE_RATE_2) {
  }
}
window.OfflineAudioContext = OfflineAudioContext
const transcription = require('@magenta/music/node/transcription');

const readFile = path => new Promise((resolve, reject) => {
  // fileReader.setNodeChunkedEncoding(true || false);
  fileReader.readAsDataURL(new File(path));

  // // non-standard alias of `addEventListener` listening to non-standard `data` event
  // fileReader.on('data', function (data) {
  //   console.log("chunkSize:", data.length);
  // });

  // // `onload` as listener
  // fileReader.addEventListener('load', function (ev) {
  //   console.log("dataUrlSize:", ev.target.result.length);
  // });

  // `onloadend` as property
  fileReader.on('loadend', () => {
    console.log("Success");
    resolve();
  });
});

(async () => {
  const model = new transcription.OnsetsAndFrames('https://storage.googleapis.com/magentadata/js/checkpoints/transcription/onsets_frames_uni');
  console.log('initializing model');

  await model.initialize();
  console.log('initialized model');

  const file_path = '../audio/shortened\ samples/healing-short.wav'
  const file = fs.readFileSync(file_path);
  console.log('read file', file);
  const ns = await model.transcribeFromAudioBuffer(file);
  // const ns = await model.transcribeFromAudioFile(new File(file_path));
  console.log('got ns', ns);
  const data = mm.sequenceProtoToMidi(ns);
  console.log('got data', data);
  fs.writeFileSync('./test.mid', data);
  // saveAs(new File([data], 'transcription.mid'));
})();


  // fileInput.addEventListener('change', (e) => {
  //   transcribeFromFile(e.target.files[0]);
  // });

  // async function transcribeFromFile(blob) {
  //   const ns = await model.transcribeFromAudioFile(blob);
  //   const data = mm.sequenceProtoToMidi(ns);
  //   saveAs(new File([data], 'transcription.mid'));
  // }
// }
