console.log('v9')

const INPUTS = '/pfs/dev-audio-processed-wav';
const OUTPUTS = '/pfs/out';

const express = require('express');
const puppeteer = require('puppeteer');
const fs = require('fs');

const startExpressServer = () => {
  const app = express();
  const port = 3000;

  app.use(express.static(INPUTS));

  const server = app.listen(port, () => console.log(`Example app listening at http://localhost:${port}`))
  return server;
};

const app = startExpressServer();

console.log(`Reading files from ${INPUTS}`);
const files = fs.readdirSync(INPUTS).filter(file => {
  const ext = file.split('.').pop();
  if ([
    'wav',
    'mp3',
    'aif',
    'aiff',
  ].includes(ext)) {
    return true;
  }
  console.warn(`Ignoring file ${file} as it does not appear to be an audio file.`);
  return false;
});

if (files.length === 0) {
  console.error(`
    No files were found in the ${INPUTS} directory.

    Confirm you correctly mounted a directory to ${INPUTS}.
  `);
  process.exit(1);
}

console.log(`Transcribing ${files.length} audio file${files.length === 1 ? '' : 's'}. Payload: ${JSON.stringify(files)}`);

const getHtml = (filepath) => {
  const html =`
  <html>
    <body>
      <button id="btn" onclick="main()">Start</button>
      <input type="file" id="input" />
      <script>
        console.log('filepath', '${filepath}');
        const addScript = (src) => new Promise(resolve => {
          var s = document.createElement( 'script' );
          s.setAttribute( 'src', src );
          s.onload=resolve;
          document.body.appendChild( s );
        });

        const callback = (fn, ...data) => {
          if (fn) {
            fn(...data);
          } else {
            throw new Error('No callback function could be found.');
          }
        };

        let model;

        async function main() {
          console.log('page main');

          const context = new AudioContext();
          console.log('context made');

          model = new mm.OnsetsAndFrames('https://storage.googleapis.com/magentadata/js/checkpoints/transcription/onsets_frames_uni');
          console.log('initializing model');
          await model.initialize();
          callback(window.initialized);
        }

        function ArrayBufferToString(buffer) {
          return BinaryToString(String.fromCharCode.apply(null, Array.prototype.slice.apply(new Uint8Array(buffer))));
        }

        function BinaryToString(binary) {
          var error;

          try {
            return decodeURIComponent(escape(binary));
          } catch (_error) {
            error = _error;
            if (error instanceof URIError) {
              return binary;
            } else {
              throw error;
            }
          }
        }
      </script>
    </body>
  </html>
  `;
  return html;
};

const transcribeFile = async (file) => {
  const filepath = `${INPUTS}/${file}`;
  console.log(`Transcribe the following file: ${file}`);
  const browser = await puppeteer.launch({
    // headless: false,
    args: [
      // Required for Docker version of Puppeteer
      '--no-sandbox',
      '--disable-setuid-sandbox',

      // This will write shared memory files into /tmp instead of /dev/shm,
      // because Docker’s default for /dev/shm is 64MB
      '--disable-dev-shm-usage',

      // testing
      '--unlimited-storage',
      '--force-gpu-mem-available-mb',
      '--full-memory-crash-report',
    ],
  });
  console.log('Puppeteer launched');
  const page = await browser.newPage();

  page.on('console', msg => {
    if (!msg.text().includes('This browser does not support WebGL')) {
      console.log('[PAGE]', msg.text());
    }
  });
  const url = `data:text/html,${getHtml(filepath).split('\n').filter(line => {
    if (!line) {
      return false;
    }
    if (line.startsWith('//')) {
      return false;
    }
    return true;
  }).join('\n')}`;
  await page.goto(url);
  await page.addScriptTag({ url: "https://cdn.jsdelivr.net/npm/@magenta/music@1.2" });
  await page.addScriptTag({ url: "https://cdn.jsdelivr.net/npm/buffer@5.6.0/index.min.js" });
  console.log('Launched page');

  await page.waitForSelector('#btn');
  console.log('Found #btn');
  const btn = await page.$('#btn');
  console.log('Got #btn element');
  btn.click();
  console.log('Clicked button');
  await page.evaluate(() => new Promise(resolve => {
    window.initialized = resolve;
  }), []);
  console.log('[MAIN] initialized');

  console.log(`[MAIN] begin transcribing file ${file}`);

  // const binaryDataOnDisk = fs.readFileSync(filepath).toString('binary');
  // console.log(`[MAIN] read binary data on disk. Size is: ${getSize(binaryDataOnDisk.length)}`);
  await page.evaluate(async (filepath) => {
    console.log('inner filepath', filepath);
    const resp = await fetch(filepath);
    console.log('got resp back', resp);
    // var oReq = new XMLHttpRequest();
    // oReq.open("GET", "/myfile.png", true);
    // oReq.responseType = "blob";

    // oReq.onload = function(oEvent) {
    //   var blob = oReq.response;
    //   // ...
    // };

    // oReq.send();
    //
    //
    // function checkStatus(response) {
    //   if (!response.ok) {
    //     throw new Error(`HTTP ${response.status} - ${response.statusText}`);
    //   }
    //   return response;
    // }
    //
    // try {
    //   const resp = await fetch(filepath);
    //   const buffer = await checkStatus(response) && response.arrayBuffer();
    //   console.log('buffer', buffer);
    // } catch(err) {
    //   console.error('Error parsing buffer', err);
    // }
  }, filepath);
  // await page.evaluate(async (input) => {
  //   console.log('the binary data is', input);
  // }, binaryDataOnDisk);
  console.log(`[MAIN] continue`);
  // const inputUploadHandle = await page.$('input[type=file]');
  // inputUploadHandle.uploadFile(filepath);


  // const bufferedData = await page.evaluate(async (s) => {
  //   console.log('Begin to read buffered data');
  //   const incomingData = window.buffer.Buffer.from(s, 'binary');
  //   console.log('model transcribe from audio file');
  //   const ns = await model.transcribeFromAudioFile(new Blob([incomingData]));
  //   console.log('go ns, sequent to midi');
  //   const data = mm.sequenceProtoToMidi(ns);
  //   console.log('transcribed successfully, calling back data');
  //   console.log('return to string');
  //   return ArrayBufferToString(data);
  // }, binaryDataOnDisk);
  // console.log('Got data from transcribe');
  // const transcribedData = Buffer.from(bufferedData, 'binary');
  // const outputPath = `${OUTPUTS}/${file.split('/').pop()}.mid`;
  // console.log(`Transcription successful, writing to disk at ${outputPath}`);
  // fs.writeFileSync(outputPath, transcribedData);
  console.log('[MAIN] done, browser is closing');

  await browser.close();
};

(async () => {
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    try {
      await transcribeFile(file);
    } catch(err) {
      console.error('Error transcribing file', err);
    }
  }
  app.close();
  process.exit();
})();

// https://github.com/puppeteer/puppeteer/issues/1260#issue-270736774
async function testForWebGLSupport(page) {
  const webgl = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl');
    const expGl = canvas.getContext('experimental-webgl');

    return {
      gl: gl && gl instanceof WebGLRenderingContext,
      expGl: expGl && expGl instanceof WebGLRenderingContext,
    };
  });

  console.log('WebGL Support:', webgl);
};

const getSize = (size) => {
  if (size / 1024 / 1024 / 1024 > 1) {
    return `${size / 1024 / 1024 / 1024}gb`;
  }
  if (size / 1024 / 1024 > 1) {
    return `${size / 1024 / 1024}mb`;
  }
  if (size / 1024 > 1) {
    return `${size / 1024}kb`;
  }
  return `${size}b`;
};
