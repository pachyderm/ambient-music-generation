const puppeteer = require('puppeteer');
const fs = require('fs');

const INPUTS = '/pfs/dev-audio-processed-wav';
const OUTPUTS = '/pfs/out';

// testThatOutputFolderIsWritable();

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

console.log(`Transcribing ${files.length} audio file${files.length === 1 ? '' : 's'}`);

const html =`
<html>
  <body>
    <button id="btn" onclick="main()">Start</button>
    <script>
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
        await Promise.all([
          addScript("https://cdn.jsdelivr.net/npm/@magenta/music@1.2"),
          addScript("https://bundle.run/buffer@5.5.0"),
        ]);

        const context = new AudioContext();

        model = new mm.OnsetsAndFrames('https://storage.googleapis.com/magentadata/js/checkpoints/transcription/onsets_frames_uni');
        console.log('initializing model');
        await model.initialize();
        callback(window.initialized);
      }

    </script>
  </body>
</html>
`;

const transcribeFile = async (filepath) => {
  const browser = await puppeteer.launch({
    // headless: false,
    args: [
      // Required for Docker version of Puppeteer
      '--no-sandbox',
      '--disable-setuid-sandbox',
      // This will write shared memory files into /tmp instead of /dev/shm,
      // because Docker’s default for /dev/shm is 64MB
      '--disable-dev-shm-usage'
    ],
  });
  const page = await browser.newPage();

  // await testForWebGLSupport();

  // page.on('console', msg => console.log('[PAGE]', msg.text()));
  const url = `data:text/html,${html.split('\n').filter(line => {
    if (!line) {
      return false;
    }
    if (line.startsWith('//')) {
      return false;
    }
    return true;
  }).join('\n')}`;
  await page.goto(url);

  await page.waitForSelector('#btn');
  const btn = await page.$('#btn');
  btn.click();
  await page.evaluate(() => new Promise(resolve => {
    window.initialized = resolve;
  }), []);
  // console.log('[MAIN] initialized');

  const transcribeFile = async file => {
    console.log(`[MAIN] begin transcribing file ${file}`);

    // // https://github.com/puppeteer/puppeteer/issues/2427#issuecomment-536002538
    // return await Buffer.from(page.evaluate(async (s) => {
    //   const bufferedData = window.buffer.Buffer(s, 'binary');
    //   return bufferedData.toString('binary');
    // }, fs.readFileSync(file).toString('binary')));

    return Buffer.from(await page.evaluate(async (s) => {
      const incomingData = window.buffer.Buffer.from(s, 'binary');
      const ns = await model.transcribeFromAudioFile(new Blob([incomingData]));
      // console.log('got ns, sequencing proto to midi');
      const data = mm.sequenceProtoToMidi(ns);
      // console.log('transcribed successfully, calling back data');
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

      return ArrayBufferToString(data);
    }, fs.readFileSync(file).toString('binary')), 'binary');
  };

  console.log(`Transcribing file ${filepath} ...`);
  const data = await transcribeFile(`${INPUTS}/${filepath}`);
  const outputPath = `${OUTPUTS}/${file.split('/').pop()}.mid`;
  console.log(`Transcription successful, writing to disk at ${outputPath}`);
  fs.writeFileSync(outputPath, data);
  // console.log('[MAIN] done');

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
})();

// function testThatOutputFolderIsWritable () {
//   const path = `test-${Math.random()}`;
//   const outputPath = `${OUTPUTS}/path`;
//   const expected = 'test content';
//   fs.writeFileSync(outputPath, expected, 'utf-8');
//   try {
//     const contents = fs.readFileSync(outputPath, 'utf-8');

//     // allow loose equality
//     if (contents != expected) {
//       console.error(`Could not write to output path ${OUTPUTS}. The written file did not match.\n\nExpected: ${expected}\n\nReceived: ${contents}`);
//       process.exit(1);
//     }
//   } catch(err) {
//     console.error(`Could not write to output path ${OUTPUTS}. Error was:`)
//     console.error(err.stack);
//     process.exit(1);
//   }
// };

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
}
