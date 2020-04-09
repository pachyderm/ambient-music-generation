const puppeteer = require('puppeteer');
const fs = require('fs');

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

(async () => {
  const browser = await puppeteer.launch({
    // headless: true,
    args: [
      // '--headless',
      // '--use-gl=desktop'
    ],

    // args: [
    //   '--headless',
    //   '--hide-scrollbars',
    //   '--mute-audio'
    // ]
  });
  const page = await browser.newPage();
  page.on('console', msg => console.log('[PAGE]', msg.text()));
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
  console.log('[MAIN] initialized');

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
      console.log('got ns, sequencing proto to midi');
      const data = mm.sequenceProtoToMidi(ns);
      console.log('transcribed successfully, calling back data');
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

  const files = ['../audio/shortened samples/healing-short.wav', '../audio/shortened samples/brian eno.mp3'];
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const data = await transcribeFile(file);
    const outputPath = `outputs/${file.split('/').pop()}.mid`;
    console.log(`transcribed file, writing to disk at ${outputPath}`);
    fs.writeFileSync(outputPath, data);
  }
  console.log('[MAIN] done');

  await browser.close();
})();
