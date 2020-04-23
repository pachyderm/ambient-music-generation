const fs = require('fs');
const argv = require('yargs').argv;
const child_process = require('child_process');

const docker_name = argv.name;
if (!docker_name) {
  throw new Error('Provide a --name for the running docker container');
}

const version = argv.v;
if (!version) {
  throw new Error('Provide a --v for the running docker container');
}

const exec = (cmd) => new Promise(resolve => {
  const child = child_process.exec(cmd);
  child.stdout.pipe(process.stdout)
  child.on('exit', resolve);
});

const wait = d => new Promise(resolve => setTimeout(resolve, d));

const updatePipeline = (target) => {
  const pipeline = JSON.parse(fs.readFileSync(target));
  pipeline.transform.image = `hitheory/dev-midi-transcriber-dj:${version}`;
  fs.writeFileSync(target, JSON.stringify(pipeline, null, 2));
};

const updateScript = (target) => {
  const script = fs.readFileSync(target, 'utf8');
  const newScript = [`console.log('${version}')`].concat(
    script.split('\n').slice(1)
  );
  fs.writeFileSync(target, newScript.join('\n'));
};

(async () => {
  await updatePipeline('midi.json');
  await updateScript('./src/index.js');

  await exec(`docker cp src/index.js ${docker_name}:/code/src/index.js`)
  await exec(`docker cp package.json ${docker_name}:/code/package.json`)
  await exec(`docker commit -m "Update transcription src" ${docker_name} hitheory/dev-midi-transcriber-dj:${version}`);
  await exec(`docker push hitheory/dev-midi-transcriber-dj:${version}`)

  await exec(`pachctl update pipeline -f ./midi.json`);
  await exec(`pachctl put file dev-audio-unprocessed@master:healing-short.mp3 -f ../../audio/samples/healing-short.mp3`);
  await wait(2);
  // await exec(`pachctl logs -p dev-midi -f`);
})();
