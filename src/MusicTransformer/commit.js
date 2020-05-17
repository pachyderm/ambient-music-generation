const fs = require("fs");
const argv = require("yargs").argv;
const child_process = require("child_process");

const docker_name = argv.name;
if (!docker_name) {
  throw new Error("Provide a --name for the running docker container");
}

const version = argv.v;
if (!version) {
  throw new Error("Provide a --v for the running docker container");
}

const exec = (cmd) =>
  new Promise((resolve) => {
    const child = child_process.exec(cmd);
    child.stdout.pipe(process.stdout);
    child.on("exit", resolve);
  });

const wait = (d) => new Promise((resolve) => setTimeout(resolve, d));

const updatePipeline = (target, targetContainerName) => {
  const pipeline = JSON.parse(fs.readFileSync(target));
  pipeline.transform.image = `hitheory/${targetContainerName}:${version}`;
  fs.writeFileSync(target, JSON.stringify(pipeline, null, 2));
};

const pipelineJSON = 'musictransformer.json';

(async () => {
  const targetContainerName = "musictransformer";
  await updatePipeline(pipelineJSON, targetContainerName);
  await exec(`docker build -t ${docker_name} -f Dockerfile .`);
  await exec(
    `docker tag ${docker_name} hitheory/${targetContainerName}:${version}`
  );
  await exec(`docker push hitheory/${targetContainerName}:${version}`);

  await exec(`pachctl update pipeline -f ./${pipelineJSON}`);
  // await exec(`pachctl put file dev-audio-unprocessed@master:aphextwin.mp3 -f ../../audio/samples/aphextwin.mp3`);
  await wait(2);
  // await exec(`pachctl logs -p dev-midi -f`);
})();
