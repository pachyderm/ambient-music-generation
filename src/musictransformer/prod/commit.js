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

const recreate = argv.r;

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

const pipelineJSON = `${docker_name}.json`;

(async () => {
  if (recreate) {
    await exec(`pachctl delete pipeline ${docker_name}`);
  }
  await updatePipeline(pipelineJSON, docker_name);
  console.log('**** building');
  await exec(`docker build -t ${docker_name} -f Dockerfile .`);
  console.log('**** tagging');
  await exec(
    `docker tag ${docker_name} hitheory/${docker_name}:${version}`
  );
  console.log('**** pushing');
  await exec(`docker push hitheory/${docker_name}:${version}`);

  if (recreate) {
    console.log('**** creating pipeline');
    await exec(`pachctl create pipeline -f ./${pipelineJSON}`);
  } else {
    console.log('**** updating pipeline');
    await exec(`pachctl update pipeline -f ./${pipelineJSON}`);
  }
  await wait(5);
  console.log('**** monitoring'); 
  await exec(`pachctl list pipeline | grep ${docker_name}`);
  await wait(2);
  console.log(`pachctl logs -p ${docker_name} -f`)
  // await exec(`pachctl put file dev-audio-unprocessed@master:aphextwin.mp3 -f ../../audio/samples/aphextwin.mp3`);
  // await exec(`pachctl logs -p ${docker_name} -f`);
})();
