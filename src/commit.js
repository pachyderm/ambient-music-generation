const fs = require("fs");
const yargs = require('yargs');
const child_process = require("child_process");
const argv = require("yargs").argv;


const folder_name = argv.name;
if (!folder_name) {
  throw new Error("Provide a --name for the running docker container");
}

const version = argv.v;
if (!version) {
  throw new Error("Provide a --v for the running docker container");
}

const json = argv.json;
if (!json) {
  throw new Error("Provide a --json for the running docker container");
}

const docker_image_name = folder_name;
const docker_container_name = folder_name;
const pipeline_name = folder_name;

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

const pipelineJSON = `${json}`;

(async () => {
  if (recreate) {
    await exec(`pachctl delete pipeline ${pipeline_name}`);
  }
  await updatePipeline(pipelineJSON, pipeline_name);
  console.log('**** building');
  await exec(`docker build -t ${docker_image_name} -f ${folder_name}/Dockerfile .`);
  console.log('**** tagging');
  await exec(
    `docker tag ${docker_image_name} hitheory/${docker_container_name}:${version}`
  );
  console.log('**** pushing');
  await exec(`docker push hitheory/${docker_container_name}:${version}`);

  if (recreate) {
    console.log('**** creating pipeline');
    await exec(`pachctl create pipeline -f ./${pipelineJSON}`);
  } else {
    console.log('**** updating pipeline');
    await exec(`pachctl update pipeline -f ./${pipelineJSON}`);
  }
  await wait(5);
  console.log('**** monitoring'); 
  await exec(`pachctl list pipeline | grep ${pipeline_name}`);
  await wait(2);
  console.log(`pachctl logs -p ${pipeline_name} -f`)
})();
