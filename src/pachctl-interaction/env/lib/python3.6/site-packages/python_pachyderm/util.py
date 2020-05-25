import os

from .service import Service
from .mixin import pfs
from .proto.pps.pps_pb2 import Input, Transform, PFSInput, ParallelismSpec

# Default script for running python code with wheels in a pipeline that was
# deployed with `create_python_pipeline`.
RUNNER_SCRIPT_WITH_WHEELS = """
#!/bin/bash
set -{set_args}

cd /pfs/{source_repo_name}
pip install /pfs/{build_pipeline_name}/*.whl
python main.py
"""

# Default script for running python code without wheels in a pipeline that was
# deployed with `create_python_pipeline`.
RUNNER_SCRIPT_WITHOUT_WHEELS = """
#!/bin/bash
set -{set_args}

cd /pfs/{source_repo_name}
python main.py
"""

# Default script for building python wheels for a pipeline that was deployed
# with `create_python_pipeline`.
BUILDER_SCRIPT = """
#!/bin/bash
set -{set_args}
python --version
pip --version

cd /pfs/{source_repo_name}
test -f requirements.txt && pip wheel -r requirements.txt -w /pfs/out
"""


def put_files(client, source_path, commit, dest_path, **kwargs):
    """
    Utility function for recursively inserting files from the local
    `source_path` to pachyderm. Roughly equivalent to `pachctl put file -r`.

    Params:

    * `client`: The `Client` instance to use.
    * `source_path`: The directory to recursively insert content from.
    * `commit`: The `Commit` object to use for inserting files.
    * `dest_path`: The destination path in PFS.
    * `kwargs`: Keyword arguments to forward to `put_file_bytes`.
    """

    def reqs():
        for root, _, filenames in os.walk(source_path):
            for filename in filenames:
                source_filepath = os.path.join(root, filename)
                dest_filepath = os.path.join(dest_path, os.path.relpath(source_filepath, start=source_path))

                with open(source_filepath, "rb") as f:
                    yield from pfs.put_file_from_filelike(commit, dest_filepath, f, **kwargs)

    # Call the lower-level `PutFile` function, because the higher-level ones
    # only support putting a single file at a time
    return client._req(Service.PFS, "PutFile", req=reqs())


def create_python_pipeline(client, path, input=None, pipeline_name=None, image_pull_secrets=None, debug=None,
                           env=None, secrets=None, image=None, update=False, **pipeline_kwargs):
    """
    Utility function for creating (or updating) a pipeline specially built for
    executing python code that is stored locally at `path`. `path` can either
    reference a directory with python code, or a single python file.

    A normal pipeline creation process (i.e. a call to
    `client.create_pipeline`) requires you to first build and push a container
    image with the source and dependencies baked in. As an alternative
    process, this function circumvents container image creation by creating:

    1) a PFS repo that stores the source code at `path`.
    2) If there's a `requirements.txt` in `path`, a  pipeline for building the
    dependencies into wheels.
    3) A pipeline for executing the PFS stored source code with the built
    dependencies.

    This is what the DAG looks like:

    ```
    .------------------------.      .-----------------------.
    | <pipeline_name>_source | ---▶ | <pipeline_name>_build |
    '------------------------'      '-----------------------'
                 |                 /
                 ▼                /
        .-----------------.      /
        | <pipeline_name> | ◀---'
        '-----------------'
                 ▲
                 |
            .---------.
            | <input> |
            '---------'

    ```

    (without a `requirements.txt`, there is no build pipeline.)

    If `path` references a directory, it should have following:

    * A `main.py`, as the pipeline entry-point.
    * An optional `requirements.txt` that specifies pip requirements.
    * An optional `build.sh` if you wish to override the default build
    process.
    * An optional `run.sh` if you wish to override the default pipeline
    execution process.

    Params:

    * `client`: The `Client` instance to use.
    * `path`: The directory containing the python pipeline source, or an
    individual python file.
    * `input`: An optional `Input` object specifying the pipeline input.
    * `pipeline_name`: An optional string specifying the pipeline name.
    Defaults to using the last directory name in `path`.
    * `image_pull_secrets`: An optional list of strings specifying the
    pipeline transform's image pull secrets, which are used for pulling images
    from a private registry. Defaults to `None`, in which case the public
    docker registry will be used. See the pipeline spec document for more
    details.
    * `debug`: An optional bool specifying whether debug logging should be
    enabled for the pipeline. Defaults to `False`.
    * `env`: An optional mapping of string keys to string values specifying
    custom environment variables.
    * `secrets`: An optional list of `Secret` objects for secret environment
    variables.
    * `image`: An optional string specifying the docker image to use for the
    pipeline. Defaults to `python`.
    * `update`: Whether to act as an upsert.
    * `pipeline_kwargs`: Keyword arguments to forward to `create_pipeline`.
    """

    # Verify & set defaults for arguments
    if not os.path.exists(path):
        raise Exception("path does not exist")

    if not os.path.isfile(path) and not os.path.exists(os.path.join(path, "main.py")):
        raise Exception("no main.py detected")

    if pipeline_name is None:
        pipeline_name = os.path.basename(path)
        if os.path.isfile(path):
            if path.endswith(".py"):
                pipeline_name = pipeline_name[:-3]
        else:
            if path.endswith("/"):
                pipeline_name = os.path.basename(path[:-1])

    if not pipeline_name:
        raise Exception("could not derive pipeline name")

    image = image or "python:3"

    # Create the source repo
    source_repo_name = "{}_source".format(pipeline_name)

    client.create_repo(
        source_repo_name,
        description="python_pachyderm.create_python_pipeline: source code for pipeline {}.".format(pipeline_name),
        update=update,
    )

    # Create the build pipeline
    build_pipeline_name = None
    if os.path.exists(os.path.join(path, "requirements.txt")):
        build_pipeline_name = "{}_build".format(pipeline_name)

    if build_pipeline_name is not None:
        build_pipeline_desc = """
            python_pachyderm.create_python_pipeline: build artifacts for pipeline {}.
        """.format(pipeline_name).strip()

        client.create_pipeline(
            build_pipeline_name,
            Transform(
                image=image,
                cmd=["bash", "/pfs/{}/build.sh".format(source_repo_name)],
                image_pull_secrets=image_pull_secrets,
                debug=debug,
            ),
            input=Input(pfs=PFSInput(glob="/", repo=source_repo_name)),
            update=update,
            description=build_pipeline_desc,
            parallelism_spec=ParallelismSpec(constant=1),
        )

    source_commit_desc = "python_pachyderm.create_python_pipeline: sync source code."
    with client.commit(source_repo_name, branch="master", description=source_commit_desc) as commit:
        # Utility function for inserting build.sh/run.sh
        def put_templated_script(filename, template):
            client.put_file_bytes(commit, filename, template.format(
                set_args="ex" if debug else "e",
                source_repo_name=source_repo_name,
                build_pipeline_name=build_pipeline_name,
            ).encode("utf8"))

        # Delete any existing source code
        if update:
            client.delete_file(commit, "/")

        # Insert the source code
        if build_pipeline_name is None:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    client.put_file_bytes(commit, "main.py", f)
            else:
                put_files(client, path, commit, "/")

            put_templated_script("run.sh", RUNNER_SCRIPT_WITHOUT_WHEELS)
        else:
            put_files(client, path, commit, "/")

            if not os.path.exists(os.path.join(path, "run.sh")):
                put_templated_script("run.sh", RUNNER_SCRIPT_WITH_WHEELS)
            if not os.path.exists(os.path.join(path, "build.sh")):
                put_templated_script("build.sh", BUILDER_SCRIPT)

    # Create the pipeline
    inputs = [Input(pfs=PFSInput(glob="/", repo=source_repo_name))]

    if input is not None:
        inputs.append(input)
    if build_pipeline_name is not None:
        inputs.append(Input(pfs=PFSInput(glob="/", repo=build_pipeline_name)))

    return client.create_pipeline(
        pipeline_name,
        Transform(
            image=image,
            cmd=["bash", "/pfs/{}/run.sh".format(source_repo_name)],
            image_pull_secrets=image_pull_secrets,
            debug=debug,
            env=env,
            secrets=secrets,
        ),
        input=Input(cross=inputs) if len(inputs) > 1 else inputs[0],
        update=update,
        **pipeline_kwargs
    )
