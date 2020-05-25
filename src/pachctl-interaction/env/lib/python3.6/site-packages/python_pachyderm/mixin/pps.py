import json
import base64

from python_pachyderm.proto.pps import pps_pb2 as pps_proto
from python_pachyderm.service import Service
from .util import commit_from


class PPSMixin:
    def inspect_job(self, job_id, block_state=None, output_commit=None):
        """
        Inspects a job with a given ID. Returns a `JobInfo`.

        Params:

        * `job_id`: The ID of the job to inspect.
        * `block_state`: If true, block until the job completes.
        * `output_commit`: An optional tuple, string, or `Commit` object
        representing an output commit to filter on.
        """
        return self._req(
            Service.PPS, "InspectJob",
            job=pps_proto.Job(id=job_id),
            block_state=block_state,
            output_commit=commit_from(output_commit) if output_commit is not None else None,
        )

    def list_job(self, pipeline_name=None, input_commit=None, output_commit=None, history=None, full=None):
        """
        Lists jobs. Yields `JobInfo` objects.

        Params:

        * `pipeline_name`: An optional string representing a pipeline name to
        filter on.
        * `input_commit`: An optional list of tuples, strings, or `Commit`
        objects representing input commits to filter on.
        * `output_commit`: An optional tuple, string, or `Commit` object
        representing an output commit to filter on.
        * `history`: An optional int that indicates to return jobs from
          historical versions of pipelines. Semantics are:
            * 0: Return jobs from the current version of the pipeline or
              pipelines.
            * 1: Return the above and jobs from the next most recent version
            * 2: etc.
            * -1: Return jobs from all historical versions.
        * `full`: An optional bool indicating whether the result should
        include all pipeline details in each `JobInfo`, or limited information
        including name and status, but excluding information in the pipeline
        spec. Leaving this `None` (or `False`) can make the call significantly
        faster in clusters with a large number of pipelines and jobs. Note
        that if `input_commit` is set, this field is coerced to `True`.
        """
        if isinstance(input_commit, list):
            input_commit = [commit_from(ic) for ic in input_commit]
        elif input_commit is not None:
            input_commit = [commit_from(input_commit)]

        return self._req(
            Service.PPS, "ListJobStream",
            pipeline=pps_proto.Pipeline(name=pipeline_name) if pipeline_name is not None else None,
            input_commit=input_commit,
            output_commit=commit_from(output_commit) if output_commit is not None else None,
            history=history,
            full=full,
        )

    def flush_job(self, commits, pipeline_names=None):
        """
        Blocks until all of the jobs which have a set of commits as
        provenance have finished. Yields `JobInfo` objects.

        Params:

        * `commits`: A list of tuples, strings, or `Commit` objects
        representing the commits to flush.
        * `pipeline_names`: An optional list of strings specifying pipeline
        names. If specified, only jobs within these pipelines will be flushed.
        """
        if pipeline_names is not None:
            to_pipelines = [pps_proto.Pipeline(name=name) for name in pipeline_names]
        else:
            to_pipelines = None

        return self._req(
            Service.PPS, "FlushJob",
            commits=[commit_from(c) for c in commits],
            to_pipelines=to_pipelines,
        )

    def delete_job(self, job_id):
        """
        Deletes a job by its ID.

        Params:

        * `job_id`: The ID of the job to delete.
        """
        return self._req(Service.PPS, "DeleteJob", job=pps_proto.Job(id=job_id))

    def stop_job(self, job_id):
        """
        Stops a job by its ID.

        Params:

        * `job_id`: The ID of the job to stop.
        """
        return self._req(Service.PPS, "StopJob", job=pps_proto.Job(id=job_id))

    def inspect_datum(self, job_id, datum_id):
        """
        Inspects a datum. Returns a `DatumInfo` object.

        Params:

        * `job_id`: The ID of the job.
        * `datum_id`: The ID of the datum.
        """
        return self._req(
            Service.PPS, "InspectDatum",
            datum=pps_proto.Datum(id=datum_id, job=pps_proto.Job(id=job_id)),
        )

    def list_datum(self, job_id, page_size=None, page=None):
        """
        Lists datums. Yields `ListDatumStreamResponse` objects.

        Params:

        * `job_id`: The ID of the job.
        * `page_size`: An optional int specifying the size of the page.
        * `page`: An optional int specifying the page number.
        """
        return self._req(
            Service.PPS, "ListDatumStream",
            job=pps_proto.Job(id=job_id), page_size=page_size, page=page,
        )

    def restart_datum(self, job_id, data_filters=None):
        """
        Restarts a datum.

        Params:

        * `job_id`: The ID of the job.
        * `data_filters`: An optional iterable of strings.
        """
        return self._req(
            Service.PPS, "RestartDatum",
            job=pps_proto.Job(id=job_id), data_filters=data_filters,
        )

    def create_pipeline(self, pipeline_name, transform, parallelism_spec=None, hashtree_spec=None, egress=None,
                        update=None, output_branch=None, resource_requests=None, resource_limits=None, input=None,
                        description=None, cache_size=None, enable_stats=None, reprocess=None, max_queue_size=None,
                        service=None, chunk_spec=None, datum_timeout=None, job_timeout=None, salt=None, standby=None,
                        datum_tries=None, scheduling_spec=None, pod_patch=None, spout=None, spec_commit=None,
                        metadata=None, s3_out=None):
        """
        Creates a pipeline. For more info, please refer to the pipeline spec
        document:
        http://docs.pachyderm.io/en/latest/reference/pipeline_spec.html

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        * `transform`: A `Transform` object.
        * `parallelism_spec`: An optional `ParallelismSpec` object.
        * `hashtree_spec`: An optional `HashtreeSpec` object.
        * `egress`: An optional `Egress` object.
        * `update`: An optional bool specifying whether this should behave as
        an upsert.
        * `output_branch`: An optional string representing the branch to output
        results on.
        * `resource_requests`: An optional `ResourceSpec` object.
        * `resource_limits`: An optional `ResourceSpec` object.
        * `input`: An optional `Input` object.
        * `description`: An optional string describing the pipeline.
        * `cache_size`: An optional string.
        * `enable_stats`: An optional bool.
        * `reprocess`: An optional bool. If true, pachyderm forces the pipeline
        to reprocess all datums. It only has meaning if `update` is `True`.
        * `max_queue_size`: An optional int.
        * `service`: An optional `Service` object.
        * `chunk_spec`: An optional `ChunkSpec` object.
        * `datum_timeout`: An optional `Duration` object.
        * `job_timeout`: An optional `Duration` object.
        * `salt`: An optional string.
        * `standby`: An optional bool.
        * `datum_tries`: An optional int.
        * `scheduling_spec`: An optional `SchedulingSpec` object.
        * `pod_patch`: An optional string.
        * `spout`: An optional `Spout` object.
        * `spec_commit`: An optional `Commit` object.
        * `metadata`: An optional `Metadata` object.
        * `s3_out`: An optional bool specifying whether the output repo should
        be exposed as an s3 gateway bucket.
        """
        return self._req(
            Service.PPS, "CreatePipeline",
            pipeline=pps_proto.Pipeline(name=pipeline_name),
            transform=transform,
            parallelism_spec=parallelism_spec,
            hashtree_spec=hashtree_spec,
            egress=egress,
            update=update,
            output_branch=output_branch,
            resource_requests=resource_requests,
            resource_limits=resource_limits,
            input=input,
            description=description,
            cache_size=cache_size,
            enable_stats=enable_stats,
            reprocess=reprocess,
            max_queue_size=max_queue_size,
            service=service,
            chunk_spec=chunk_spec,
            datum_timeout=datum_timeout,
            job_timeout=job_timeout,
            salt=salt,
            standby=standby,
            datum_tries=datum_tries,
            scheduling_spec=scheduling_spec,
            pod_patch=pod_patch,
            spout=spout,
            spec_commit=spec_commit,
        )

    def create_tf_job_pipeline(self, pipeline_name, tf_job, parallelism_spec=None,
                               hashtree_spec=None, egress=None, update=None, output_branch=None,
                               scale_down_threshold=None, resource_requests=None,
                               resource_limits=None, input=None, description=None, cache_size=None,
                               enable_stats=None, reprocess=None, max_queue_size=None,
                               service=None, chunk_spec=None, datum_timeout=None,
                               job_timeout=None, salt=None, standby=None, datum_tries=None,
                               scheduling_spec=None, pod_patch=None, spout=None, spec_commit=None):
        """
        Creates a pipeline. For more info, please refer to the pipeline spec
        document:
        http://docs.pachyderm.io/en/latest/reference/pipeline_spec.html

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        * `tf_job`: A `TFJob` object. Pachyderm uses this to create TFJobs
        when running in a Kubernetes cluster on which kubeflow has been
        installed.
        * `parallelism_spec`: An optional `ParallelismSpec` object.
        * `hashtree_spec`: An optional `HashtreeSpec` object.
        * `egress`: An optional `Egress` object.
        * `update`: An optional bool specifying whether this should behave as
        an upsert.
        * `output_branch`: An optional string representing the branch to output
        results on.
        * `scale_down_threshold`: An optional `Duration` object.
        * `resource_requests`: An optional `ResourceSpec` object.
        * `resource_limits`: An optional `ResourceSpec` object.
        * `input`: An optional `Input` object.
        * `description`: An optional string describing the pipeline.
        * `cache_size`: An optional string.
        * `enable_stats`: An optional bool.
        * `reprocess`: An optional bool. If true, pachyderm forces the pipeline
        to reprocess all datums. It only has meaning if `update` is `True`.
        * `max_queue_size`: An optional int.
        * `service`: An optional `Service` object.
        * `chunk_spec`: An optional `ChunkSpec` object.
        * `datum_timeout`: An optional `Duration` object.
        * `job_timeout`: An optional `Duration` object.
        * `salt`: An optional string.
        * `standby`: An optional bool.
        * `datum_tries`: An optional int.
        * `scheduling_spec`: An optional `SchedulingSpec` object.
        * `pod_patch`: An optional string.
        * `spout`: An optional `Spout` object.
        * `spec_commit`: An optional `Commit` object.
        """
        return self._req(
            Service.PPS, "CreatePipeline",
            pipeline=pps_proto.Pipeline(name=pipeline_name),
            tf_job=tf_job,
            parallelism_spec=parallelism_spec,
            hashtree_spec=hashtree_spec,
            egress=egress,
            update=update,
            output_branch=output_branch,
            scale_down_threshold=scale_down_threshold,
            resource_requests=resource_requests,
            resource_limits=resource_limits,
            input=input,
            description=description,
            cache_size=cache_size,
            enable_stats=enable_stats,
            reprocess=reprocess,
            max_queue_size=max_queue_size,
            service=service,
            chunk_spec=chunk_spec,
            datum_timeout=datum_timeout,
            job_timeout=job_timeout,
            salt=salt,
            standby=standby,
            datum_tries=datum_tries,
            scheduling_spec=scheduling_spec,
            pod_patch=pod_patch,
            spout=spout,
            spec_commit=spec_commit,
        )

    def inspect_pipeline(self, pipeline_name, history=None):
        """
        Inspects a pipeline. Returns a `PipelineInfo` object.

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        * `history`: An optional int that indicates to return jobs from
        historical versions of pipelines. Semantics are:
            * 0: Return jobs from the current version of the pipeline or
              pipelines.
            * 1: Return the above and jobs from the next most recent version
            * 2: etc.
            * -1: Return jobs from all historical versions.
        """
        pipeline = pps_proto.Pipeline(name=pipeline_name)

        if history is None:
            return self._req(Service.PPS, "InspectPipeline", pipeline=pipeline)
        else:
            # `InspectPipeline` doesn't support history, but `ListPipeline`
            # with a pipeline filter does, so we use that here
            pipelines = self._req(Service.PPS, "ListPipeline", pipeline=pipeline, history=history).pipeline_info
            assert len(pipelines) <= 1
            return pipelines[0] if len(pipelines) else None

    def list_pipeline(self, history=None):
        """
        Lists pipelines. Returns a `PipelineInfos` object.

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        * `history`: An optional int that indicates to return jobs from
        historical versions of pipelines. Semantics are:
            * 0: Return jobs from the current version of the pipeline or
              pipelines.
            * 1: Return the above and jobs from the next most recent version
            * 2: etc.
            * -1: Return jobs from all historical versions.
        """
        return self._req(Service.PPS, "ListPipeline", history=history)

    def delete_pipeline(self, pipeline_name, force=None, keep_repo=None):
        """
        Deletes a pipeline.

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        * `force`: Whether to force delete.
        * `keep_repo`: Whether to keep the repo.
        """
        return self._req(
            Service.PPS,
            "DeletePipeline",
            pipeline=pps_proto.Pipeline(name=pipeline_name),
            force=force,
            keep_repo=keep_repo
        )

    def delete_all_pipelines(self, force=None):
        """
        Deletes all pipelines.

        Params:

        * `force`: Whether to force delete.
        """
        return self._req(Service.PPS, "DeletePipeline", all=True, force=force)

    def start_pipeline(self, pipeline_name):
        """
        Starts a pipeline.

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        """
        return self._req(Service.PPS, "StartPipeline", pipeline=pps_proto.Pipeline(name=pipeline_name))

    def stop_pipeline(self, pipeline_name):
        """
        Stops a pipeline.

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        """
        return self._req(Service.PPS, "StopPipeline", pipeline=pps_proto.Pipeline(name=pipeline_name))

    def run_pipeline(self, pipeline_name, provenance=None, job_id=None):
        """
        Runs a pipeline.

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        * `provenance`: An optional iterable of `CommitProvenance` objects
        representing the pipeline execution provenance.
        * `job_id`: An optional string specifying a specific job ID to run.
        """
        return self._req(
            Service.PPS, "RunPipeline",
            pipeline=pps_proto.Pipeline(name=pipeline_name),
            provenance=provenance,
            job_id=job_id,
        )

    def run_cron(self, pipeline_name):
        """
        Explicitly triggers a pipeline with one or more cron inputs to run
        now.

        Params:

        * `pipeline_name`: A string representing the pipeline name.
        """

        return self._req(
            Service.PPS, "RunCron",
            pipeline=pps_proto.Pipeline(name=pipeline_name),
        )

    def create_secret(self, secret_name, data, labels=None, annotations=None):
        """
        Creates a new secret.

        Params:

        * `secret_name`: The name of the secret to create.
        * `data`: A dict of string keys -> string or bytestring values to
        store in the secret. Each key must consist of alphanumeric characters,
        `-`, `_` or `.`.
        * `labels`: A dict of string keys -> string values representing the
        kubernetes labels to attach to the secret.
        * `annotations`: A dict representing the kubernetes annotations to
        attach to the secret.
        """

        encoded_data = {}
        for k, v in data.items():
            if isinstance(v, str):
                v = v.encode("utf8")
            encoded_data[k] = base64.b64encode(v).decode("utf8")

        f = json.dumps({
            "kind": "Secret",
            "apiVersion": "v1",
            "metadata": {
                "name": secret_name,
                "labels": labels,
                "annotations": annotations,
            },
            "data": encoded_data,
        }).encode("utf8")

        return self._req(Service.PPS, "CreateSecret", file=f)

    def delete_secret(self, secret_name):
        """
        Deletes a new secret.

        Params:

        * `secret_name`: The name of the secret to delete.
        """
        secret = pps_proto.Secret(name=secret_name)
        return self._req(Service.PPS, "DeleteSecret", secret=secret)

    def list_secret(self):
        """
        Lists secrets. Returns a list of `SecretInfo` objects.
        """

        return self._req(
            Service.PPS, "ListSecret",
            req=pps_proto.google_dot_protobuf_dot_empty__pb2.Empty(),
        ).secret_info

    def inspect_secret(self, secret_name):
        """
        Inspects a secret.

        Params:

        * `secret_name`: The name of the secret to inspect.
        """
        secret = pps_proto.Secret(name=secret_name)
        return self._req(Service.PPS, "InspectSecret", secret=secret)

    def delete_all(self):
        """
        Deletes everything in pachyderm.
        """
        return self._req(
            Service.PPS, "DeleteAll",
            req=pps_proto.google_dot_protobuf_dot_empty__pb2.Empty(),
        )

    def get_pipeline_logs(self, pipeline_name, data_filters=None, master=None,
                          datum=None, follow=None, tail=None):
        """
        Gets logs for a pipeline. Yields `LogMessage` objects.

        Params:

        * `pipeline_name`: A string representing a pipeline to get
        logs of.
        * `data_filters`: An optional iterable of strings specifying the names
        of input files from which we want processing logs. This may contain
        multiple files, to query pipelines that contain multiple inputs. Each
        filter may be an absolute path of a file within a pps repo, or it may
        be a hash for that file (to search for files at specific versions.)
        * `master`: An optional bool.
        * `datum`: An optional `Datum` object.
        * `follow`: An optional bool specifying whether logs should continue to
        stream forever.
        * `tail`: An optional int. If nonzero, the number of lines from the end
        of the logs to return.  Note: tail applies per container, so you will
        get tail * <number of pods> total lines back.
        """
        return self._req(
            Service.PPS, "GetLogs",
            pipeline=pps_proto.Pipeline(name=pipeline_name),
            data_filters=data_filters,
            master=master,
            datum=datum,
            follow=follow,
            tail=tail,
        )

    def get_job_logs(self, job_id, data_filters=None, datum=None, follow=None,
                     tail=None):
        """
        Gets logs for a job. Yields `LogMessage` objects.

        Params:

        * `job_id`: A string representing a job to get logs of.
        * `data_filters`: An optional iterable of strings specifying the names
        of input files from which we want processing logs. This may contain
        multiple files, to query pipelines that contain multiple inputs. Each
        filter may be an absolute path of a file within a pps repo, or it may
        be a hash for that file (to search for files at specific versions.)
        * `datum`: An optional `Datum` object.
        * `follow`: An optional bool specifying whether logs should continue to
        stream forever.
        * `tail`: An optional int. If nonzero, the number of lines from the end
        of the logs to return.  Note: tail applies per container, so you will
        get tail * <number of pods> total lines back.
        """
        return self._req(
            Service.PPS, "GetLogs",
            job=pps_proto.Job(id=job_id),
            data_filters=data_filters,
            datum=datum,
            follow=follow,
            tail=tail,
        )

    def garbage_collect(self, memory_bytes=None):
        """
        Runs garbage collection.

        Params:

        * `memory_bytes`: An optional int specifying how much memory to use in
        computing which objects are alive. A larger number will result in more
        precise garbage collection (at the cost of more memory usage).
        """
        return self._req(Service.PPS, "GarbageCollect", memory_bytes=memory_bytes)
