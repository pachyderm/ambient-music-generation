from python_pachyderm.service import Service
from python_pachyderm.proto.version.versionpb import version_pb2_grpc as version_grpc


class VersionMixin:
    def get_remote_version(self):
        return self._req(Service.VERSION, "GetVersion", req=version_grpc.google_dot_protobuf_dot_empty__pb2.Empty())
