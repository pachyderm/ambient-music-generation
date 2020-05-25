from python_pachyderm.service import Service
from python_pachyderm.proto.health import health_pb2_grpc as health_grpc


class HealthMixin:
    def health(self):
        return self._req(Service.HEALTH, "Health", req=health_grpc.google_dot_protobuf_dot_empty__pb2.Empty())
