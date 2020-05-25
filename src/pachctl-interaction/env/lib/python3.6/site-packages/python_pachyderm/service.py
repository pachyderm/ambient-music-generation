from enum import Enum

from python_pachyderm.proto.admin import admin_pb2 as admin_proto
from python_pachyderm.proto.admin import admin_pb2_grpc as admin_grpc
from python_pachyderm.proto.pfs import pfs_pb2 as pfs_proto
from python_pachyderm.proto.pfs import pfs_pb2_grpc as pfs_grpc
from python_pachyderm.proto.pps import pps_pb2 as pps_proto
from python_pachyderm.proto.pps import pps_pb2_grpc as pps_grpc
from python_pachyderm.proto.transaction import transaction_pb2 as transaction_proto
from python_pachyderm.proto.transaction import transaction_pb2_grpc as transaction_grpc
from python_pachyderm.proto.version.versionpb import version_pb2 as version_proto
from python_pachyderm.proto.version.versionpb import version_pb2_grpc as version_grpc
from python_pachyderm.proto.debug import debug_pb2 as debug_proto
from python_pachyderm.proto.debug import debug_pb2_grpc as debug_grpc
from python_pachyderm.proto.auth import auth_pb2 as auth_proto
from python_pachyderm.proto.auth import auth_pb2_grpc as auth_grpc
from python_pachyderm.proto.enterprise import enterprise_pb2 as enterprise_proto
from python_pachyderm.proto.enterprise import enterprise_pb2_grpc as enterprise_grpc
from python_pachyderm.proto.health import health_pb2 as health_proto
from python_pachyderm.proto.health import health_pb2_grpc as health_grpc


class Service(Enum):
    ADMIN = 0
    AUTH = 1
    DEBUG = 2
    HEALTH = 3
    ENTERPRISE = 4
    PFS = 5
    PPS = 6
    TRANSACTION = 7
    VERSION = 8

    @property
    def grpc_module(self):
        return GRPC_MODULES[self]

    @property
    def stub(self):
        grpc_module = self.grpc_module

        for key in dir(grpc_module):
            if key.endswith("Stub"):
                return getattr(self.grpc_module, key)

    @property
    def servicer(self):
        grpc_module = self.grpc_module

        for key in dir(grpc_module):
            if key.endswith("Servicer"):
                return getattr(self.grpc_module, key)

    @property
    def proto_module(self):
        return PROTO_MODULES[self]


GRPC_MODULES = {
    Service.ADMIN: admin_grpc,
    Service.AUTH: auth_grpc,
    Service.DEBUG: debug_grpc,
    Service.ENTERPRISE: enterprise_grpc,
    Service.HEALTH: health_grpc,
    Service.PFS: pfs_grpc,
    Service.PPS: pps_grpc,
    Service.TRANSACTION: transaction_grpc,
    Service.VERSION: version_grpc,
}

PROTO_MODULES = {
    Service.ADMIN: admin_proto,
    Service.AUTH: auth_proto,
    Service.DEBUG: debug_proto,
    Service.ENTERPRISE: enterprise_proto,
    Service.HEALTH: health_proto,
    Service.PFS: pfs_proto,
    Service.PPS: pps_proto,
    Service.TRANSACTION: transaction_proto,
    Service.VERSION: version_proto,
}
