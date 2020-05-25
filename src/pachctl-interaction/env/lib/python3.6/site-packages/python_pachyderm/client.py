import os
from urllib.parse import urlparse

from .mixin.admin import AdminMixin
from .mixin.auth import AuthMixin
from .mixin.debug import DebugMixin
from .mixin.enterprise import EnterpriseMixin
from .mixin.health import HealthMixin
from .mixin.pfs import PFSMixin
from .mixin.pps import PPSMixin
from .mixin.transaction import TransactionMixin
from .mixin.version import VersionMixin


class Client(
    AdminMixin,
    AuthMixin,
    DebugMixin,
    EnterpriseMixin,
    HealthMixin,
    PFSMixin,
    PPSMixin,
    TransactionMixin,
    VersionMixin,
    object
):
    def __init__(self, host=None, port=None, auth_token=None, root_certs=None, transaction_id=None, tls=None):
        """
        Creates a Pachyderm client.

        Params:

        * `host`: The pachd host. Default is 'localhost', which is used with
        `pachctl port-forward`.
        * `port`: The port to connect to. Default is 30650.
        * `auth_token`: The authentication token; used if authentication is
        enabled on the cluster. Defaults to `None`.
        * `root_certs`:  The PEM-encoded root certificates as byte string.
        * `transaction_id`: The ID of the transaction to run operations on.
        * `tls`: Specifies whether TLS should be used. If `root_certs` are
        specified, they are used; otherwise, we use the certs provided by
        certifi.
        """

        host = host or "localhost"
        port = port or 30650

        if auth_token is None:
            auth_token = os.environ.get("PACH_PYTHON_AUTH_TOKEN")

        if tls is None:
            tls = root_certs is not None
        if tls and root_certs is None:
            # load default certs if none are specified
            import certifi
            with open(certifi.where(), "rb") as f:
                root_certs = f.read()

        self.address = "{}:{}".format(host, port)
        self.root_certs = root_certs
        self._stubs = {}
        self._auth_token = auth_token
        self._transaction_id = transaction_id
        self._metadata = self._build_metadata()

    @classmethod
    def new_in_cluster(cls, auth_token=None, transaction_id=None):
        """
        Creates a Pachyderm client that operates within a Pachyderm cluster.

        Params:

        * `auth_token`: The authentication token; used if authentication is
        enabled on the cluster. Default to `None`.
        * `transaction_id`: The ID of the transaction to run operations on.
        """

        if "PACHD_PEER_SERVICE_HOST" in os.environ and "PACHD_PEER_SERVICE_PORT" in os.environ:
            # Try to use the pachd peer service if it's available. This is
            # only supported in pachyderm>=1.10, but is more reliable because
            # it'll work when TLS is enabled on the cluster.
            host = os.environ["PACHD_PEER_SERVICE_HOST"]
            port = int(os.environ["PACHD_PEER_SERVICE_PORT"])
        else:
            # Otherwise use the normal service host/port, which will not work
            # when TLS is enabled on the cluster.
            host = os.environ["PACHD_SERVICE_HOST"]
            port = int(os.environ["PACHD_SERVICE_PORT"])

        return cls(host=host, port=port, auth_token=auth_token, transaction_id=transaction_id)

    @classmethod
    def new_from_pachd_address(cls, pachd_address, auth_token=None, root_certs=None, transaction_id=None):
        """
        Creates a Pachyderm client from a given pachd address.

        Params:

        * `auth_token`: The authentication token; used if authentication is
        enabled on the cluster. Default to `None`.
        * `root_certs`: The PEM-encoded root certificates as byte string. If
        unspecified, this will load default certs from certifi.
        * `transaction_id`: The ID of the transaction to run operations on.
        """

        if "://" not in pachd_address:
            pachd_address = "grpc://{}".format(pachd_address)

        u = urlparse(pachd_address)

        if u.scheme not in ("grpc", "http", "grpcs", "https"):
            raise ValueError("unrecognized pachd address scheme: {}".format(u.scheme))
        if u.path != "" or u.params != "" or u.query != "" or u.fragment != "":
            raise ValueError("invalid pachd address")
        if u.username is not None or u.password is not None:
            raise ValueError("invalid pachd address")

        return cls(
            host=u.hostname,
            port=u.port,
            auth_token=auth_token,
            root_certs=root_certs,
            transaction_id=transaction_id,
            tls=u.scheme == "grpcs" or u.scheme == "https",
        )

    @property
    def auth_token(self):
        return self._auth_token

    @auth_token.setter
    def auth_token(self, value):
        self._auth_token = value
        self._metadata = self._build_metadata()

    @property
    def transaction_id(self):
        return self._transaction_id

    @transaction_id.setter
    def transaction_id(self, value):
        self._transaction_id = value
        self._metadata = self._build_metadata()

    def _build_metadata(self):
        metadata = []
        if self._auth_token is not None:
            metadata.append(("authn-token", self._auth_token))
        if self._transaction_id is not None:
            metadata.append(("pach-transaction", self._transaction_id))
        return metadata

    def _req(self, grpc_service, grpc_method_name, req=None, **kwargs):
        stub = self._stubs.get(grpc_service)
        if stub is None:
            grpc_module = grpc_service.grpc_module
            if self.root_certs:
                ssl_channel_credentials = grpc_module.grpc.ssl_channel_credentials
                ssl = ssl_channel_credentials(root_certificates=self.root_certs)
                channel = grpc_module.grpc.secure_channel(self.address, ssl)
            else:
                channel = grpc_module.grpc.insecure_channel(self.address)
            stub = grpc_service.stub(channel)
            self._stubs[grpc_service] = stub

        assert req is None or len(kwargs) == 0
        assert self._metadata is not None

        if req is None:
            proto_module = grpc_service.proto_module
            if grpc_method_name.endswith("Stream"):
                req_cls_name_prefix = grpc_method_name[:-6]
            else:
                req_cls_name_prefix = grpc_method_name
            req_cls = getattr(proto_module, "{}Request".format(req_cls_name_prefix))
            req = req_cls(**kwargs)

        grpc_method = getattr(stub, grpc_method_name)
        return grpc_method(req, metadata=self._metadata)
