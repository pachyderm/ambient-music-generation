from python_pachyderm.proto.pfs import pfs_pb2 as pfs_proto


def commit_from(src, allow_just_repo=False):
    if isinstance(src, pfs_proto.Commit):
        return src
    elif isinstance(src, (tuple, list)) and len(src) == 2:
        return pfs_proto.Commit(repo=pfs_proto.Repo(name=src[0]), id=src[1])
    elif isinstance(src, str):
        repo_name, commit_id = src.split('/', 1)
        return pfs_proto.Commit(repo=pfs_proto.Repo(name=repo_name), id=commit_id)

    if not allow_just_repo:
        raise ValueError("Invalid commit type")
    return pfs_proto.Commit(repo=pfs_proto.Repo(name=src))
