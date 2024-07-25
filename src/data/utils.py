import tarfile
import pathlib


def extract_from_archive(src: pathlib.Path, files: list[str], dst: pathlib.Path):
    all_exist = True
    for artifact in files:
        if not (dst / artifact).exists():
            all_exist = False

    if all_exist:
        return None

    with tarfile.open(src) as tar:
        artifact_members = list()
        for member in tar.getmembers():
            if member.name in files:
                artifact_members.append(member)
        tar.extractall(members=artifact_members, path=dst)

    return None
