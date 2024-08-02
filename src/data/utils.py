import tarfile
import pathlib

from tqdm import tqdm


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


def convert_edge_index_to_adjacency_map(edge_index):
    adjacency_map = {}
    num_of_edges = edge_index.shape[1]
    numpy_edge_index = edge_index.numpy()
    for i in tqdm(range(num_of_edges)):
        u, v = numpy_edge_index[0][i], numpy_edge_index[1][i]
        if u not in adjacency_map:
            adjacency_map[u] = []

        adjacency_map[u].append(v)
    return adjacency_map
