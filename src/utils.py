import json
import pickle
import hashlib
import pathlib

from functools import partial


def serialize_dataset(filename, datamodule):
    with open(filename, 'wb') as handle:
        pickle.dump(datamodule, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_artifact_id(length: int = 8, **params: dict) -> str:
    hashcode = hashlib.sha256()

    for key, value in params.items():
        if not isinstance(value, str) and not isinstance(value, int):
            raise Exception(f"Unsupported type for key {key}")

        hashcode.update(f"{key}={value}".encode('utf-8'))

    return hashcode.hexdigest()[:length]


def create_from_json_file(cls, json_path: pathlib.Path):
    with open(json_path, "r") as f:
        js = json.load(f)

    return partial(cls, **js)


def create_from_json_string(cls, json_str: str):
    return partial(cls, **json.loads(json_str.replace("'", "\"")))
