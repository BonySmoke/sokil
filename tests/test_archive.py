"""
A directory packed as one .tar.gz and unpacked again, as release assets are.
"""

import io
import tarfile
from pathlib import Path

import pytest

from sokil.archive import directory_name, pack_directory, sha256, unpack_directory


@pytest.fixture
def build(tmp_path):
    """A directory as an export leaves it, plus a file a file browser added."""
    directory = tmp_path / "train-hailo8_hailo_model"
    directory.mkdir()
    (directory / "train-hailo8.hef").write_bytes(b"hef")
    (directory / "metadata.yaml").write_text("task: detect")
    (directory / ".hidden").write_bytes(b"left by a file browser")
    return directory


def test_the_directory_is_named_after_the_archive():
    assert directory_name(Path("a/shuttle_hailo_model.tar.gz")) == "shuttle_hailo_model"


def test_a_packed_directory_unpacks_under_the_archive_name(build, tmp_path):
    archive = pack_directory(build, tmp_path / "out" / "shuttle_hailo_model.tar.gz")

    unpacked = unpack_directory(archive)

    assert unpacked == archive.with_name("shuttle_hailo_model")
    assert sorted(child.name for child in unpacked.iterdir()) == [
        "metadata.yaml",
        "train-hailo8.hef",
    ]
    assert (unpacked / "train-hailo8.hef").read_bytes() == b"hef"


def test_unpacking_replaces_an_earlier_copy(build, tmp_path):
    archive = pack_directory(build, tmp_path / "shuttle_hailo_model.tar.gz")
    stale = archive.with_name("shuttle_hailo_model")
    stale.mkdir()
    (stale / "left-over.hef").write_bytes(b"old")

    unpack_directory(archive)

    assert not (stale / "left-over.hef").exists()


def test_an_archive_holding_anything_else_is_refused(tmp_path):
    archive = tmp_path / "shuttle_hailo_model.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in ("shuttle_hailo_model/model.hef", "elsewhere.txt"):
            info = tarfile.TarInfo(name)
            info.size = 1
            bundle.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="elsewhere.txt"):
        unpack_directory(archive)

    assert not (tmp_path / "elsewhere.txt").exists()


def test_hashes_a_file(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"abc")

    assert (
        sha256(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
