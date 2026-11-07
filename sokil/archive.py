"""
Files as release assets: checksums, and a directory packed as one .tar.gz.

An archive is named after the one directory it holds: <name>.tar.gz holds
<name>/, so packing and unpacking need nothing but the archive's path.

Standard library only: settings.py depends on this, and every command imports
settings.py.
"""

import hashlib
import shutil
import tarfile
from pathlib import Path

SUFFIX = ".tar.gz"

CHUNK_BYTES = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_name(archive: Path) -> str:
    """The directory an archive holds, which is its name without .tar.gz."""
    return archive.name.removesuffix(SUFFIX)


def pack_directory(directory: Path, archive: Path) -> Path:
    """
    Pack a directory as archive, under the name the archive's own gives it.

    Hidden files are left out: an Ultralytics export has none, and a file
    browser may have added some.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(directory, arcname=directory_name(archive), filter=_not_hidden)
    return archive


def unpack_directory(archive: Path) -> Path:
    """
    Unpack an archive next to itself, replacing an earlier copy.

    The "data" filter refuses absolute paths, links out of the target and
    device files, so a tampered archive cannot write anywhere else.

    :returns: the directory it unpacked.
    :raises ValueError: if the archive holds anything but its one directory.
    """
    name = directory_name(archive)
    destination = archive.with_name(name)

    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        stray = [
            member
            for member in names
            if member.removeprefix("./").split("/", 1)[0] != name
        ]
        if not names or stray:
            raise ValueError(
                f"{archive.name} should hold just {name}/, "
                f"but also has: {', '.join(stray[:5]) or 'nothing'}"
            )

        if destination.exists():
            shutil.rmtree(destination)
        bundle.extractall(destination.parent, filter="data")

    return destination


def _not_hidden(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if info.name.rsplit("/", 1)[-1].startswith(".") else info
