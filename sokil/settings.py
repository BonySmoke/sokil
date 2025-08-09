"""
Where the trained weights come from.

Nothing here decides which model to use: models.yaml pins a version and a
checksum per model, and this module only resolves what the manifest names,
downloading it when it is not already on disk.

Resolution order for one model:

1. SOKIL_MODELS_DIR, if set — an explicit directory of weights.
2. The checkout's own models/ directory — a model just trained, not yet
   published.
3. The download cache, keyed by version.
4. Otherwise it is downloaded from the release the manifest pins, and the run
   fails if that is not possible.
"""

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

import platformdirs
import requests
import yaml

logger = logging.getLogger(__name__)

# Shipped with the package: an installed Sokil has no checkout to read, so the
# manifest has to travel with the code that depends on it.
DEFAULT_MANIFEST_PATH = Path(__file__).parent / "models.yaml"

# A checkout's own weights, tried before the cache so a model that was just
# trained can be reviewed with before it is published anywhere.
CHECKOUT_MODELS_DIR = Path(__file__).parent.parent / "models"

DOWNLOAD_CHUNK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT_SECONDS = 30
API_TIMEOUT_SECONDS = 15


class ModelName(str, Enum):
    SHUTTLE = "shuttle"
    COURT = "court"


class ModelResolutionError(RuntimeError):
    """A model could not be resolved to a usable file on disk."""


@dataclass(frozen=True)
class ModelSpec:
    """One model, as the manifest pins it (see models.yaml)."""

    name: str
    version: str
    release_tag: str
    asset: str
    sha256: str | None
    expects: dict

    @property
    def cache_path(self) -> Path:
        """
        Where a downloaded copy of THIS version lives.

        The version is part of the path, so bumping the manifest is a cache
        miss rather than a stale hit — which is the whole point — and two
        versions coexist, so moving between branches does not re-download.
        """
        return cache_dir() / f"{self.name}-{self.version}" / self.asset

    @property
    def url(self) -> str:
        return (
            f"https://github.com/{models_repo()}/releases/download/"
            f"{self.release_tag}/{self.asset}"
        )

    @property
    def api_url(self) -> str:
        return (
            f"https://api.github.com/repos/{models_repo()}/releases/tags/"
            f"{self.release_tag}"
        )

    def __str__(self) -> str:
        return f"{self.name} {self.version}"


def manifest_path() -> Path:
    """The manifest in use. SOKIL_MODELS_MANIFEST points at another one."""
    configured = os.environ.get("SOKIL_MODELS_MANIFEST")
    return Path(configured) if configured else DEFAULT_MANIFEST_PATH


@lru_cache(maxsize=4)
def _load_manifest(path: str) -> dict:
    try:
        document = yaml.safe_load(Path(path).read_text())
    except OSError as error:
        raise ModelResolutionError(f"Could not read the manifest {path}: {error}")
    except yaml.YAMLError as error:
        raise ModelResolutionError(f"The manifest {path} is not valid YAML: {error}")

    if not isinstance(document, dict) or "models" not in document:
        raise ModelResolutionError(f"The manifest {path} has no 'models' section")

    return document


def manifest() -> dict:
    return _load_manifest(str(manifest_path()))


def model_names() -> list[str]:
    """Every model the manifest pins."""
    return list(manifest()["models"])


def model_spec(name: ModelName | str) -> ModelSpec:
    """The manifest entry for one model."""
    key = name.value if isinstance(name, ModelName) else str(name)
    entries = manifest()["models"]

    if key not in entries:
        known = ", ".join(sorted(entries))
        raise ModelResolutionError(f"Unknown model '{key}'. Known models: {known}")

    entry = entries[key]
    missing = [
        field for field in ("version", "release_tag", "asset") if not entry.get(field)
    ]
    if missing:
        raise ModelResolutionError(
            f"The manifest entry for '{key}' is missing: {', '.join(missing)}"
        )

    # Everything is coerced to str because YAML infers types: an unquoted
    # date version arrives as datetime.date and an all-digit checksum as int.
    checksum = entry.get("sha256")
    return ModelSpec(
        name=key,
        version=str(entry["version"]),
        release_tag=str(entry["release_tag"]),
        asset=str(entry["asset"]),
        sha256=str(checksum).strip() if checksum else None,
        expects=entry.get("expects") or {},
    )


def models_repo() -> str:
    """The repository whose releases carry the assets."""
    configured = os.environ.get("SOKIL_MODELS_REPO")
    if configured:
        return configured

    repo = manifest().get("repo")
    if not repo:
        raise ModelResolutionError(
            "The manifest names no 'repo', so weights cannot be downloaded. "
            "Set SOKIL_MODELS_REPO, or pass the model paths explicitly."
        )
    return str(repo)


def model_dir() -> Path | None:
    """The explicitly configured directory of weights, if SOKIL_MODELS_DIR is set."""
    configured = os.environ.get("SOKIL_MODELS_DIR")

    # The variable used to be MODEL_DIR. It is not honoured under that name any
    # more, but silently ignoring it would send someone hunting for why their
    # weights are being downloaded instead of read.
    if not configured and os.environ.get("MODEL_DIR"):
        logger.warning(
            "MODEL_DIR is set but is no longer read — the variable is now "
            "SOKIL_MODELS_DIR. Resolving weights without it."
        )

    return Path(configured) if configured else None


def cache_dir() -> Path:
    """Where downloads are kept, shared by every checkout on the machine."""
    configured = os.environ.get("SOKIL_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path(platformdirs.user_cache_dir("sokil")) / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def search_paths(name: ModelName | str) -> list[Path]:
    """Every location this model is looked for, in precedence order."""
    spec = model_spec(name)
    configured = model_dir()

    candidates = [] if configured is None else [configured / spec.asset]
    candidates.append(CHECKOUT_MODELS_DIR / spec.asset)
    candidates.append(Path.cwd() / "models" / spec.asset)
    candidates.append(spec.cache_path)

    seen: set[Path] = set()
    paths = []
    for path in candidates:
        resolved = Path(os.path.normpath(path.absolute()))
        if resolved in seen:
            continue

        seen.add(resolved)
        paths.append(path)
    return paths


def resolve_model_path(name: ModelName | str) -> Path:
    """
    The path to an already-present model.

    Never downloads, so a notebook or a sweep over saved tracks stays offline.
    Use ensure_model_path for the fetch-if-missing behaviour.
    """
    spec = model_spec(name)
    candidates = search_paths(name)

    for path in candidates:
        if not path.is_file():
            continue

        if path == spec.cache_path:
            # A cached copy is one we fetched, so it is the one the manifest
            # checksum describes. A mismatch is reported and then tolerated:
            # refusing to run would strand someone mid-session over a file
            # that is very probably just an older download.
            _warn_unless_verified(path, spec)
        else:
            logger.info("Using the local %s weights at %s", spec.name, path)

        logger.debug("Resolved %s to %s", spec, path)
        return path

    looked_in = ", ".join(str(path) for path in candidates)
    raise ModelResolutionError(
        f"No weights for {spec} found (looked in: {looked_in}). Fetch them with "
        f"`sokil models download`, or pass the model path explicitly."
    )


def ensure_model_path(name: ModelName | str) -> Path:
    """
    The path to a model, downloading it if it is not already on disk.

    A missing model is always fetched — including when SOKIL_MODELS_DIR is set
    and did not contain it, which is reported so a broken mount is visible
    rather than silent.
    """
    try:
        return resolve_model_path(name)
    except ModelResolutionError:
        configured = model_dir()
        if configured is not None:
            logger.warning(
                "SOKIL_MODELS_DIR is %s but holds no weights for %s; "
                "downloading instead",
                configured,
                model_spec(name),
            )
        return download_model(name)


def _warn_unless_verified(path: Path, spec: ModelSpec) -> bool:
    """Check a file against the manifest checksum, reporting rather than raising."""
    if spec.sha256 is None:
        logger.warning(
            "The manifest pins no checksum for %s, so %s is unverified",
            spec,
            path.name,
        )
        return True

    actual = sha256(path)
    if actual == spec.sha256:
        logger.debug("%s matches the manifest checksum", path.name)
        return True

    logger.warning(
        "%s does not match the checksum the manifest pins for %s "
        "(expected %s, found %s). Continuing with it; run "
        "`sokil models download --force` to replace it.",
        path,
        spec,
        spec.sha256[:12],
        actual[:12],
    )
    return False


def download_model(name: ModelName | str, force: bool = False) -> Path:
    """
    Fetch one model from the release the manifest pins.

    Downloads to a sibling .part file and renames on success, so an interrupted
    run leaves no half-written weights for the next one to pick up. A fresh
    download that does not match the manifest checksum is deleted and raises:
    unlike an older cached copy, it can only be a corrupted transfer.
    """
    spec = model_spec(name)
    target = spec.cache_path

    if target.is_file() and not force:
        logger.info("%s already cached at %s", spec, target)
        _warn_unless_verified(target, spec)
        return target

    url = spec.url
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    logger.info("Downloading %s -> %s", url, target)
    try:
        with requests.get(
            url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            if response.status_code == 404:
                raise ModelResolutionError(
                    f"{url} returned 404. Check that release "
                    f"'{spec.release_tag}' exists in {models_repo()} and has a "
                    f"'{spec.asset}' asset."
                )
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
    except requests.RequestException as error:
        partial.unlink(missing_ok=True)
        raise ModelResolutionError(f"Could not download {url}: {error}") from error

    if spec.sha256 is None:
        logger.warning(
            "The manifest pins no checksum for %s, so the download was not "
            "verified. Pin one with `sokil models pin`.",
            spec,
        )
    else:
        actual = sha256(partial)
        if actual != spec.sha256:
            partial.unlink(missing_ok=True)
            raise ModelResolutionError(
                f"Checksum mismatch for {spec.asset}: the manifest pins "
                f"{spec.sha256}, the download hashed to {actual}. The transfer "
                f"was corrupted — retry."
            )
        logger.debug("Verified %s against the manifest checksum", spec.asset)

    partial.replace(target)
    logger.info("Cached %s at %s", spec, target)
    return target


def download_models(force: bool = False) -> dict[str, Path]:
    """Fetch every model the manifest pins."""
    return {name: download_model(name, force=force) for name in model_names()}


def clear_cache() -> Path:
    """Remove the download cache and return the directory that was cleared."""
    directory = cache_dir()
    if directory.exists():
        shutil.rmtree(directory)
        logger.info("Removed %s", directory)
    return directory


def release_asset_digests(release_tag: str) -> dict[str, str]:
    """
    The sha256 GitHub computed for each asset of a published release.

    Used by `sokil models pin` to fill the manifest from a release that is
    already published. Pinning from the local files before publishing is
    preferred, since it needs no network and no trust in the API.
    """
    url = f"https://api.github.com/repos/{models_repo()}/releases/tags/{release_tag}"

    response = requests.get(
        url,
        timeout=API_TIMEOUT_SECONDS,
        headers={"Accept": "application/vnd.github+json"},
    )
    if response.status_code == 404:
        raise ModelResolutionError(
            f"{url} returned 404. Either release '{release_tag}' does not "
            f"exist in {models_repo()}, or the repository is private and this "
            f"machine is not authenticated."
        )
    response.raise_for_status()

    digests = {}
    for asset in response.json().get("assets", []):
        # the API reports "sha256:<hex>"; older releases may carry no digest
        algorithm, _, value = (asset.get("digest") or "").partition(":")
        if asset.get("name") and algorithm == "sha256" and value:
            digests[asset["name"]] = value

    return digests
