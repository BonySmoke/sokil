"""
Where the trained weights come from: reading the manifest, resolving a model to
a file on disk, and fetching one that is not there yet.

Nothing here reaches the network — requests is replaced — and every environment
variable the module reads is cleared first, so a developer's own SOKIL_* setup
cannot change what these tests see.
"""

import hashlib
import io
import tarfile

import pytest
import requests

from sokil import settings
from sokil.archive import pack_directory, sha256
from sokil.settings import (
    ModelFormat,
    ModelName,
    ModelResolutionError,
    ModelSpec,
    _load_manifest,
    cache_dir,
    clear_cache,
    download_model,
    download_models,
    ensure_model_path,
    manifest,
    manifest_path,
    model_dir,
    model_names,
    model_spec,
    models_repo,
    release_asset_digests,
    resolve_model_path,
    search_paths,
)

MANIFEST = """
repo: someone/sokil
models:
  shuttle:
    version: "v3"
    release_tag: models-v3
    asset: shuttle-detect.pt
    sha256: {shuttle_sha}
    expects:
      imgsz: 640
      grayscale: true
    formats:
      hailo8:
        asset: shuttle-detect-hailo8_hailo_model.tar.gz
        sha256: {hailo_sha}
  court:
    version: "v3"
    release_tag: models-v3
    asset: court-segment.pt
    sha256:
"""

SHUTTLE_BYTES = b"shuttle weights"
SHUTTLE_SHA = hashlib.sha256(SHUTTLE_BYTES).hexdigest()

HAILO_DIRECTORY = "shuttle-detect-hailo8_hailo_model"


def archive(*members: tuple[str, bytes]) -> bytes:
    """A .tar.gz holding the given (name, content) files."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


HAILO_BYTES = archive(
    (f"{HAILO_DIRECTORY}/train-hailo8.hef", b"hef"),
    (f"{HAILO_DIRECTORY}/metadata.yaml", b"task: detect"),
)
HAILO_SHA = hashlib.sha256(HAILO_BYTES).hexdigest()

SOKIL_VARIABLES = (
    "SOKIL_MODELS_MANIFEST",
    "SOKIL_MODELS_REPO",
    "SOKIL_MODELS_DIR",
    "SOKIL_CACHE_DIR",
    "SOKIL_MODEL_FORMAT",
    "MODEL_DIR",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """
    Point the module at a manifest and a cache of this test's own.

    The manifest loader is memoised on the path, so the cache is cleared too:
    two tests writing different manifests to the same tmp path would otherwise
    see each other's.
    """
    for variable in SOKIL_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    # search_paths also consults ./models, and the repository this runs from
    # has a real one — so give the tests a working directory of their own.
    monkeypatch.chdir(tmp_path)

    manifest_file = tmp_path / "models.yaml"
    manifest_file.write_text(
        MANIFEST.format(shuttle_sha=SHUTTLE_SHA, hailo_sha=HAILO_SHA)
    )
    monkeypatch.setenv("SOKIL_MODELS_MANIFEST", str(manifest_file))
    monkeypatch.setenv("SOKIL_CACHE_DIR", str(tmp_path / "cache"))
    # the checkout's own models/ directory is outside the tests' control
    monkeypatch.setattr(settings, "CHECKOUT_MODELS_DIR", tmp_path / "checkout-models")

    _load_manifest.cache_clear()
    yield manifest_file
    _load_manifest.cache_clear()


class FakeResponse:
    def __init__(self, status_code=200, chunks=(), payload=None):
        self.status_code = status_code
        self._chunks = chunks
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size):
        yield from self._chunks

    def json(self):
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    """
    Replace requests.get with a recorded, scripted response.

    Called with a FakeResponse every request gets that one; called with a
    function it is handed the URL, so one test can script several assets.
    """
    calls = []

    def install(response):
        def get(url, **kwargs):
            calls.append((url, kwargs))
            return response(url) if callable(response) else response

        monkeypatch.setattr(settings.requests, "get", get)
        return calls

    return install


class TestManifest:
    def test_the_manifest_path_follows_the_environment(self, isolated_environment):
        assert manifest_path() == isolated_environment

    def test_the_shipped_manifest_is_the_default(self, monkeypatch):
        monkeypatch.delenv("SOKIL_MODELS_MANIFEST")

        assert manifest_path() == settings.DEFAULT_MANIFEST_PATH

    def test_the_shipped_manifest_is_readable_and_pins_both_models(self, monkeypatch):
        monkeypatch.delenv("SOKIL_MODELS_MANIFEST")
        _load_manifest.cache_clear()

        assert set(model_names()) == {name.value for name in ModelName}

    def test_it_is_read_once_per_path(self, isolated_environment):
        manifest()
        isolated_environment.write_text("repo: rewritten\nmodels: {}\n")

        assert manifest()["repo"] == "someone/sokil"

    def test_a_missing_manifest_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SOKIL_MODELS_MANIFEST", str(tmp_path / "gone.yaml"))
        _load_manifest.cache_clear()

        with pytest.raises(ModelResolutionError, match="Could not read the manifest"):
            manifest()

    def test_a_manifest_that_is_not_yaml_is_reported(self, isolated_environment):
        isolated_environment.write_text("models: [unclosed\n")
        _load_manifest.cache_clear()

        with pytest.raises(ModelResolutionError, match="not valid YAML"):
            manifest()

    def test_a_manifest_without_models_is_reported(self, isolated_environment):
        isolated_environment.write_text("repo: someone/sokil\n")
        _load_manifest.cache_clear()

        with pytest.raises(ModelResolutionError, match="no 'models' section"):
            manifest()


class TestModelSpec:
    def test_reads_one_entry(self):
        spec = model_spec("shuttle")

        assert spec == ModelSpec(
            name="shuttle",
            version="v3",
            release_tag="models-v3",
            asset="shuttle-detect.pt",
            sha256=SHUTTLE_SHA,
            expects={"imgsz": 640, "grayscale": True},
        )

    def test_an_enum_names_the_same_entry_as_the_string(self):
        assert model_spec(ModelName.SHUTTLE) == model_spec("shuttle")

    def test_an_unknown_model_lists_the_known_ones(self):
        with pytest.raises(ModelResolutionError, match="court, shuttle"):
            model_spec("racket")

    def test_an_entry_missing_a_required_field_is_reported(self, isolated_environment):
        isolated_environment.write_text(
            "repo: x\nmodels:\n  shuttle:\n    version: v1\n"
        )
        _load_manifest.cache_clear()

        with pytest.raises(ModelResolutionError, match="release_tag, asset"):
            model_spec("shuttle")

    def test_an_unpinned_checksum_is_none_rather_than_empty(self):
        assert model_spec("court").sha256 is None

    def test_a_missing_expects_block_is_an_empty_dict(self):
        assert model_spec("court").expects == {}

    def test_the_cache_path_carries_the_version(self):
        spec = model_spec("shuttle")

        assert spec.cache_path == cache_dir() / "shuttle-v3" / "shuttle-detect.pt"

    def test_bumping_the_version_is_a_cache_miss(self, isolated_environment):
        before = model_spec("shuttle").cache_path
        isolated_environment.write_text(
            MANIFEST.format(shuttle_sha=SHUTTLE_SHA, hailo_sha=HAILO_SHA).replace(
                '"v3"', '"v4"', 1
            )
        )
        _load_manifest.cache_clear()

        assert model_spec("shuttle").cache_path != before

    def test_the_download_url_points_at_the_pinned_release(self):
        assert model_spec("shuttle").url == (
            "https://github.com/someone/sokil/releases/download/"
            "models-v3/shuttle-detect.pt"
        )

    def test_the_api_url_points_at_the_same_release(self):
        assert model_spec("shuttle").api_url == (
            "https://api.github.com/repos/someone/sokil/releases/tags/models-v3"
        )

    def test_a_spec_prints_as_its_name_and_version(self):
        assert str(model_spec("shuttle")) == "shuttle v3"


class TestRepoAndDirectories:
    def test_the_repo_comes_from_the_manifest(self):
        assert models_repo() == "someone/sokil"

    def test_the_environment_overrides_it(self, monkeypatch):
        monkeypatch.setenv("SOKIL_MODELS_REPO", "fork/sokil")

        assert models_repo() == "fork/sokil"

    def test_a_manifest_naming_no_repo_cannot_download(self, isolated_environment):
        isolated_environment.write_text("models:\n  shuttle:\n    version: v1\n")
        _load_manifest.cache_clear()

        with pytest.raises(ModelResolutionError, match="names no 'repo'"):
            models_repo()

    def test_no_model_dir_is_configured_by_default(self):
        assert model_dir() is None

    def test_the_configured_model_dir_is_read(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SOKIL_MODELS_DIR", str(tmp_path))

        assert model_dir() == tmp_path

    def test_the_retired_variable_is_ignored_with_a_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("MODEL_DIR", "/somewhere")

        assert model_dir() is None
        assert "SOKIL_MODELS_DIR" in caplog.text

    def test_the_cache_directory_follows_the_environment(self, tmp_path):
        assert cache_dir() == tmp_path / "cache"

    def test_without_configuration_the_cache_is_a_user_cache_directory(
        self, monkeypatch
    ):
        monkeypatch.delenv("SOKIL_CACHE_DIR")

        assert cache_dir().parts[-1] == "models"


class TestSearchPaths:
    def test_the_checkout_and_working_directory_come_before_the_cache(self, tmp_path):
        paths = search_paths("shuttle")

        assert paths == [
            tmp_path / "checkout-models" / "shuttle-detect.pt",
            tmp_path / "models" / "shuttle-detect.pt",
            model_spec("shuttle").cache_path,
        ]

    def test_a_models_directory_beside_the_working_directory_is_searched(
        self, tmp_path
    ):
        # what the container relies on: the package is installed, so the
        # checkout path points into site-packages, and the repository is
        # mounted at the working directory instead
        assert tmp_path / "models" / "shuttle-detect.pt" in search_paths("shuttle")

    def test_the_same_directory_is_not_listed_twice(self, monkeypatch, tmp_path):
        # run from its own root, the checkout IS the working directory
        monkeypatch.setattr(settings, "CHECKOUT_MODELS_DIR", tmp_path / "models")

        paths = search_paths("shuttle")

        assert paths.count(tmp_path / "models" / "shuttle-detect.pt") == 1

    def test_a_configured_directory_comes_first_of_all(self, monkeypatch, tmp_path):
        explicit = tmp_path / "explicit"
        monkeypatch.setenv("SOKIL_MODELS_DIR", str(explicit))

        assert search_paths("shuttle")[0] == explicit / "shuttle-detect.pt"


class TestResolveModelPath:
    def test_finds_a_checkout_copy(self, tmp_path):
        checkout = tmp_path / "checkout-models"
        checkout.mkdir()
        weights = checkout / "shuttle-detect.pt"
        weights.write_bytes(b"anything at all")

        assert resolve_model_path("shuttle") == weights

    def test_finds_a_cached_copy(self):
        cached = model_spec("shuttle").cache_path
        cached.parent.mkdir(parents=True)
        cached.write_bytes(SHUTTLE_BYTES)

        assert resolve_model_path("shuttle") == cached

    def test_a_cached_copy_that_does_not_match_the_checksum_is_still_used(self, caplog):
        cached = model_spec("shuttle").cache_path
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"an older download")

        assert resolve_model_path("shuttle") == cached
        assert "does not match the checksum" in caplog.text

    def test_an_unpinned_model_is_used_but_reported_as_unverified(self, caplog):
        cached = model_spec("court").cache_path
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"court weights")

        assert resolve_model_path("court") == cached
        assert "unverified" in caplog.text

    def test_nothing_on_disk_names_everywhere_it_looked(self):
        with pytest.raises(ModelResolutionError, match="looked in:"):
            resolve_model_path("shuttle")

    def test_it_never_downloads(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("resolve_model_path must stay offline")

        monkeypatch.setattr(settings.requests, "get", refuse)

        with pytest.raises(ModelResolutionError):
            resolve_model_path("shuttle")


class TestEnsureModelPath:
    def test_an_existing_model_is_returned_without_a_download(self, tmp_path):
        checkout = tmp_path / "checkout-models"
        checkout.mkdir()
        (checkout / "shuttle-detect.pt").write_bytes(b"weights")

        assert ensure_model_path("shuttle") == checkout / "shuttle-detect.pt"

    def test_a_missing_model_is_downloaded(self, fake_get):
        fake_get(FakeResponse(chunks=[SHUTTLE_BYTES]))

        path = ensure_model_path("shuttle")

        assert path.read_bytes() == SHUTTLE_BYTES

    def test_an_empty_models_dir_is_reported_before_falling_back(
        self, monkeypatch, tmp_path, fake_get, caplog
    ):
        monkeypatch.setenv("SOKIL_MODELS_DIR", str(tmp_path / "mount"))
        fake_get(FakeResponse(chunks=[SHUTTLE_BYTES]))

        ensure_model_path("shuttle")

        assert "holds no weights" in caplog.text


class TestDownloadModel:
    def test_writes_the_asset_into_the_cache(self, fake_get):
        calls = fake_get(FakeResponse(chunks=[b"shuttle ", b"weights"]))

        path = download_model("shuttle")

        assert path == model_spec("shuttle").cache_path
        assert path.read_bytes() == SHUTTLE_BYTES
        assert calls[0][0] == model_spec("shuttle").url

    def test_an_already_cached_model_is_not_fetched_again(self, monkeypatch):
        cached = model_spec("shuttle").cache_path
        cached.parent.mkdir(parents=True)
        cached.write_bytes(SHUTTLE_BYTES)
        monkeypatch.setattr(
            settings.requests,
            "get",
            lambda *a, **k: pytest.fail("should not re-download"),
        )

        assert download_model("shuttle") == cached

    def test_force_fetches_it_again(self, fake_get):
        cached = model_spec("shuttle").cache_path
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"stale")
        fake_get(FakeResponse(chunks=[SHUTTLE_BYTES]))

        assert download_model("shuttle", force=True).read_bytes() == SHUTTLE_BYTES

    def test_a_corrupted_transfer_is_deleted_and_reported(self, fake_get):
        fake_get(FakeResponse(chunks=[b"truncated"]))

        with pytest.raises(ModelResolutionError, match="Checksum mismatch"):
            download_model("shuttle")

        spec = model_spec("shuttle")
        assert not spec.cache_path.exists()
        assert not spec.cache_path.with_suffix(".pt.part").exists()

    def test_a_missing_release_asset_says_where_to_look(self, fake_get):
        fake_get(FakeResponse(status_code=404))

        with pytest.raises(ModelResolutionError, match="models-v3"):
            download_model("shuttle")

    def test_a_network_failure_leaves_no_half_written_weights(self, monkeypatch):
        def explode(*args, **kwargs):
            raise requests.ConnectionError("no route to host")

        monkeypatch.setattr(settings.requests, "get", explode)

        with pytest.raises(ModelResolutionError, match="Could not download"):
            download_model("shuttle")

        assert not model_spec("shuttle").cache_path.exists()

    def test_an_unpinned_model_downloads_but_says_it_was_not_verified(
        self, fake_get, caplog
    ):
        fake_get(FakeResponse(chunks=[b"court weights"]))

        download_model("court")

        assert "not" in caplog.text and "verified" in caplog.text

    def test_downloading_everything_covers_every_manifest_entry(self, fake_get):
        payloads = {
            "shuttle-detect.pt": SHUTTLE_BYTES,
            "court-segment.pt": b"court weights",
        }
        fake_get(lambda url: FakeResponse(chunks=[payloads[url.rsplit("/", 1)[-1]]]))

        downloaded = download_models()

        assert set(downloaded) == {"shuttle", "court"}
        assert downloaded["shuttle"].read_bytes() == SHUTTLE_BYTES


class TestSha256AndCache:
    def test_hashes_a_file(self, tmp_path):
        path = tmp_path / "weights.pt"
        path.write_bytes(SHUTTLE_BYTES)

        assert sha256(path) == SHUTTLE_SHA

    def test_clearing_an_empty_cache_is_not_an_error(self):
        assert clear_cache() == cache_dir()

    def test_clearing_removes_the_cached_weights(self):
        cached = model_spec("shuttle").cache_path
        cached.parent.mkdir(parents=True)
        cached.write_bytes(SHUTTLE_BYTES)

        clear_cache()

        assert not cached.exists()


class TestReleaseAssetDigests:
    def test_reads_the_sha256_the_api_reports(self, fake_get):
        fake_get(
            FakeResponse(
                payload={
                    "assets": [
                        {"name": "shuttle-detect.pt", "digest": f"sha256:{SHUTTLE_SHA}"}
                    ]
                }
            )
        )

        assert release_asset_digests("models-v3") == {"shuttle-detect.pt": SHUTTLE_SHA}

    def test_an_asset_with_no_digest_is_skipped(self, fake_get):
        fake_get(FakeResponse(payload={"assets": [{"name": "old.pt"}]}))

        assert release_asset_digests("models-v3") == {}

    def test_a_digest_of_another_algorithm_is_skipped(self, fake_get):
        fake_get(
            FakeResponse(payload={"assets": [{"name": "old.pt", "digest": "md5:abc"}]})
        )

        assert release_asset_digests("models-v3") == {}

    def test_a_missing_release_says_so(self, fake_get):
        fake_get(FakeResponse(status_code=404))

        with pytest.raises(ModelResolutionError, match="does not exist"):
            release_asset_digests("models-v9")


class TestWorkingDirectoryWeights:
    """
    Resolving weights that sit in ./models — the path an installed sokil needs,
    because CHECKOUT_MODELS_DIR then points inside site-packages.
    """

    def test_weights_in_the_working_directory_are_found(self, tmp_path):
        weights = tmp_path / "models" / "shuttle-detect.pt"
        weights.parent.mkdir()
        weights.write_bytes(SHUTTLE_BYTES)

        assert resolve_model_path("shuttle") == weights

    def test_they_are_used_without_a_download(self, tmp_path, monkeypatch):
        weights = tmp_path / "models" / "shuttle-detect.pt"
        weights.parent.mkdir()
        weights.write_bytes(SHUTTLE_BYTES)
        monkeypatch.setattr(
            settings.requests, "get", lambda *a, **k: pytest.fail("must not download")
        )

        assert ensure_model_path("shuttle") == weights

    def test_a_checkout_copy_still_wins(self, tmp_path):
        checkout = tmp_path / "checkout-models"
        checkout.mkdir()
        (checkout / "shuttle-detect.pt").write_bytes(b"checkout")
        working = tmp_path / "models"
        working.mkdir()
        (working / "shuttle-detect.pt").write_bytes(b"working directory")

        assert resolve_model_path("shuttle") == checkout / "shuttle-detect.pt"

    def test_an_explicit_models_dir_still_wins_over_both(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        (explicit / "shuttle-detect.pt").write_bytes(b"explicit")
        working = tmp_path / "models"
        working.mkdir()
        (working / "shuttle-detect.pt").write_bytes(b"working directory")
        monkeypatch.setenv("SOKIL_MODELS_DIR", str(explicit))

        assert resolve_model_path("shuttle") == explicit / "shuttle-detect.pt"


class TestFormats:
    """A model pinned as an accelerator build, picked by SOKIL_MODEL_FORMAT."""

    @pytest.fixture
    def hailo(self, monkeypatch):
        monkeypatch.setenv("SOKIL_MODEL_FORMAT", "hailo8")

    def test_the_weights_are_the_default_format(self):
        spec = model_spec("shuttle")

        assert spec.format == "pt"
        assert spec.asset == "shuttle-detect.pt"

    def test_the_environment_picks_a_format(self, hailo):
        spec = model_spec("shuttle")

        assert spec.asset == f"{HAILO_DIRECTORY}.tar.gz"
        assert spec.sha256 == HAILO_SHA
        # the format is published with the weights it was built from
        assert spec.release_tag == "models-v3"
        assert spec.expects == {"imgsz": 640, "grayscale": True}

    def test_an_explicit_format_wins_over_the_environment(self, hailo):
        assert model_spec("shuttle", "pt").asset == "shuttle-detect.pt"

    def test_an_unknown_format_in_the_environment_lists_the_known_ones(
        self, monkeypatch
    ):
        monkeypatch.setenv("SOKIL_MODEL_FORMAT", "haillo8")

        with pytest.raises(ModelResolutionError, match="pt, hailo8, hailo8l"):
            model_spec("shuttle")

    def test_an_unknown_format_in_the_manifest_is_reported_on_load(
        self, isolated_environment
    ):
        isolated_environment.write_text(
            MANIFEST.format(shuttle_sha=SHUTTLE_SHA, hailo_sha=HAILO_SHA).replace(
                "      hailo8:", "      haillo8:"
            )
        )
        _load_manifest.cache_clear()

        with pytest.raises(ModelResolutionError, match="'haillo8'"):
            manifest()

    def test_the_weights_are_not_a_format_of_their_own(self, isolated_environment):
        isolated_environment.write_text(
            MANIFEST.format(shuttle_sha=SHUTTLE_SHA, hailo_sha=HAILO_SHA).replace(
                "      hailo8:", "      pt:"
            )
        )
        _load_manifest.cache_clear()

        with pytest.raises(ModelResolutionError, match="pt under formats"):
            manifest()

    def test_a_spec_carries_the_enum(self, hailo):
        assert model_spec("shuttle").format is ModelFormat.HAILO8

    def test_a_format_the_manifest_does_not_pin_is_reported(self, hailo):
        with pytest.raises(ModelResolutionError, match="no hailo8 build of 'court'"):
            model_spec("court")

    def test_an_archive_resolves_to_the_directory_it_unpacks_to(self, hailo):
        spec = model_spec("shuttle")

        assert spec.is_archive
        assert spec.local_name == HAILO_DIRECTORY
        assert spec.cache_model_path == spec.cache_path.with_name(HAILO_DIRECTORY)
        assert str(spec) == "shuttle v3 (hailo8)"

    def test_a_local_directory_is_found(self, hailo, tmp_path):
        local = tmp_path / "models" / HAILO_DIRECTORY
        local.mkdir(parents=True)

        assert resolve_model_path("shuttle") == local

    def test_a_file_of_the_directory_name_is_not_taken_for_it(self, hailo, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / HAILO_DIRECTORY).write_bytes(b"not a directory")

        with pytest.raises(ModelResolutionError):
            resolve_model_path("shuttle")

    def test_a_download_is_verified_and_unpacked(self, hailo, fake_get):
        fake_get(FakeResponse(chunks=[HAILO_BYTES]))

        path = download_model("shuttle")

        assert path == model_spec("shuttle").cache_model_path
        assert (path / "train-hailo8.hef").read_bytes() == b"hef"
        assert resolve_model_path("shuttle") == path

    def test_an_unpacked_download_is_not_fetched_again(self, hailo, fake_get):
        calls = fake_get(FakeResponse(chunks=[HAILO_BYTES]))
        download_model("shuttle")

        ensure_model_path("shuttle")

        assert len(calls) == 1

    def test_an_archive_with_anything_else_is_refused(
        self, hailo, fake_get, isolated_environment
    ):
        stray = archive(
            (f"{HAILO_DIRECTORY}/train-hailo8.hef", b"hef"), ("elsewhere.txt", b"x")
        )
        isolated_environment.write_text(
            MANIFEST.format(
                shuttle_sha=SHUTTLE_SHA, hailo_sha=hashlib.sha256(stray).hexdigest()
            )
        )
        _load_manifest.cache_clear()
        fake_get(FakeResponse(chunks=[stray]))

        with pytest.raises(ModelResolutionError, match="elsewhere.txt"):
            download_model("shuttle")

    def test_a_packaged_directory_is_what_a_download_unpacks(
        self, hailo, fake_get, isolated_environment, tmp_path
    ):
        build = tmp_path / "train-hailo8_hailo_model"
        build.mkdir()
        (build / "train-hailo8.hef").write_bytes(b"hef")
        (build / ".hidden").write_bytes(b"left by a file browser")

        packaged = pack_directory(build, tmp_path / f"{HAILO_DIRECTORY}.tar.gz")
        isolated_environment.write_text(
            MANIFEST.format(shuttle_sha=SHUTTLE_SHA, hailo_sha=sha256(packaged))
        )
        _load_manifest.cache_clear()
        fake_get(FakeResponse(chunks=[packaged.read_bytes()]))

        path = download_model("shuttle")

        assert path.name == HAILO_DIRECTORY
        assert sorted(child.name for child in path.iterdir()) == ["train-hailo8.hef"]
