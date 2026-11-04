# Hailo image

The CLI with HailoRT, for running Hailo HEF models on a Raspberry Pi with a
Hailo accelerator, e.g. a Raspberry Pi 5 with a Hailo-8. linux/arm64 only.

```bash
docker buildx build -f docker/hailo/Dockerfile -t sokil-hailo:dev .
docker run --device /dev/hailo0 ... sokil-hailo:dev review ...
```

Releases publish it as `ghcr.io/bonysmoke/sokil` with a `-hailo` suffix on the
CPU image's tags, e.g. `latest-hailo` or `1.2.0-hailo`, built with the
`HAILORT_VERSION` in the Dockerfile; a host on another HailoRT version needs
its own build.

Like `docker/cpu`, it builds from the repo root; `Dockerfile.dockerignore`
here keeps the context to what it needs.

## Why Debian's Python

HailoRT comes from Raspberry Pi's apt repository, the same packages the host
runs, and `python3-hailort` there is built for Debian trixie's Python 3.13. The
CPU image's Python 3.14 is not an option, because HailoRT pins a pybind11 that
predates 3.14. So this image uses Debian's Python rather than the python image.

The venv is made with `--system-site-packages` so it sees `hailo_platform` in
Debian's dist-packages, and the lockfile's packages, numpy included, come first
on the path.

## Matching the host

HailoRT's userspace library talks to the host's kernel driver and the device's
firmware, and all three must be the same version. Check the host's with
`apt-cache policy hailort python3-hailort` and pass them as `HAILORT_VERSION`
and `PYHAILORT_VERSION` build args.

The host keeps the driver (`hailort-pcie-driver`) and the firmware; nothing in
the image installs them.

## The apt setup in `rootfs/`

Copied into the image as is:

- `etc/apt/sources.list.d/raspi.sources`: Raspberry Pi's repository, trusted
  with its own key only.
- `etc/apt/preferences.d/raspi`: pins it so only `hailort` and
  `python3-hailort` come from it.
- `usr/share/keyrings/raspberrypi-archive-keyring.pgp`: the repository's
  signing key.

The key is the one the `raspberrypi-archive-keyring` package, which Raspberry
Pi OS itself installs, ships at `/usr/share/keyrings/`. The standalone
`raspberrypi.gpg.key` on the archive is older and bound with SHA-1, which
trixie's verifier rejects.

| | |
|---|---|
| Package | `raspberrypi-archive-keyring_2025.1+rpt1_all.deb` |
| Package sha256 | `2e727149d7acb8cc7f604e66d0049161039c8aa1eaf1175e54f9e69d963d60e4` |
| Key sha256 | `3a29901549ae65a910de13a32551e6eea7f45ba1fd70f0cb67e5c70d98488071` |

To update it, take the newest package from
https://archive.raspberrypi.com/debian/pool/main/r/raspberrypi-archive-keyring/
and, on a Debian or Raspberry Pi OS machine:

```bash
dpkg-deb --fsys-tarfile raspberrypi-archive-keyring_<version>_all.deb \
    | tar -xO ./usr/share/keyrings/raspberrypi-archive-keyring.pgp \
    > docker/hailo/rootfs/usr/share/keyrings/raspberrypi-archive-keyring.pgp
```

Keep it world-readable: apt verifies as the unprivileged `_apt` user.
