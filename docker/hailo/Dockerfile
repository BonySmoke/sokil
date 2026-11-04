# syntax=docker/dockerfile:1

# The CLI with HailoRT, for a Raspberry Pi with a Hailo accelerator. linux/arm64
# only. Build from the repo root; see README.md here for why it is built this
# way and how to match the host's HailoRT.

FROM debian:trixie-slim AS base

ARG HAILORT_VERSION=4.23.0
ARG PYHAILORT_VERSION=4.23.0-1

# Raspberry Pi's apt repository, its key, and the pins that take only HailoRT
# from it.
COPY docker/hailo/rootfs/ /

# opencv-python links against the system GL and glib even when it never opens a
# window; the t64 suffix is trixie's name for glib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        "hailort=${HAILORT_VERSION}" \
        "python3-hailort=${PYHAILORT_VERSION}" \
        libgl1 libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Debian's Python, so the venv sees hailo_platform in its dist-packages.
RUN uv venv --system-site-packages --python /usr/bin/python3 /opt/venv

# Dependencies first, without the project, so a source change keeps them cached.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-default-groups --no-install-project

# --no-editable copies the project into the venv, as /app is not in the runtime.
COPY sokil/ /app/sokil/
COPY pyproject.toml uv.lock README.md /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable


FROM base AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SOKIL_CACHE_DIR=/cache \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# ultralytics and matplotlib write a config on import, and /cache holds the
# downloaded weights (mount a volume to keep them). All world-writable, as the
# image runs under whatever --user the host passes.
RUN useradd --create-home --uid 1000 sokil \
    && mkdir -p /tmp/ultralytics /tmp/matplotlib /cache /data \
    && chmod 1777 /tmp/ultralytics /tmp/matplotlib /cache \
    && chown sokil:sokil /data

COPY --from=builder /opt/venv /opt/venv

# Needs no device, so a build on any arm64 machine checks that the project, the
# venv's packages and Debian's HailoRT load together.
RUN python -c "import sokil, hailo_platform, numpy; print('hailo_platform with numpy', numpy.__version__)"

# Bind-mount the working directory here, so relative paths on the command line
# resolve against the host directory the user ran from.
WORKDIR /data
USER sokil

ENTRYPOINT ["sokil"]
CMD ["--help"]
