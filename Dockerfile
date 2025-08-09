# CPU-only image for the CLI

FROM python:3.14-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Dependencies first, without the project itself, so a source change does not
# invalidate the cached dependency layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-default-groups --no-install-project

# Then the project. --no-editable copies it into the venv rather than linking
# back to /app, which does not exist in the runtime stage.
COPY sokil/ /app/sokil/
COPY pyproject.toml uv.lock README.md /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable


FROM python:3.14-slim-bookworm AS runtime

# opencv-python (as opposed to opencv-python-headless) links against the system
# GL and glib stacks even when it never opens a window.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 sokil

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SOKIL_CACHE_DIR=/cache \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# Both libraries write a config file on import and warn (ultralytics) or fail
# (matplotlib) if the directory isn't writable. Pre-created world-writable so
# the image works under any --user, not just the sokil uid.
RUN mkdir -p /tmp/ultralytics /tmp/matplotlib \
    && chmod 1777 /tmp/ultralytics /tmp/matplotlib

# The weights cache. Mount a volume here to keep downloads between runs; without
# one the image still works and simply fetches again. World-writable for the
# same reason as above: the image runs under whatever --user the host passes.
RUN mkdir -p /cache && chmod 1777 /cache

COPY --from=builder /opt/venv /opt/venv

# Bind-mount the working directory here, so relative paths on the command line
# resolve against the host directory the user ran from.
WORKDIR /data
RUN chown sokil:sokil /data

USER sokil

ENTRYPOINT ["sokil"]
CMD ["--help"]
