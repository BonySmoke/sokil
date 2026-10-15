# Compiles YOLO weights to Hailo HEF. linux/amd64 only: the Dataflow Compiler
# ships no other build.
#
# The base is Hailo's AI Software Suite, which bundles the DFC with the CUDA and
# cuDNN it was built against. It is not on a public registry: download the
# tarball from the Hailo Developer Zone, `docker load` it, and pass the tag it
# loads as HAILO_SUITE_IMAGE. See docs/training.md.
#
# Its own context (docker/hailo), rather than the repo root, so a build sends
# just the export script and not the datasets: those are mounted at run time.

ARG HAILO_SUITE_IMAGE=hailo8_ai_sw_suite_2025-10:1

FROM ${HAILO_SUITE_IMAGE}

# The interpreter that can import the DFC. The suite installs it in a virtualenv;
# if that is not first on PATH, the check below fails and names this argument.
ARG HAILO_PYTHON=python3
ENV HAILO_PYTHON=${HAILO_PYTHON}

RUN "$HAILO_PYTHON" -c "import hailo_sdk_client" \
    || { echo "HAILO_PYTHON=$HAILO_PYTHON cannot import hailo_sdk_client;" \
              "pass --build-arg HAILO_PYTHON=<the suite virtualenv's python>" >&2; exit 1; }

# The ultralytics exporter drives the DFC from the same process, so both have to
# be importable from one interpreter. Everything already installed is pinned as
# a constraint: the suite's packages are the versions the DFC was validated
# against, and pip would otherwise upgrade shared ones like numpy to whatever
# ultralytics prefers. A clash fails the build here instead of the export later.
#
# Torch is the CPU build: it only loads the .pt and traces the ONNX graph. The
# GPU work is the DFC's, through the suite's own TensorFlow.
#
# onnxruntime and onnxslim are what the ONNX simplify step needs; ultralytics
# would otherwise pip-install them mid-export. onnxruntime is left unpinned so
# pip picks a build for the suite's Python.
RUN "$HAILO_PYTHON" -m pip list --format=freeze > /tmp/suite-constraints.txt \
    && "$HAILO_PYTHON" -m pip install --no-cache-dir -c /tmp/suite-constraints.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0 torchvision==0.29.0 \
    && "$HAILO_PYTHON" -m pip install --no-cache-dir -c /tmp/suite-constraints.txt \
        ultralytics==8.4.149 onnxslim==0.1.98 onnxruntime \
    && "$HAILO_PYTHON" -m pip check \
    && "$HAILO_PYTHON" -c "import hailo_sdk_client, ultralytics" \
    && rm /tmp/suite-constraints.txt

# The container runs under the host's uid (see the Makefile), which has no home
# here and cannot write the suite user's. ultralytics writes a settings file on
# import; the DFC keeps its logs in the working directory, which is the output.
ENV HOME=/tmp/home \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_AUTOINSTALL=false \
    PYTHONUNBUFFERED=1

COPY export_hailo.py /opt/sokil/export_hailo.py

# Mount points: the repo read-only at /data, the results at /out.
WORKDIR /out

# Through a shell so the interpreter can come from HAILO_PYTHON; the trailing
# "--" becomes $0, leaving the docker run arguments as "$@".
ENTRYPOINT ["/bin/sh", "-c", "mkdir -p \"$HOME\" \"$YOLO_CONFIG_DIR\" && exec \"$HAILO_PYTHON\" /opt/sokil/export_hailo.py \"$@\"", "--"]
CMD ["--help"]
