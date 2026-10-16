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

ARG HAILO_SUITE_IMAGE=hailo8_ai_sw_suite_2026-07:latest

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
# Torch, torchvision and onnxruntime come from the suite, which ships them for
# the DFC; ultralytics only uses torch to load the .pt and trace the ONNX graph.
# onnxslim is what the ONNX simplify step needs; ultralytics would otherwise
# pip-install it mid-export.
#
# cloudpickle is the one exception to the pins: ultralytics needs >=3.1.1, the
# suite has 2.2.0, and only tensordict and tensorflow-probability use it, with
# no upper bound.
#
# The suite fails `pip check` as shipped (the DFC pins versions its own suite
# does not install), so the check only fails on conflicts this step adds.
RUN "$HAILO_PYTHON" -m pip list --format=freeze | grep -v '^cloudpickle==' > /tmp/suite-constraints.txt \
    && { "$HAILO_PYTHON" -m pip check > /tmp/check-before.txt || true; } \
    && "$HAILO_PYTHON" -m pip install --no-cache-dir -c /tmp/suite-constraints.txt \
        ultralytics==8.4.149 onnxslim==0.1.98 \
    && { "$HAILO_PYTHON" -m pip check > /tmp/check-after.txt || true; } \
    && if grep -vxFf /tmp/check-before.txt /tmp/check-after.txt; then \
           echo "installing ultralytics broke the dependencies above" >&2; exit 1; \
       fi \
    && "$HAILO_PYTHON" -c "import hailo_sdk_client, ultralytics, onnxslim" \
    && rm /tmp/suite-constraints.txt /tmp/check-before.txt /tmp/check-after.txt

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
