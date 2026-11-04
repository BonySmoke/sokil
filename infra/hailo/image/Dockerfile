# Compiles YOLO weights to Hailo HEF. linux/amd64 only: the Dataflow Compiler
# ships no other build.
#
# The base is Hailo's AI Software Suite, which bundles the DFC with the CUDA and
# cuDNN it was built against. It is not on a public registry: download the
# tarball from the Hailo Developer Zone, `docker load` it, and pass the tag it
# loads as HAILO_SUITE_IMAGE. See docs/accelerators.md.
#
# Its own context (infra/hailo/image), rather than the repo root, so a build sends
# just these files and not the datasets: those are mounted at run time.
# requirements.txt is not committed: hailo.sh writes it from uv.lock first.

ARG HAILO_SUITE_IMAGE=hailo8_ai_sw_suite_2026-07:latest

FROM ${HAILO_SUITE_IMAGE}

# The suite interpreter, which can import the DFC.
ARG HAILO_PYTHON=python3
ENV HAILO_PYTHON=${HAILO_PYTHON}

# ultralytics and onnxslim at the versions uv.lock pins, fitted around the
# suite's own packages.
COPY --chmod=0755 install.sh /opt/sokil/
COPY requirements.txt /opt/sokil/
RUN /opt/sokil/install.sh /opt/sokil/requirements.txt

# Writable places for the host's uid, and no installing mid-export; the DFC keeps
# its logs in the working directory, which is the output.
ENV HOME=/tmp/home \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_AUTOINSTALL=false \
    PYTHONUNBUFFERED=1

COPY --chmod=0755 entrypoint.sh /opt/sokil/
COPY export_hailo.py /opt/sokil/

# Mount points: the repo read-only at /data, the results at /out.
WORKDIR /out

ENTRYPOINT ["/opt/sokil/entrypoint.sh"]
CMD ["--help"]
