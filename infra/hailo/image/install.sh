#!/usr/bin/env bash
#
# Installs the export's packages into the AI Software Suite's interpreter, at
# build time. Takes the requirements file hailo.sh writes from uv.lock's hailo
# group: ultralytics and onnxslim, pinned.
#
# The ultralytics exporter drives the DFC from the same process, so both have to
# be importable from one interpreter, the suite's. Torch, torchvision and
# onnxruntime come from the suite, which ships them for the DFC; ultralytics
# only uses torch to load the .pt and trace the ONNX graph. onnxslim is what the
# ONNX simplify step needs; ultralytics would otherwise pip-install it
# mid-export.

set -euo pipefail

requirements="$1"
python="$HAILO_PYTHON"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# The suite installs the DFC in a virtualenv; if that is not first on PATH,
# HAILO_PYTHON has to name its interpreter.
"$python" -c "import hailo_sdk_client" || {
    echo "HAILO_PYTHON=$python cannot import hailo_sdk_client;" \
        "pass --build-arg HAILO_PYTHON=<the suite virtualenv's python>" >&2
    exit 1
}

# Everything the suite ships is pinned as a constraint: those are the versions
# the DFC was validated against, and pip would otherwise upgrade shared ones like
# numpy to whatever ultralytics prefers. A clash fails the build here instead of
# the export later.
#
# cloudpickle is the one exception: ultralytics needs >=3.1.1, the suite has
# 2.2.0, and only tensordict and tensorflow-probability use it, with no upper
# bound.
"$python" -m pip list --format=freeze | grep -v '^cloudpickle==' >"$work/constraints.txt"

# The suite fails `pip check` as shipped (the DFC pins versions its own suite
# does not install), so only the conflicts this install adds fail the build.
"$python" -m pip check >"$work/check-before.txt" || true

"$python" -m pip install --no-cache-dir -c "$work/constraints.txt" -r "$requirements"

"$python" -m pip check >"$work/check-after.txt" || true
if grep -vxFf "$work/check-before.txt" "$work/check-after.txt"; then
    echo "installing $(tr '\n' ' ' <"$requirements")broke the dependencies above" >&2
    exit 1
fi

"$python" -c "import hailo_sdk_client, ultralytics, onnxslim"
