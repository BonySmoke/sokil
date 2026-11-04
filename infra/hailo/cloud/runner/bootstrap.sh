#!/usr/bin/env bash
#
# A compile run, start to finish, on the VM: runs once at boot as root, from
# user data. main.tf puts these assignments ahead of it:
#
#   HAILO_BUCKET, HAILO_RUN_ID, HAILO_SUITE_KEY, HAILO_MAX_MINUTES, AWS_DEFAULT_REGION
#
# and infra/hailo/hailo.sh uploads the run's run.env, code and inputs before the VM exists.
# Whatever happens, the script ends by writing runs/<id>/done and powering off,
# which terminates the instance.

set -euo pipefail

# The kill switch comes first, so even a hang anywhere below ends the run.
shutdown -P "+$HAILO_MAX_MINUTES"

run="s3://$HAILO_BUCKET/runs/$HAILO_RUN_ID"

# cloud-init captures this script's output in its own log, which is uploaded
# alongside the results for debugging a failed run from the laptop.
# shellcheck disable=SC2329  # invoked by the trap
finish() {
    local status=$?
    echo "bootstrap finished with status $status"
    aws s3 cp --no-progress /var/log/cloud-init-output.log "$run/logs/cloud-init-output.log" || true
    echo "$status" | aws s3 cp - "$run/done" || true
    shutdown -P now
}
trap finish EXIT

step() {
    echo
    echo "=== $(date -u +%H:%M:%S) $*"
}

# --- Scratch space -------------------------------------------------------------

# g4dn's local NVMe, which the DLAMI mounts here, is faster than the root EBS
# volume and costs nothing. The suite unpacks to tens of GB, more than the root
# volume holds, so the image stores go there too.
#
# Both of them, bind-mounted: this Docker keeps images in containerd's store,
# under /var/lib/containerd, where moving Docker's data-root would not reach.
# A bind mount covers either store without depending on which one is in use.
scratch=/opt/dlami/nvme
if mountpoint -q "$scratch"; then
    step "moving the image stores to $scratch"
    systemctl stop docker.socket docker containerd
    for store in /var/lib/docker /var/lib/containerd; do
        mkdir -p "$store" "$scratch$store"
        cp -a "$store/." "$scratch$store/"
        mount --bind "$scratch$store" "$store"
    done
    systemctl start containerd docker
    df -h / "$scratch"
else
    echo "no local NVMe at $scratch, using the root volume"
    scratch=/var/tmp/sokil
fi
work="$scratch/work"
mkdir -p "$work"

# --- Inputs --------------------------------------------------------------------

step "downloading the run"
aws s3 cp --no-progress "$run/run.env" "$work/run.env"
aws s3 sync --no-progress "$run/code" "$work/code"

# HAILO_MODELS, HAILO_ARCH, HAILO_PYTHON, HAILO_ARGS, as hailo.sh wrote them.
# shellcheck source=/dev/null
. "$work/run.env"

# One tar per model: the weights and datasets/, as hailo_upload_model packs it.
for model in $HAILO_MODELS; do
    mkdir -p "$work/inputs/$model"
    aws s3 cp --no-progress "$run/inputs/$model.tar" - | tar -xf - -C "$work/inputs/$model"
done

# Streamed straight into docker load, which reads the gzip itself, so the
# tarball never lands on disk.
step "loading $HAILO_SUITE_KEY"
loaded="$(aws s3 cp --no-progress "s3://$HAILO_BUCKET/$HAILO_SUITE_KEY" - | docker load 2>&1)"
echo "$loaded"
# docker load can name the image and exit 0 even when unpacking it failed, as
# when the disk fills; the build would then fail later and less clearly.
! grep -q '^Error' <<<"$loaded" || { echo "docker load failed" >&2; exit 1; }
suite_image="$(sed -n 's/^Loaded image: //p' <<<"$loaded" | head -n 1)"
[ -n "$suite_image" ] || { echo "docker load named no image" >&2; exit 1; }

step "building the export image on $suite_image"
docker build \
    --build-arg HAILO_SUITE_IMAGE="$suite_image" \
    --build-arg HAILO_PYTHON="$HAILO_PYTHON" \
    -t sokil-hailo-compiler "$work/code"

# --- GPU -----------------------------------------------------------------------

# Without a usable GPU the DFC quietly optimizes on the CPU, which for the
# court model outlasts the kill switch. Better to fail here in minutes.
step "checking the GPU"
nvidia-smi
docker run --rm --gpus all --entrypoint /bin/sh sokil-hailo-compiler -c '
    "$HAILO_PYTHON" -c "
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices(\"GPU\")
print(\"GPUs visible to TensorFlow:\", gpus)
sys.exit(not gpus)
"'

# --- Exports -------------------------------------------------------------------

# One after another: the court export alone filled 16 GB of RAM on a laptop,
# and two TensorFlow processes would contend for the one GPU's memory. A
# failure moves on to the next model; each one's status tells hailo.sh which
# results to keep.
read -r -a extra_args <<<"$HAILO_ARGS"
failed=0
for model in $HAILO_MODELS; do
    step "exporting $model"
    inputs="$work/inputs/$model"
    results="$work/results/$model"
    mkdir -p "$results"
    weights="$(find "$inputs" -maxdepth 1 -name '*.pt' -printf '%f\n' | head -n 1)"

    # --ipc=host as Hailo's own docker_run script does, for TensorFlow's
    # shared memory. Root, unlike `make hailo-local`, as nothing here belongs
    # to a user.
    if docker run --rm --gpus all --ipc=host --user root \
        -v "$inputs":/data:ro -v "$results":/out sokil-hailo-compiler \
        --weights "/data/$weights" --data /data/datasets/data.yaml \
        --arch "$HAILO_ARCH" "${extra_args[@]}"; then
        status=ok
    else
        status=failed
        failed=1
    fi

    aws s3 sync --no-progress "$results" "$run/results/$model"
    echo "$status" | aws s3 cp - "$run/status/$model"
    echo "$model: $status"
done

exit "$failed"
