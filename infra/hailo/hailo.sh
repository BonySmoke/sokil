#!/usr/bin/env bash
#
# Compiling YOLO weights to Hailo HEF, locally or on a cloud GPU. Called by the
# targets in hailo.mk; see docs/accelerators.md.
#
# Either way a model compiles into training/<model>/hailo/, and only once it
# has succeeded: a failed compile leaves the previous HEF where it was and its
# own logs under training/hailo-<local|cloud>/<run>/.

set -euo pipefail

HAILO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HAILO_ROOT/../.." && pwd)"
HAILO_CLOUD="$HAILO_ROOT/cloud"

die() {
    echo "$@" >&2
    exit 1
}

# <model>=<weights> pairs into the models and pairs arrays of the caller. One
# with no weights is skipped, so the Makefile can pass every model
# unconditionally.
hailo_parse_pairs() {
    local pair model weights
    models=()
    pairs=()
    for pair in "$@"; do
        model="${pair%%=*}"
        weights="${pair#*=}"
        [ "$model" != "$pair" ] || die "expected <model>=<weights>, got '$pair'"
        [ -n "$weights" ] || continue
        [ -f "$weights" ] || die "missing $weights"
        models+=("$model")
        pairs+=("$model=$weights")
    done
    [ "${#models[@]}" -gt 0 ] || die "no weights given: set COURT_WEIGHTS and/or SHUTTLE_WEIGHTS"
}

# Where a model's HEF goes, as hailo-local and hailo-cloud both leave it.
hailo_destination() {
    echo "$REPO_ROOT/training/$1/hailo"
}

# The image's requirements.txt: the direct dependencies of uv.lock's hailo group,
# pinned, which install.sh fits around the suite's packages. Written before
# every build, local or on the VM, so the image follows uv lock --upgrade. The
# rest of the lock does not apply: it resolves for this project's Python, not
# the suite's.
hailo_requirements() {
    command -v uv >/dev/null 2>&1 || die "uv is required to read the hailo group's versions from uv.lock"

    # `uv tree` lists them as "├── ultralytics v8.4.149 (group: hailo)".
    uv tree --frozen --quiet --project "$REPO_ROOT" --only-group hailo --depth 1 \
        | awk '$NF == "hailo)" { print $2 "==" substr($3, 2) }' \
        >"$HAILO_ROOT/image/requirements.txt"
    [ -s "$HAILO_ROOT/image/requirements.txt" ] || die "no hailo group in $REPO_ROOT/uv.lock"
    sed 's/^/  /' "$HAILO_ROOT/image/requirements.txt"
}

# --------------------------------------------------------------------------
# local
# --------------------------------------------------------------------------

# Compile on this machine, in the image make hailo-build made. One model after
# another, as on the VM; a failure moves on to the next model.
#
# The repository is mounted read-only, so the weights must be inside it. Each
# model compiles into training/hailo-local/<run>/<model>/ and is copied into
# place once it succeeds, since the exporter empties its output directory
# before it starts.
hailo_local() {
    local image="${HAILO_IMAGE:-sokil-hailo-compiler:dev}"
    docker image inspect "$image" >/dev/null 2>&1 \
        || die "no $image image: build it first with make hailo-build (see docs/accelerators.md)"

    local models pairs
    hailo_parse_pairs "$@"

    local run_id staging
    run_id="$(date -u +%Y%m%d-%H%M%S)"
    staging="$REPO_ROOT/training/hailo-local/$run_id"

    local run_args=(--rm --user "$(id -u):$(id -g)" -v "$REPO_ROOT":/data:ro)
    [ ! -t 1 ] || run_args+=(-t)
    # --gpus fails outright without the NVIDIA container toolkit, so it is only
    # passed when hailo.mk found a GPU.
    [ -z "${HAILO_GPUS:-}" ] || run_args+=(--gpus "$HAILO_GPUS")

    # Expanded below as ${extra_args[@]+...}: macOS's bash 3.2 calls an empty
    # array unbound under set -u.
    local extra_args
    read -r -a extra_args <<<"${HAILO_ARGS:-}"

    local pair model weights relative results failed=0 summary=()
    for pair in "${pairs[@]}"; do
        model="${pair%%=*}"
        weights="$(cd "$(dirname "${pair#*=}")" && pwd)/$(basename "${pair#*=}")"
        relative="${weights#"$REPO_ROOT/"}"
        [ "$relative" != "$weights" ] || die "$weights is outside the repository, which is all the container sees"
        [ -f "$REPO_ROOT/training/$model/datasets/data.yaml" ] \
            || die "missing training/$model/datasets/data.yaml: export the $model dataset first"

        echo "=== $(date +%H:%M:%S) compiling $model"
        results="$staging/$model"
        mkdir -p "$results"
        if docker run "${run_args[@]}" -v "$results":/out "$image" \
            --weights "/data/$relative" --data "/data/training/$model/datasets/data.yaml" \
            --arch "${HAILO_ARCH:-hailo8}" ${extra_args[@]+"${extra_args[@]}"}; then
            mkdir -p "$(hailo_destination "$model")"
            cp -R "$results/." "$(hailo_destination "$model")/"
            rm -rf "$results"
            summary+=("$model: ok, in training/$model/hailo")
        else
            summary+=("$model: failed, logs in ${results#"$REPO_ROOT/"}")
            failed=1
        fi
    done
    rmdir "$staging" 2>/dev/null || true
    rmdir "$REPO_ROOT/training/hailo-local" 2>/dev/null || true

    echo
    printf '%s\n' "${summary[@]}"
    return "$failed"
}

# --------------------------------------------------------------------------
# cloud
# --------------------------------------------------------------------------

# Compiling on an AWS spot GPU. cloud/persistent is applied once by hand; these
# commands upload a run, apply cloud/runner for it, wait for the VM to finish
# and terminate itself, download the results and destroy the stack.

hailo_require() {
    local tool
    for tool in aws tofu; do
        command -v "$tool" >/dev/null 2>&1 || die "$tool is required: brew install awscli opentofu"
    done
}

# One output of a stack, or nothing when its state has none. `output -raw` alone
# will not do: with no outputs it prints a warning to stdout and exits 0, so
# the warning would be read as the value. `-json` prints just {} then.
hailo_output() {
    local dir="$1" name="$2"

    [ "$(tofu -chdir="$dir" output -json 2>/dev/null)" != "{}" ] || return 0
    tofu -chdir="$dir" output -raw "$name" 2>/dev/null || true
}

# An output of the persistent stack, which must have been applied first.
hailo_persistent() {
    local value
    value="$(hailo_output "$HAILO_CLOUD/persistent" "$1")"
    [ -n "$value" ] || die "no '$1' output: run make hailo-cloud-setup first (see docs/accelerators.md)"
    echo "$value"
}

hailo_runner() {
    [ -d "$HAILO_CLOUD/runner/.terraform" ] \
        || tofu -chdir="$HAILO_CLOUD/runner" init -input=false >&2
    tofu -chdir="$HAILO_CLOUD/runner" "$@"
}

# An output of the runner stack, or nothing when no run is active.
hailo_runner_output() {
    [ -d "$HAILO_CLOUD/runner/.terraform" ] \
        || tofu -chdir="$HAILO_CLOUD/runner" init -input=false >&2
    hailo_output "$HAILO_CLOUD/runner" "$1"
}

hailo_setup() {
    hailo_require
    bucket="$(hailo_persistent bucket)"
    AWS_REGION="$(hailo_persistent region)"
    export AWS_REGION
}

# The one suite tarball in the bucket.
hailo_suite_key() {
    local keys
    keys="$(aws s3api list-objects-v2 --bucket "$bucket" --prefix suite/ \
        --query 'Contents[].Key' --output text)"
    [ -n "$keys" ] && [ "$keys" != "None" ] || die "no suite in s3://$bucket/suite/: run make hailo-cloud-suite first"
    [ "$(wc -w <<<"$keys")" -eq 1 ] || die "more than one suite in s3://$bucket/suite/: $keys"
    echo "$keys"
}

# Upload the AI Software Suite tarball, replacing any earlier one: the bucket
# keeps exactly one, which every run loads.
hailo_cloud_suite() {
    local tarball="${1:-}"

    [ -n "$tarball" ] || die "usage: hailo.sh cloud-suite <suite .tar.gz>"
    # zsh leaves a ~ after `=` unexpanded, so `make … SUITE=~/x` passes it on
    # literally, and the Makefile's quoting keeps the shell here from expanding it.
    tarball="${tarball/#\~/$HOME}"
    [ -f "$tarball" ] || die "missing $tarball"
    hailo_setup

    local key
    key="suite/$(basename "$tarball")"
    aws s3 cp "$tarball" "s3://$bucket/$key"

    local old
    for old in $(aws s3api list-objects-v2 --bucket "$bucket" --prefix suite/ \
        --query 'Contents[].Key' --output text); do
        [ "$old" = "$key" ] || aws s3 rm "s3://$bucket/$old"
    done
}

# Upload one model's weights and the train split of its dataset, which is all
# calibration reads, as a single tar: S3 bills per request, and a dataset is
# thousands of small files. JPEGs do not compress, so it is not gzipped.
#
# The tar holds <weights>.pt and datasets/{data.yaml,images/<train>,labels/<train>},
# the layout bootstrap.sh mounts into the export container.
hailo_upload_model() {
    local run="$1" model="$2" weights="$3"
    local dataset="$REPO_ROOT/training/$model/datasets"

    [ -f "$dataset/data.yaml" ] || die "missing $dataset/data.yaml: export the $model dataset first"

    local train labels
    train="$(sed -n 's/^train: *//p' "$dataset/data.yaml" | tr -d "\"'")"
    # ultralytics finds an image's labels by swapping images/ for labels/.
    [ "${train#images/}" != "$train" ] || die "$dataset/data.yaml: expected train under images/, got '$train'"
    labels="labels/${train#images/}"

    [ -d "$dataset/$train" ] || die "missing $dataset/$train"
    [ -d "$dataset/$labels" ] || die "missing $dataset/$labels"

    # tar resolves each -C against the one before it, so both must be absolute.
    weights="$(cd "$(dirname "$weights")" && pwd)/$(basename "$weights")"

    # Without these, macOS tar adds ._ AppleDouble files and extended
    # attributes, which GNU tar on the VM would warn about.
    COPYFILE_DISABLE=1 tar -cf - --no-mac-metadata --no-xattrs --exclude .DS_Store \
        -C "$(dirname "$weights")" "$(basename "$weights")" \
        -C "$dataset/.." datasets/data.yaml "datasets/$train" "datasets/$labels" \
        | aws s3 cp - "$run/inputs/$model.tar"
}

# Start a run and see it through.
hailo_cloud_run() {
    hailo_setup

    local active
    active="$(hailo_runner_output run_id)"
    [ -z "$active" ] || die "run $active is still active: make hailo-cloud-fetch, or make hailo-cloud-destroy to abandon it"

    local models pairs
    hailo_parse_pairs "$@"

    local suite_key run_id run pair
    suite_key="$(hailo_suite_key)"
    run_id="$(date -u +%Y%m%d-%H%M%S)"
    run="s3://$bucket/runs/$run_id"

    echo "uploading run $run_id: ${models[*]}, with"
    hailo_requirements
    aws s3 sync --no-progress --exclude '*.pyc' --exclude '__pycache__/*' \
        "$HAILO_ROOT/image" "$run/code"

    for pair in "${pairs[@]}"; do
        hailo_upload_model "$run" "${pair%%=*}" "${pair#*=}"
    done
    
    printf 'HAILO_MODELS=%q\nHAILO_ARCH=%q\nHAILO_PYTHON=%q\nHAILO_ARGS=%q\n' \
        "${models[*]}" "${HAILO_ARCH:-hailo8}" "${HAILO_PYTHON:-python3}" "${HAILO_ARGS:-}" \
        | aws s3 cp - "$run/run.env"

    # The fleet picks a zone with spot capacity itself, and fails the apply
    # when no zone has any.
    if ! hailo_runner apply -auto-approve -input=false \
        -var "run_id=$run_id" -var "suite_key=$suite_key" \
        -var "max_minutes=${HAILO_CLOUD_MAX_MINUTES:-90}"; then
        # Nothing ran, so nothing is worth keeping.
        echo "starting the VM failed; cleaning up" >&2
        hailo_cloud_destroy
        aws s3 rm --recursive --quiet "$run"
        die "no run started"
    fi

    hailo_cloud_fetch
}

# Wait for the active run's VM to finish, download the results and destroy the
# stack. Safe to re-run: after a Ctrl-C, or with the laptop asleep, the VM
# carries on and this picks up where it is.
hailo_cloud_fetch() {
    hailo_setup

    local run_id instance_id
    run_id="$(hailo_runner_output run_id)"
    instance_id="$(hailo_runner_output instance_id)"
    [ -n "$run_id" ] || die "no active run"

    local run="s3://$bucket/runs/$run_id" state started="$SECONDS"
    echo "waiting for run $run_id on $instance_id (Ctrl-C is safe; make hailo-cloud-fetch resumes)"
    until aws s3api head-object --bucket "$bucket" --key "runs/$run_id/done" >/dev/null 2>&1; do
        state="$(aws ec2 describe-instances --instance-ids "$instance_id" \
            --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo gone)"
        case "$state" in
            shutting-down|terminated|gone)
                # A spot interruption, or a VM that died before its trap ran.
                echo "the VM ended without finishing ($state)" >&2
                break ;;
        esac
        printf '  %3d min, VM %s, done: %s\n' $(((SECONDS - started) / 60)) "$state" \
            "$(aws s3 ls "$run/status/" 2>/dev/null | awk '{printf "%s ", $4}')"
        sleep 60
    done

    local diagnostics="$REPO_ROOT/training/hailo-cloud/$run_id" failed=0 status model
    for model in $(aws s3 ls "$run/inputs/" | awk '{print $4}' | sed 's/\.tar$//'); do
        status="$(aws s3 cp "$run/status/$model" - 2>/dev/null || echo missing)"
        if [ "$status" = ok ]; then
            # --exact-timestamps, or a file the same size as an older local copy,
            # like metadata.yaml with only its date changed, would be skipped.
            aws s3 sync --no-progress --exact-timestamps "$run/results/$model" "$(hailo_destination "$model")"
            echo "$model: ok, in training/$model/hailo"
        else
            aws s3 sync --no-progress "$run/results/$model" "$diagnostics/$model"
            echo "$model: $status, logs in ${diagnostics#"$REPO_ROOT/"}/$model" >&2
            failed=1
        fi
    done
    aws s3 sync --no-progress "$run/logs" "$diagnostics/logs"
    echo "VM log in ${diagnostics#"$REPO_ROOT/"}/logs"

    aws s3 rm --recursive --quiet "$run"
    hailo_cloud_destroy

    return "$failed"
}

# Destroy the runner stack, terminating the VM if it is still up. The run's
# files stay in the bucket until they expire.
hailo_cloud_destroy() {
    hailo_require

    local run_id suite_key
    run_id="$(hailo_runner_output run_id)"
    suite_key="$(hailo_runner_output suite_key)"
    [ -n "$run_id" ] || { echo "no active run"; return 0; }

    hailo_runner destroy -auto-approve -input=false \
        -var "run_id=$run_id" -var "suite_key=$suite_key"
}

# The running VM, for a Session Manager session to it. Run hailo_setup first:
# this runs in $(...), where its exports would not reach the caller.
hailo_session_target() {
    command -v session-manager-plugin >/dev/null 2>&1 \
        || die "the Session Manager plugin is required: brew install --cask session-manager-plugin"

    local instance_id
    instance_id="$(hailo_runner_output instance_id)"
    [ -n "$instance_id" ] || die "no active run; a finished run's log is under training/hailo-cloud/<run>/logs/"
    echo "$instance_id"
}

# A shell on the running VM, for watching or debugging a run.
hailo_cloud_shell() {
    hailo_setup
    local instance_id
    instance_id="$(hailo_session_target)"
    aws ssm start-session --target "$instance_id"
}

# Follow the running VM's log, which is bootstrap.sh's output, from the start.
# Ctrl-C stops following and leaves the run alone. The log belongs to root, and
# Session Manager's ssm-user has passwordless sudo.
hailo_cloud_logs() {
    hailo_setup
    local instance_id
    instance_id="$(hailo_session_target)"
    aws ssm start-session --target "$instance_id" \
        --document-name AWS-StartInteractiveCommand \
        --parameters 'command=["sudo tail -n +1 -F /var/log/cloud-init-output.log"]'
}

# Create or update the persistent stack. tofu shows the plan and asks before
# applying; the region comes from cloud/persistent/terraform.tfvars, if there is
# one, so every apply sees the same.
hailo_cloud_setup() {
    hailo_require
    tofu -chdir="$HAILO_CLOUD/persistent" init -input=false
    tofu -chdir="$HAILO_CLOUD/persistent" apply
}

# Remove everything: the VM if a run is up, the suite and the bucket, the role
# and the security group. The bucket has to be empty before tofu can delete it,
# so this asks first, as emptying it cannot be undone.
hailo_cloud_teardown() {
    hailo_setup

    echo "This deletes s3://$bucket, the suite in it, and the VM's role and security group."
    local answer
    read -r -p "Type the bucket name to confirm: " answer
    [ "$answer" = "$bucket" ] || die "not confirmed; nothing deleted"

    hailo_cloud_destroy
    aws s3 rm --recursive --quiet "s3://$bucket"
    tofu -chdir="$HAILO_CLOUD/persistent" destroy -auto-approve -input=false
}

# --------------------------------------------------------------------------

usage() {
    cat <<'USAGE'
usage: infra/hailo/hailo.sh <command> [args...]

  requirements                    write the image's requirements.txt from uv.lock
  local <model>=<path>...         compile on this machine, in the hailo-build image
  cloud-setup                     create or update the bucket, role and security group
  cloud-teardown                  delete the bucket, suite, role and security group
  cloud-suite <tarball>           upload the Hailo suite for cloud compiles
  cloud-run <model>=<path>...     compile on an AWS spot GPU and fetch the HEFs
  cloud-fetch                     wait for the active run, fetch, destroy the VM
  cloud-destroy                   destroy the VM without fetching
  cloud-shell                     open a Session Manager shell on the VM
  cloud-logs                      follow the running VM's log
USAGE
}

command="${1:-}"
shift || true

case "$command" in
    requirements)   hailo_requirements ;;
    local)          hailo_local "$@" ;;
    cloud-setup)    hailo_cloud_setup ;;
    cloud-teardown) hailo_cloud_teardown ;;
    cloud-suite)    hailo_cloud_suite "$@" ;;
    cloud-run)      hailo_cloud_run "$@" ;;
    cloud-fetch)    hailo_cloud_fetch ;;
    cloud-destroy)  hailo_cloud_destroy ;;
    cloud-shell)    hailo_cloud_shell ;;
    cloud-logs)     hailo_cloud_logs ;;
    ""|-h|--help)   usage ;;
    *)              usage >&2; die "unknown command: $command" ;;
esac
