#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$REPO_ROOT/sokil/models.yaml"

die() {
    echo "$@" >&2
    exit 1
}

# --------------------------------------------------------------------------
# make help
# --------------------------------------------------------------------------

# Render the self-documenting target list: every target whose line carries a
# `## ` comment, with that comment as its description.
make_help() {
    [ "$#" -ge 1 ] || die "usage: helper.sh make-help <makefile>..."

    grep -hE '^[a-zA-Z_-]+:.*?## ' "$@" \
        | awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $1, $2}'
}

# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

# The published file name for one model, read from the manifest so this and the
# code cannot disagree about what a release should contain.
asset_name() {
    local model="$1"

    command -v uv >/dev/null 2>&1 || die "uv is required to read $MANIFEST"

    uv run --quiet --project "$REPO_ROOT" python -c "
import sys, yaml
manifest = yaml.safe_load(open('$MANIFEST'))
entry = manifest['models'].get('$model')
if entry is None:
    sys.exit(\"no model '$model' in the manifest\")
print(entry['asset'])
"
}

# Publish trained weights as a GitHub release.
#
# A release asset is named after the file uploaded, and a training run leaves
# its output as best.pt, so the weights are copied to their published names in a
# scratch directory. Nothing is staged into the repository.
models_release() {
    local tag="${1:-}"
    shift || true

    [ -n "$tag" ] || die "usage: helper.sh models-release <tag> <model>=<weights>..."
    [ "$#" -gt 0 ] || die "no weights given, e.g. shuttle=runs/detect/train8/weights/best.pt"

    command -v gh >/dev/null 2>&1 || die "gh is required to publish a release"

    local staging
    staging="$(mktemp -d)"
    # shellcheck disable=SC2064  # expand staging now, not when the trap fires
    trap "rm -rf '$staging'" EXIT

    local uploads=() pin_args=()
    local pair model weights asset
    for pair in "$@"; do
        model="${pair%%=*}"
        weights="${pair#*=}"

        [ "$model" != "$pair" ] || die "expected <model>=<weights>, got '$pair'"
        [ -n "$weights" ] || die "no weights given for '$model'"
        [ -f "$weights" ] || die "missing $weights"

        asset="$(asset_name "$model")"
        cp "$weights" "$staging/$asset"
        uploads+=("$staging/$asset")
        pin_args+=("--weights $model=$weights")
    done

    gh release create "$tag" \
        --title "Weights $tag" \
        --notes "Trained weights for the shuttle detector and court segmenter." \
        "${uploads[@]}"

    # Publishing alone changes nothing for users: the manifest is what the code
    # reads, so say so rather than leaving the release half-applied.
    echo
    echo "Published. Now pin them in sokil/models.yaml:"
    echo "  uv run sokil models pin --tag $tag \\"
    printf '    %s \\\n' "${pin_args[@]::${#pin_args[@]}-1}"
    printf '    %s\n' "${pin_args[${#pin_args[@]}-1]}"
}

# --------------------------------------------------------------------------
# cvat
# --------------------------------------------------------------------------

# CVAT is configured by its own env file, separate from the project's.
cvat_env() {
    local env_file="${1:-}"

    [ -n "$env_file" ] || die "usage: helper.sh cvat-env <env-file>"
    [ -f "$env_file" ] || die "$env_file not found — copy infra/.env.example to $env_file first"
}

# What a freshly started CVAT is reachable at, and where its file share lives.
cvat_up_notice() {
    local env_file="${1:-}" share_path="${2:-}"

    echo "CVAT UI on port 8080, at the CVAT_HOST set in $env_file"

    if [ -n "$share_path" ]; then
        echo "File share: $share_path"
    else
        echo "File share: Docker-managed volume (set CVAT_SHARE_PATH in $env_file to bind a host directory)"
    fi
}

# --------------------------------------------------------------------------

usage() {
    cat <<'USAGE'
usage: ./helper.sh <command> [args...]

  make-help <makefile>...              render the Makefile target list
  models-release <tag> <model>=<path>  publish trained weights as a release
  cvat-env <env-file>                  check the CVAT env file exists
  cvat-up-notice <env-file> [share]    report where CVAT and its share are
USAGE
}

command="${1:-}"
shift || true

case "$command" in
    make-help)       make_help "$@" ;;
    models-release)  models_release "$@" ;;
    cvat-env)        cvat_env "$@" ;;
    cvat-up-notice)  cvat_up_notice "$@" ;;
    ""|-h|--help)    usage ;;
    *)               usage >&2; die "unknown command: $command" ;;
esac
