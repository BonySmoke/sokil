#!/bin/sh
#
# Runs export_hailo.py with the docker run arguments, through the interpreter
# HAILO_PYTHON names.
#
# The container runs under the host's uid (see hailo.sh), which has no home here
# and cannot write the suite user's; ultralytics writes a settings file on
# import, so both directories are made on the way in.

set -e

mkdir -p "$HOME" "$YOLO_CONFIG_DIR"
exec "$HAILO_PYTHON" /opt/sokil/export_hailo.py "$@"
