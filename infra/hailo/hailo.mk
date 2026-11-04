# Compiling YOLO weights to Hailo HEF, included by the root Makefile so these
# run from the repo root. As there, recipes stay one-liners: the shell lives in
# infra/hailo/hailo.sh. See docs/accelerators.md.
#
# What to compile comes from the root Makefile's SHUTTLE_WEIGHTS and
# COURT_WEIGHTS, relative to the repo root; set either or both.

hailo_sh = infra/hailo/hailo.sh

# --- Compile image -----------------------------------------------------------
# The tag `docker load` printed for the AI Software Suite tarball.
HAILO_SUITE_IMAGE ?= hailo8_ai_sw_suite_2026-07:latest
HAILO_IMAGE ?= sokil-hailo-compiler:dev
# Only needed when the suite's virtualenv is not first on its PATH; the build
# says so if it is not.
HAILO_PYTHON ?= python3

# --- Compile settings, local and cloud ---------------------------------------
HAILO_ARCH ?= hailo8
# Extra export_hailo.py flags, e.g. HAILO_ARGS="--calibration-images 256".
HAILO_ARGS ?=

# --- Local -------------------------------------------------------------------
# All GPUs when the host has an NVIDIA driver, none otherwise: --gpus fails
# outright without the NVIDIA container toolkit, and the DFC then optimizes on
# the CPU. Override either way, e.g. HAILO_GPUS= or HAILO_GPUS=device=0.
HAILO_GPUS ?= $(if $(shell command -v nvidia-smi),all)

# --- Cloud -------------------------------------------------------------------
# The suite tarball from the Hailo Developer Zone, for hailo-cloud-suite.
SUITE ?=
# Minutes after which the VM powers off and terminates, finished or not.
HAILO_CLOUD_MAX_MINUTES ?= 90

hailo_env = HAILO_ARCH="$(HAILO_ARCH)" HAILO_PYTHON="$(HAILO_PYTHON)" HAILO_ARGS="$(HAILO_ARGS)"

# Court first in both: the longer of the two, so a compile that fails late, or
# a run that hits the time limit, loses the quicker one to redo.
hailo_models = court="$(COURT_WEIGHTS)" shuttle="$(SHUTTLE_WEIGHTS)"

hailo-build:  ## Build the Hailo compile image for hailo-local (linux/amd64, needs the suite image loaded)
	@$(hailo_sh) requirements
	docker buildx build --platform linux/amd64 --load \
		--build-arg HAILO_SUITE_IMAGE=$(HAILO_SUITE_IMAGE) --build-arg HAILO_PYTHON=$(HAILO_PYTHON) \
		-t $(HAILO_IMAGE) infra/hailo/image

hailo-local:  ## Compile to HEF on this machine, e.g. make hailo-local COURT_WEIGHTS=… SHUTTLE_WEIGHTS=…
	@$(hailo_env) HAILO_IMAGE="$(HAILO_IMAGE)" HAILO_GPUS="$(HAILO_GPUS)" \
		$(hailo_sh) local $(hailo_models)

hailo-cloud-setup:  ## Create or update the cloud compile's bucket, role and security group (asks first)
	@$(hailo_sh) cloud-setup

hailo-cloud-teardown:  ## Delete the cloud compile's bucket, suite, role and security group (asks first)
	@$(hailo_sh) cloud-teardown

hailo-cloud-suite:  ## Upload the Hailo suite for cloud compiles, e.g. make hailo-cloud-suite SUITE=hailo8_ai_sw_suite_2026-07.tar.gz
	@$(hailo_sh) cloud-suite "$(SUITE)"

hailo-cloud:  ## Compile to HEF on an AWS spot GPU, e.g. make hailo-cloud COURT_WEIGHTS=… SHUTTLE_WEIGHTS=…
	@$(hailo_env) HAILO_CLOUD_MAX_MINUTES="$(HAILO_CLOUD_MAX_MINUTES)" \
		$(hailo_sh) cloud-run $(hailo_models)

hailo-cloud-fetch:  ## Wait for the cloud compile, download the HEFs and destroy the VM
	@$(hailo_sh) cloud-fetch

hailo-cloud-destroy:  ## Destroy the cloud compile VM without fetching its results
	@$(hailo_sh) cloud-destroy

hailo-cloud-shell:  ## Open a shell on the cloud compile VM (needs the Session Manager plugin)
	@$(hailo_sh) cloud-shell

hailo-cloud-logs:  ## Follow the cloud compile VM's log while it runs (needs the Session Manager plugin)
	@$(hailo_sh) cloud-logs
