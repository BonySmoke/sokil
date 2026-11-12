# Recipes stay one-liners: anything needing real shell lives in ./helper.sh,
# where it can be read, run by hand and checked by shellcheck.
COMPOSE ?= docker compose
CVAT_COMPOSE ?= infra/docker-compose.yaml
# compose resolves .env relative to the compose file, not the shell's cwd.
# Override to point somewhere else, e.g. make cvat-up CVAT_ENV=.env
CVAT_ENV ?= infra/.env
CVAT_SHARE_COMPOSE ?= infra/docker-compose.share.yaml

# Optional: the host directory CVAT exposes as its file share, needed only to
# annotate images already on the host. Left unset the share is a plain
# Docker-managed volume. Read from CVAT_ENV so it lives beside the other
# deployment settings, and overridable per invocation:
#   make cvat-up CVAT_SHARE_PATH=$(CURDIR)/images
CVAT_SHARE_PATH ?= $(shell sed -n 's/^CVAT_SHARE_PATH=//p' $(CVAT_ENV) 2>/dev/null | tail -1)
export CVAT_SHARE_PATH

# The overlay redefines the cvat_share volume as a bind, so it is only added
# when a path was given — an empty device would fail to start.
cvat_share_overlay = $(if $(strip $(CVAT_SHARE_PATH)),-f $(CVAT_SHARE_COMPOSE))

cvat = $(COMPOSE) -f $(CVAT_COMPOSE) $(cvat_share_overlay) --env-file $(CVAT_ENV)

# --- Application image -------------------------------------------------------
IMAGE ?= sokil:dev
PLATFORMS ?= linux/amd64,linux/arm64
# Dependency groups to add to an image, space separated, e.g.
# make build-hailo GROUPS=dataset HAILO_BUILD_IMAGE=sokil-hailo:eval for an
# image that can run `sokil evaluate`.
GROUPS ?=
groups_arg = $(if $(strip $(GROUPS)),--build-arg UV_GROUPS="$(GROUPS)")

# the release tag to publish to, e.g. make models-release MODELS_TAG=models-2026-09-06
MODELS_TAG ?=

# Required by models-release, hailo-local and hailo-cloud compile
SHUTTLE_WEIGHTS ?=
COURT_WEIGHTS ?=

# Weights are downloaded on first use and kept in a named volume, so a review
# needs nothing staged by hand and later runs do not re-download.
MODELS_VOLUME ?= sokil-models

# Host uid/gid, so files the container writes into the bind mount belong to you.
UID := $(shell id -u)
GID := $(shell id -g)
mounts = -v "$(PWD)":/data -v "$(MODELS_VOLUME)":/cache

# rich draws its progress bars only onto a terminal, so without a TTY a long
# review looks hung. Allocated conditionally: `docker run -t` fails outright
# when there is no terminal to attach, e.g. under CI.
tty_flag = $$([ -t 1 ] && echo -t)
docker_run = docker run --rm $(tty_flag) $(mounts) --user $(UID):$(GID) $(IMAGE)

# --- Hailo image (Raspberry Pi with a Hailo accelerator) ---------------------
HAILO_BUILD_IMAGE ?= sokil-hailo:dev
HAILO_REVIEW_IMAGE ?= ghcr.io/bonysmoke/sokil:latest-hailo
HAILO_DEVICE ?= /dev/hailo0
# The host's HailoRT, which the image's must match (see docker/hailo/README.md);
# left unset, the Dockerfile's own versions are built.
HAILORT_VERSION ?=
PYHAILORT_VERSION ?=

hailort_args = $(if $(HAILORT_VERSION),--build-arg HAILORT_VERSION=$(HAILORT_VERSION)) \
	$(if $(PYHAILORT_VERSION),--build-arg PYHAILORT_VERSION=$(PYHAILORT_VERSION))

.DEFAULT_GOAL := help

help:  ## Show this help
	@./helper.sh make-help $(MAKEFILE_LIST)

models-release:  ## Publish trained weights as a GitHub release (needs MODELS_TAG, SHUTTLE_WEIGHTS, COURT_WEIGHTS)
	@./helper.sh models-release "$(MODELS_TAG)" \
		shuttle="$(SHUTTLE_WEIGHTS)" \
		court="$(COURT_WEIGHTS)"

lint:  ## Check the Python sources with ruff, as CI does
	uv run ruff check .
	uv run ruff format --check .

format:  ## Reformat the Python sources and apply the safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

test-unit:  ## Run the unit tests with coverage, as CI does
	uv run pytest --cov --cov-report=term-missing

build:  ## Build the application image for this machine
	docker buildx build -f docker/cpu/Dockerfile $(groups_arg) -t $(IMAGE) .

build-multi-arch:  ## Build a multi-arch image
	docker buildx build -f docker/cpu/Dockerfile --platform $(PLATFORMS) $(groups_arg) -t $(IMAGE) .

build-hailo:  ## Build the Hailo image for a Raspberry Pi (linux/arm64), e.g. make build-hailo HAILORT_VERSION=4.23.0
	docker buildx build -f docker/hailo/Dockerfile --platform linux/arm64 --load $(hailort_args) $(groups_arg) -t $(HAILO_BUILD_IMAGE) .

review:  ## Run a review in the container, e.g. make review ARGS="--video-path a.mp4 --output-path b.mp4"
	$(docker_run) review $(ARGS)

review-hailo:  ## Run a review on a Hailo accelerator, e.g. make review-hailo ARGS="--video-path a.mp4 --output-path b.mp4 …"
	docker run --rm $(tty_flag) --device $(HAILO_DEVICE) $(mounts) --user $(UID):$(GID) $(HAILO_REVIEW_IMAGE) review $(ARGS)

shell:  ## Open a shell in the application image
	docker run --rm -it $(mounts) --user $(UID):$(GID) --entrypoint bash $(IMAGE)

cvat-env:
	@./helper.sh cvat-env "$(CVAT_ENV)"

cvat-up: cvat-env  ## Start CVAT in the background
	$(cvat) up -d
	@./helper.sh cvat-up-notice "$(CVAT_ENV)" "$(strip $(CVAT_SHARE_PATH))"

cvat-down: cvat-env  ## Stop CVAT
	$(cvat) down

cvat-ps: cvat-env  ## Show container status
	$(cvat) ps

cvat-logs: cvat-env  ## Follow the CVAT server logs
	$(cvat) logs -f cvat_server

cvat-pull: cvat-env  ## Pull the images for the pinned CVAT_VERSION
	$(cvat) pull

cvat-superuser: cvat-env  ## Create the admin user
	$(cvat) exec cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

include infra/hailo/hailo.mk
