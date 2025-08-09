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

# the release tag to publish to, e.g. make models-release MODELS_TAG=models-2026-09-06
MODELS_TAG ?=

# Required by models-release, deliberately without defaults: trained weights
# live wherever the training run left them, which is particular to each machine.
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
	docker buildx build -t $(IMAGE) .

build-multi-arch:  ## Build a multi-arch image
	docker buildx build --platform $(PLATFORMS) -t $(IMAGE) .

review:  ## Run a review in the container, e.g. make review ARGS="--video-path a.mp4 --output-path b.mp4"
	$(docker_run) review $(ARGS)

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
