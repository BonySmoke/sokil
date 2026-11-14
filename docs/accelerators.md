# Accelerators

`Sokil` uses Ultralytics under the hood, so any [export format](https://docs.ultralytics.com/integrations#export-formats) works with `--shuttle-detector-model-path` and `--court-segmentation-model-path`. Most formats need their runtime in a custom image.

This document describes the accelerators `Sokil` supports out of the box.

## Hailo

[Hailo](https://hailo.ai/) makes AI accelerators for edge devices such as the Raspberry Pi. To run `Sokil` on one, compile your [trained models](training.md) to HEF.

### Prerequisites

- The [Hailo AI Software Suite](https://hailo.ai/developer-zone/software-downloads/) Docker image (needs a Developer Zone login). Use the `.tar.gz` inside the downloaded `.zip`.
- An exported dataset for each model ([step 2](training.md#2-export-a-dataset)): calibration reads its `train` split.

Pass `HAILO_ARCH=hailo8l` to any command below to build for a Hailo-8L.

### Local compilation

Needs a Linux x86_64 machine with Docker, 16 GB of RAM for the shuttle model and about 32 GB for the court model. An NVIDIA GPU with the NVIDIA container toolkit is optional but makes it many times faster; set `HAILO_GPUS` to choose GPUs.

1. Load the suite and build the compiler image on it:

   ```bash
   docker load -i hailo8_ai_sw_suite_<version>.tar.gz   # prints the tag it loaded
   make hailo-build HAILO_SUITE_IMAGE=<that tag>
   ```

2. Compile. The weights must be inside the repository:

   ```bash
   make hailo-local COURT_WEIGHTS=training/court/models/train.pt \
       SHUTTLE_WEIGHTS=training/shuttle/models/train.pt
   ```

### Remote compilation

Compiles on an AWS spot GPU (`g4dn.2xlarge`) that terminates itself when done. See [infra/README.md](../infra/README.md) for what it creates and how to remove it.

#### Prerequisites

- `awscli` and `opentofu`; `session-manager-plugin` for the logs and shell targets.
- The AWS CLI configured with an admin user: setup creates S3, IAM, EC2 and SSM resources.
- A quota of 8 vCPUs for **All G and VT Spot Instance Requests** in the region. The default is `us-east-2`; for another, put `region = "<region>"` in `infra/hailo/cloud/persistent/terraform.tfvars`.

#### Setup

1. Create the bucket, the VM's role and its security group:

   ```bash
   make hailo-cloud-setup
   ```

2. Upload the suite, again whenever Hailo releases a new one:

   ```bash
   make hailo-cloud-suite SUITE=/path/to/hailo8_ai_sw_suite_<version>.tar.gz
   ```

#### Compilation

```bash
make hailo-cloud COURT_WEIGHTS=training/court/models/train.pt \
    SHUTTLE_WEIGHTS=training/shuttle/models/train.pt HAILO_CLOUD_MAX_MINUTES=130
```

Court takes about 85 minutes and shuttle about 40. The VM stops after `HAILO_CLOUD_MAX_MINUTES` (default 90), so raise it when compiling both.

- `make hailo-cloud-logs` follows the progress; `make hailo-cloud-shell` opens a shell on the VM.
- After a Ctrl-C, `make hailo-cloud-fetch` picks the run up again; the VM keeps going.
- Without spot capacity, `make hailo-cloud` fails without starting anything; try again later.

### Result

Each model that compiled ends up in `training/<model>/hailo/<stem>-hailo8_hailo_model/`. A model that failed keeps its previous HEF and leaves its logs in `training/hailo-local/<run>/` or `training/hailo-cloud/<run>/`.

Before relying on a HEF, compare it with the `.pt` weights on the `test` split: INT8 quantization can cost accuracy.

### Publishing

To publish models, run:

```bash
make hailo-release SHUTTLE_HEF=training/shuttle/hailo/train-hailo8_hailo_model \
    COURT_HEF=training/court/hailo/train-hailo8_hailo_model
```

It uploads each directory as a `.tar.gz` to the release and prints the `formats` entries to add to `sokil/models.yaml`. Once they are committed, the Hailo image downloads the HEFs instead of other formats.

### Running

On the Pi, `make review-hailo` runs the published `-hailo` image with the device passed through. Its HailoRT must match the host's (see [docker/hailo](../docker/hailo/README.md)).

To use your own build, run `make build-hailo` and set `HAILO_REVIEW_IMAGE=sokil-hailo:dev`.

```bash
make review-hailo ARGS="--video-path clip.mp4 --output-path review.mp4"
```

To use other models, pass `--shuttle-detector-model-path` and `--court-segmentation-model-path`; the directories' names must end in `_hailo_model`.
