# Training and publishing models

Sokil uses two trained models:
- Shuttle detector
- Court segmenter

This document covers the process of training them and publishing.

## Lifecycle

```mermaid
flowchart LR
    footage[(Footage)] --> annotate[Annotate in CVAT]
    annotate --> export[Export a dataset]
    export --> train[Train]
    train --> evaluate[Evaluate]
    evaluate --> publish[Publish a release]
    publish --> pin[Pin in models.yaml]
```

## 1. Annotate

We use [CVAT](https://www.cvat.ai/) for annotation. To bring it up, run this:

```bash
make cvat-up
make cvat-superuser        # first time only
```

Then, create a task from your videos:

```bash
uv run sokil create-cvat-task --video-path clip.mp4 --model shuttle # run --help to see available options
```

See [../infra/README.md](../infra/README.md) for running CVAT itself.

## 2. Export a dataset

Once the tasks are marked complete in CVAT:

```bash
uv run sokil export-cvat-dataset --model shuttle
```

This pulls the completed tasks and writes a dataset under `training/<model>/`, splitting it into train, validation and test. Frames are converted to the expected model format, e.g. `grayscale`.

## 3. Train

```bash
uv run sokil train-model --model shuttle
```

Per-model defaults come from `sokil/training/training.yaml`.

## 4. Evaluate

For now, the evaluation dataset is not published. Therefore, the best you can do is run a review using the trained model.

```bash
uv run sokil review --video-path clip.mp4 --output-path new.mp4 \
    --shuttle-detector-model-path runs/detect/train8/weights/best.pt
```

---

> [!NOTE]  
> The following steps are for project maintainers.

## 5. Publish and pin

**Publish the release.** The weights are uploaded under the asset names the manifest expects:

```bash
make models-release MODELS_TAG=models-2026-09-06 \
    SHUTTLE_WEIGHTS=runs/detect/train/weights/best.pt \
    COURT_WEIGHTS=runs/segment/train/weights/best.pt
```

**Pin them in the manifest.** This computes the checksums and prints the
manifest entries:

```bash
uv run sokil models pin --tag models-2026-09-06 \
    --weights shuttle=runs/detect/train/weights/best.pt \
    --weights court=runs/segment/train/weights/best.pt
```

Paste the output into `sokil/models.yaml`, keeping its comments, and commit it.

## 6. Compile for Hailo (optional)

A Hailo accelerator runs its own compiled format, HEF. The compiler, Hailo's
Dataflow Compiler, runs only on Linux x86_64. It needs no Hailo device, and
an NVIDIA GPU is optional: without one it optimizes on the CPU, which is
slower and can cost accuracy.

The compiler is not on a public registry. Sign in to the [Hailo Developer
Zone](https://hailo.ai/developer-zone/), download the AI Software Suite Docker
image, and load it:

```bash
docker load -i hailo8_ai_sw_suite_<version>.tar.gz   # prints the tag it loaded
make hailo-build HAILO_SUITE_IMAGE=<that tag>
```

Then compile each model. Calibration uses about 1,024 images from the
model's `train` split, so export its dataset first (step 2):

```bash
make hailo-export MODEL=shuttle WEIGHTS=training/shuttle/best.pt
make hailo-export MODEL=court WEIGHTS=training/court/best.pt
```

The repository is mounted read-only, so the weights must be inside it. Each
run writes `training/<model>/hailo/<stem>-hailo8_hailo_model/`, holding the
`.hef` and the metadata `YOLO()` needs to load it on the device. Pass
`HAILO_ARCH=hailo8l` to build for a Hailo-8L. The export uses every NVIDIA GPU
when the host has `nvidia-smi` and the CPU otherwise; set `HAILO_GPUS` to
override.

Before relying on a HEF, compare its detections with the `.pt` weights on the
`test` split. INT8 quantization can cost recall on a target as small as the
shuttle.

## Versioning

`sokil/models.yaml` is the single source of truth for which weights a given
commit expects. Nothing discovers a version at runtime.

| Field | Meaning |
|---|---|
| `version` | Human-facing label, dated `YYYY-MM-DD`. Make sure to quote it so that YAML parses it as a string. |
| `release_tag` | The release carrying the asset, `models-<version>` by convention. |
| `asset` | The published file name. |
| `sha256` | The checksum. |
| `expects` | What the weights were trained to be fed: image size, grayscale, class names. |

## Model Precedence Order

1. Command line argument.
2. `SOKIL_MODELS_DIR`, if set.
3. The checkout's own `models/` directory — useful for trying weights you have just trained, before publishing anything.
4. The download cache, keyed by version.
5. Otherwise the pinned release is downloaded. If that fails, the run fails.

```bash
uv run sokil models path     # reports what each model currently resolves to
uv run sokil models download # pre-fetches all models
```
