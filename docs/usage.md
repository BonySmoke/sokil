# Usage

This document describes how to configure, run, and interpret the results of `Sokil`.

## Setup

There are two ways to install `Sokil`.

**As a package**. Needs Python in your local system and
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run sokil --help
```

**As a container**, which avoids installing anything except [Docker](https://www.docker.com/):

```bash
make build
make review ARGS="--video-path clip.mp4 --output-path review.mp4"
```

The container mounts your working directory, so paths on the command line mean the same thing inside it as outside, and writes output as your own user.

## Models

Make sure to train models on the footage recorded in your hall as explained in [training.md](training.md).

## Configuration

```bash
uv run sokil review --help
```

### Camera Position & Court Boundaries

```
--court-camera-mode
--court-model-type
```

These flags specify what part of the court is visible on the footage, e.g. `left_corridor`, and the court boundaries, i.e. `singles` or `doubles`.

### Camera Calibration

```
--camera-intrinsics-path
--camera-dist-path
```

If you calibrated the camera as described in [camera-setup.md](camera-setup.md), path calibration matrices using these flags.

## Execution

> [!NOTE]  
> The review runs on the entire clip; therefore, it can be slow. To speed it up, pass shorter clips only where the hit happened.

Reviewing a clip is the main command:

```bash
# this is the most basic version, use --help to see available flags
uv run sokil review --video-path clip.mp4 --output-path review.mp4
```

or, in the container:

```bash
make review ARGS="--video-path clip.mp4 --output-path review.mp4"
```

Adding `--verbose` logs useful details about the hit frames, solving court homography, etc.

## Explanation & Interpretation

```
--explain
```

Use this flag to see what the model considers a court, where it sees the cork, etc.
