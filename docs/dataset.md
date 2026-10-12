# Dataset

This document describes how to build a dataset to evaluate `Sokil`.

The purpose of the dataset is to evaluate the model performance on hit detection.

This document partially contains steps from [training.md](./training.md), so please check it out first.

## Stages

The dataset creation process consists of the following stages:

- Court segment annotation
- Shuttle box annotation
- Court keypoint annotation
- Landing labeling

### Court segment annotation

At this stage, you segment the court in the video. For a static camera, this will be just a one frame annotation.

```bash
uv run sokil create-cvat-task --video-path clip.mp4 --model court
```

### Shuttle box annotation

At this stage, you draw a box around the shuttle.

```bash
uv run sokil create-cvat-task --video-path clip.mp4 --model shuttle
```

### Court keypoint annotation

At this stage, you map the court model vertices to the image. This will be used to solve the homography.
You'll only need to annotate 1 frame in case the camera is static.

By default, you will annotate the median frame of the court that removes noise such as shuttle and players leaving only the background court.
Pass `--no-median-frame` to annotate the first frame instead.

```bash
uv run sokil create-cvat-task --video-path clip.mp4 --model court_keypoints
```

Once annotated, build the dataset table:

```bash
uv run sokil dataset court-points --path dataset
# one task, mapped to a clip whose recording was renamed since upload:
uv run sokil dataset court-points --task-id 14 --clip-id 14=court-5-left-100233
```

### Landing labeling

At this stage, you annotate the frames where the shuttle touches the ground as well as its position on the court.

```bash
uv run sokil dataset label-hits --video-path videos/clip.mp4 --court-type singles
```

## Splits

There are two types of splits.

### The evaluation split

The split is used for the landing detection tuning and is based on the `Subset` field in the `Badminton Court Keypoints` CVAT project.

```bash
uv run sokil dataset split
```

### The training split

This split is used for YOLO model training. It also relies in the `Subset` field of each task but in the corresponding project for the model in CVAT.

```bash
uv run sokil export-cvat-dataset --model shuttle
uv run sokil export-cvat-dataset --model court
```

## Validation

To validate that your dataset contains all the needed data to run evaluation, run this:

```bash
uv run sokil dataset validate --path dataset
```

## Evaluation

```bash
uv run sokil evaluate --configuration manual # no YOLO model is used. Everything comes from manual annotations
uv run sokil evaluate --configuration shuttle-detector # Court coordinates come from manual annotations and the shuttle is detected using a YOLO model
uv run sokil evaluate --configuration automatic # everything is detected automatically like in a real run
```

## Shape

```
dataset/
  clips.csv
  events.csv
  court_points.csv
  tracks.csv
  splits/{train,valid,test}.txt
```

### clips.csv

The file holds metadata for clips in the dataset

| Column | Description |
|---|---|
| `clip_id` | clip ID in the form of its name without extension |
| `source_uri` | Where that recording is, as a dataset-relative path or a URL. May be empty when the video is not published |
| `frame_count`, `fps`, `width`, `height` | The extracted frame sequence |
| `session_id` | The grouping key of multiple videos to differentiate between dates and halls |
| `camera_mode`, `court_type` | Matching `CourtCameraMode` and `CourtModelType` |
| `camera_static` | Whether the camera is static or moving |

### events.csv

The file contains data about hit events, whether the shuttle was in or our, court type (singles or doubles), etc.

| Column | Description |
|---|---|
| `clip_id` | clip ID in the form of its name without extension |
| `frame` | Which frame the shuttle landed on |
| `call` | `in` or `out` |
| `court_type` | The boundary the call was made against, singles or doubles |
| `line` | Which line it was judged against, sideline/backline |
| `x_px`, `y_px` | `x` and `y` position of the landing |
| `x_m`, `y_m` | The same point in court metres, derived through the clip's court keypoints |
| `annotator` | Who made the call |

### tracks.csv

The ground-truth shuttle box on every annotated frame. Generated from the tracks in the `Badminton Shuttle Detection` project:

```bash
uv run sokil dataset tracks
```

| Column | Description |
|---|---|
| `clip_id` | clip ID in the form of its name without extension |
|`frame` | the frame ID this box is on |
| `x1`, `y1`, `x2`, `y2` | box coordinates in the opencv format (top left to bottom right) |

### court_points.csv

Where the court is, in the image. One row per court vertex per keyframe, annotated in the `Badminton Court Keypoints` CVAT project as a skeleton.

| Column | Description |
|---|---|
| `clip_id` | clip ID in the form of its name without extension |
|`frame` | The keyframe these points describe |
| `vertex` | A `CourtModel.vertices` number, 1-based |
| `x_px`, `y_px` | Where that vertex is, empty when it is out of frame |
| `visible` | Whether the vertex is in frame |
| `source` | `solver`, `solver_adjusted` or `manual` |
