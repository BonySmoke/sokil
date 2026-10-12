"""
How exported frames are attributed to clips, thinned, and split.

The split is the reason these exist: frames from one rally are near-identical,
so a clip that lands in two partitions makes every later score meaningless while
looking perfectly fine in a diff.
"""

from itertools import pairwise
from pathlib import Path

import pytest

from sokil.dataset.splits import SplitError
from sokil.training.dataset import (
    _cap_by_spread,
    _clip_id,
    _frame_index,
    _identify,
    _split,
    _split_by_subset,
    _thin_by_frame_gap,
)


def sample(name: str, clip_id: str | None = None) -> dict:
    return {"name": name, "clip_id": clip_id or _clip_id(name, "task")}


class TestFrameIndex:
    def test_reads_the_trailing_number(self):
        assert _frame_index("clip_a_001671.jpg") == 1671

    def test_reads_it_through_a_directory(self):
        assert _frame_index("clip_a/frame_000123.jpg") == 123

    def test_is_none_without_one(self):
        assert _frame_index("no_number_here.jpg") is None


class TestClipId:
    def test_prefers_the_directory_cvat_preserved(self):
        assert _clip_id("court_5_left/frame_000123.jpg", "task") == "court_5_left"

    def test_falls_back_to_the_name_before_the_frame_number(self):
        assert _clip_id("court_5_left_001671.jpg", "task") == "court_5_left"

    def test_falls_back_to_the_task_when_the_name_says_nothing(self):
        assert _clip_id("screenshot.jpg", "task") == "task"

    def test_separates_clips_uploaded_into_one_task(self):
        names = ["clip_a/frame_000001.jpg", "clip_b/frame_000001.jpg"]
        assert {_clip_id(name, "task") for name in names} == {"clip_a", "clip_b"}


class TestThinning:
    def test_thins_within_a_clip(self):
        samples = [sample(f"clip_a_{i:06d}.jpg") for i in range(10)]
        kept = _thin_by_frame_gap(samples, min_frame_gap=5)
        assert [s["name"] for s in kept] == ["clip_a_000000.jpg", "clip_a_000005.jpg"]

    def test_does_not_measure_gaps_across_clips(self):
        # Both clips restart at frame 0, so a clip-blind thinner would drop
        # clip_b's frames for sitting on top of clip_a's.
        samples = [sample(f"clip_a_{i:06d}.jpg") for i in range(3)]
        samples += [sample(f"clip_b_{i:06d}.jpg") for i in range(3)]

        kept = _thin_by_frame_gap(samples, min_frame_gap=2)

        assert sorted(s["name"] for s in kept) == [
            "clip_a_000000.jpg",
            "clip_a_000002.jpg",
            "clip_b_000000.jpg",
            "clip_b_000002.jpg",
        ]

    def test_leaves_samples_alone_when_a_frame_index_is_missing(self):
        samples = [sample("clip_a_000000.jpg"), sample("no_number.jpg")]
        assert _thin_by_frame_gap(samples, min_frame_gap=100) == samples


def clips_of(partition: list[dict]) -> set[str]:
    return {s["clip_id"] for s in partition}


class TestSplit:
    @pytest.fixture
    def samples(self) -> list[dict]:
        return [
            sample(f"clip_{letter}_{i:06d}.jpg")
            for letter in "abcde"
            for i in range(10)
        ]

    def test_no_clip_appears_in_two_partitions(self, samples):
        splits = _split(samples, seed=1, valid_size=0.2, test_size=0.2)

        train, valid, test = (clips_of(splits[p]) for p in ("train", "valid", "test"))
        assert train & valid == set()
        assert train & test == set()
        assert valid & test == set()

    def test_every_frame_lands_somewhere_exactly_once(self, samples):
        splits = _split(samples, seed=1, valid_size=0.2, test_size=0.2)

        names = [s["name"] for partition in splits.values() for s in partition]
        assert sorted(names) == sorted(s["name"] for s in samples)

    def test_a_clips_frames_stay_together(self, samples):
        splits = _split(samples, seed=3, valid_size=0.2, test_size=0.2)

        for partition in splits.values():
            for clip_id in clips_of(partition):
                assert len([s for s in partition if s["clip_id"] == clip_id]) == 10

    def test_is_deterministic_for_a_seed(self, samples):
        first = _split(samples, seed=7, valid_size=0.2, test_size=0.2)
        second = _split(samples, seed=7, valid_size=0.2, test_size=0.2)

        assert {p: clips_of(s) for p, s in first.items()} == {
            p: clips_of(s) for p, s in second.items()
        }

    def test_reaches_the_requested_size(self, samples):
        splits = _split(samples, seed=1, valid_size=0.2, test_size=0.2)

        # 50 frames in clips of 10: a fifth is exactly one clip.
        assert len(splits["valid"]) == 10
        assert len(splits["test"]) == 10
        assert len(splits["train"]) == 30

    def test_overshoots_rather_than_splitting_a_clip(self):
        # A third of 30 frames is 10, but the clips are 15 each, so the only
        # way to honour the request exactly would be to break one open.
        samples = [
            sample(f"clip_{letter}_{i:06d}.jpg") for letter in "ab" for i in range(15)
        ]

        splits = _split(samples, seed=1, valid_size=10, test_size=0)

        assert len(splits["valid"]) == 15
        assert len(splits["train"]) == 15

    def test_refuses_to_leave_a_partition_empty(self):
        samples = [sample(f"clip_a_{i:06d}.jpg") for i in range(10)]

        with pytest.raises(ValueError, match="cannot be split across partitions"):
            _split(samples, seed=1, valid_size=0.2, test_size=0.2)

    def test_a_single_clip_can_still_train(self):
        samples = [sample(f"clip_a_{i:06d}.jpg") for i in range(10)]

        splits = _split(samples, seed=1, valid_size=0, test_size=0)

        assert len(splits["train"]) == 10


class TestSplitBySubset:
    """
    Assigning clips by the Subset their CVAT task declares. With one clip per
    camera angle the assignment is a judgement about coverage, so it is taken
    from CVAT rather than invented by a shuffle.
    """

    def samples_for(self, clips: dict[str, int]) -> list[dict]:
        return [
            sample(f"{clip}_{i:06d}.jpg")
            for clip, count in clips.items()
            for i in range(count)
        ]

    def test_each_clip_goes_where_its_subset_says(self):
        samples = self.samples_for({"clipa": 3, "clipb": 2, "clipc": 1})

        splits = _split_by_subset(
            samples, {"clipa": "Train", "clipb": "Validation", "clipc": "Test"}
        )

        assert len(splits["train"]) == 3
        assert len(splits["valid"]) == 2
        assert len(splits["test"]) == 1

    def test_a_clip_s_frames_stay_together(self):
        samples = self.samples_for({"clipa": 4, "clipb": 4})

        splits = _split_by_subset(samples, {"clipa": "Train", "clipb": "Test"})

        assert {s["clip_id"] for s in splits["train"]} == {"clipa"}
        assert {s["clip_id"] for s in splits["test"]} == {"clipb"}

    def test_every_frame_lands_exactly_once(self):
        samples = self.samples_for({"clipa": 3, "clipb": 2})

        splits = _split_by_subset(samples, {"clipa": "Train", "clipb": "Test"})

        names = [s["name"] for part in splits.values() for s in part]
        assert sorted(names) == sorted(s["name"] for s in samples)

    def test_a_clip_with_no_subset_is_refused(self):
        samples = self.samples_for({"clipa": 2, "clipb": 2})

        with pytest.raises(SplitError, match="no subset in CVAT"):
            _split_by_subset(samples, {"clipa": "Train", "clipb": ""})

    def test_an_unrecognised_subset_is_refused(self):
        samples = self.samples_for({"clipa": 2})

        with pytest.raises(SplitError, match="unrecognised subset"):
            _split_by_subset(samples, {"clipa": "Holdout"})

    def test_the_refusal_says_how_to_proceed(self):
        samples = self.samples_for({"clipa": 2})

        with pytest.raises(SplitError, match="--seed-split"):
            _split_by_subset(samples, {"clipa": ""})

    def test_nothing_to_train_on_is_refused(self):
        samples = self.samples_for({"clipa": 2})

        with pytest.raises(SplitError, match="nothing to train on"):
            _split_by_subset(samples, {"clipa": "Test"})

    def test_an_empty_test_partition_is_allowed(self):
        # A camera angle with one clip belongs in train, which can leave test
        # empty for a model that only has one angle.
        samples = self.samples_for({"clipa": 2})

        splits = _split_by_subset(samples, {"clipa": "Train"})

        assert splits["test"] == []


class TestIdentifyThenThin:
    """
    The order these two run in. Thinning groups by clip, so a sample has to
    know its clip first — and with the default gap of 1 nothing thins, so the
    wrong order stays invisible until somebody passes --min-frame-gap.
    """

    def parsed(self, names: list[str]) -> list[dict]:
        """Samples as the CVAT parser leaves them: a name and nothing else."""
        return [
            {"name": name, "width": 640, "height": 480, "objects": []} for name in names
        ]

    def images(self, names: list[str]):
        paths = {Path(name).stem: Path(f"/tmp/{Path(name).stem}.jpg") for name in names}
        return {}, paths

    def test_identifying_gives_every_sample_a_clip(self):
        names = ["clip_a_000000.jpg", "clip_a_000001.jpg"]
        by_path, by_stem = self.images(names)

        samples = _identify(self.parsed(names), "task", 7, by_path, by_stem)

        assert all(sample["clip_id"] == "clip_a" for sample in samples)
        assert all(sample["task_id"] == 7 for sample in samples)

    def test_thinning_works_on_identified_samples(self):
        names = [f"clip_a_{index:06d}.jpg" for index in range(10)]
        by_path, by_stem = self.images(names)

        samples = _identify(self.parsed(names), "task", 7, by_path, by_stem)
        kept = _thin_by_frame_gap(samples, min_frame_gap=5)

        assert [sample["name"] for sample in kept] == [
            "clip_a_000000.jpg",
            "clip_a_000005.jpg",
        ]

    def test_thinning_before_identifying_is_a_mistake(self):
        # The bug this ordering exists to prevent: unidentified samples have no
        # clip to group by, and thinning cannot invent one.
        with pytest.raises(KeyError):
            _thin_by_frame_gap(self.parsed(["clip_a_000000.jpg"]), min_frame_gap=5)

    def test_a_task_named_frame_sequence_takes_the_clip_from_its_task(self):
        # Frames uploaded as frame_000123.jpg carry no clip of their own, so
        # the task name supplies it and the exported name is rewritten.
        names = ["frame_000000.jpg", "frame_000030.jpg"]
        by_path, by_stem = self.images(names)

        samples = _identify(self.parsed(names), "court_5", 7, by_path, by_stem)

        assert [sample["name"] for sample in samples] == [
            "court_5_000000.jpg",
            "court_5_000030.jpg",
        ]


class TestCappingPartitions:
    """
    How much of a partition to measure on, once the subsets have said which
    clips are in it. Three thousand frames of one rally is not three thousand
    pieces of evidence, so what matters is that the frames kept are spread
    rather than merely fewer.
    """

    def samples_for(self, clips: dict[str, int]) -> list[dict]:
        return [
            sample(f"{clip}_{index:06d}.jpg")
            for clip, count in clips.items()
            for index in range(count)
        ]

    def frames_of(self, samples: list[dict], clip: str) -> list[int]:
        return sorted(_frame_index(s["name"]) for s in samples if s["clip_id"] == clip)

    def test_no_cap_leaves_the_partition_alone(self):
        samples = self.samples_for({"clipa": 100})

        assert len(_cap_by_spread(samples, 0, "valid")) == 100

    def test_a_cap_larger_than_the_partition_leaves_it_alone(self):
        samples = self.samples_for({"clipa": 10})

        assert len(_cap_by_spread(samples, 500, "valid")) == 10

    def test_an_absolute_cap_is_honoured(self):
        samples = self.samples_for({"clipa": 100})

        assert len(_cap_by_spread(samples, 10, "valid")) == 10

    def test_a_fraction_is_of_this_partition(self):
        samples = self.samples_for({"clipa": 100})

        assert len(_cap_by_spread(samples, 0.25, "valid")) == 25

    def test_the_frames_kept_are_spread_not_the_first_few(self):
        # The whole point: taking the front of a clip would keep one moment of
        # one rally, which is what makes a validation figure hollow.
        samples = self.samples_for({"clipa": 100})

        frames = self.frames_of(_cap_by_spread(samples, 10, "valid"), "clipa")

        assert frames[0] == 0
        assert frames[-1] >= 80
        gaps = {b - a for a, b in pairwise(frames)}
        assert max(gaps) - min(gaps) <= 1

    def test_each_clip_keeps_a_share_of_its_own_length(self):
        # One long clip must not crowd the others out of the partition.
        samples = self.samples_for({"clipa": 900, "clipb": 100})

        kept = _cap_by_spread(samples, 100, "valid")

        assert len(self.frames_of(kept, "clipa")) == 90
        assert len(self.frames_of(kept, "clipb")) == 10

    def test_a_short_clip_keeps_at_least_one_frame(self):
        samples = self.samples_for({"clipa": 1000, "clipb": 3})

        kept = _cap_by_spread(samples, 10, "valid")

        assert self.frames_of(kept, "clipb")

    def test_an_empty_partition_stays_empty(self):
        assert _cap_by_spread([], 10, "test") == []

    def test_the_cap_applies_through_the_split(self):
        samples = self.samples_for({"clipa": 200, "clipb": 200})

        splits = _split_by_subset(
            samples, {"clipa": "Train", "clipb": "Test"}, test_size=20
        )

        assert len(splits["train"]) == 200
        assert len(splits["test"]) == 20

    def test_train_is_never_capped(self):
        # Every annotated frame is worth training on; only what you measure on
        # needs thinning.
        samples = self.samples_for({"clipa": 200, "clipb": 50})

        splits = _split_by_subset(
            samples,
            {"clipa": "Train", "clipb": "Valid" and "Validation"},
            valid_size=10,
        )

        assert len(splits["train"]) == 200
        assert len(splits["valid"]) == 10
