"""
Deriving the evaluation split from CVAT's task subsets.

The split exists to keep hit-detection tuning honest: train is what the grid
search may see, test is what the published figure rests on. So an unanswered
subset must not quietly become train — that is exactly the mistake the split is
there to prevent.
"""

import pytest

from sokil.dataset.splits import SplitError, mode_coverage, split_from_subsets


class TestSplitFromSubsets:
    def test_maps_cvat_s_names_onto_the_dataset_s(self):
        splits = split_from_subsets({"a": "Train", "b": "Validation", "c": "Test"})

        assert splits == {"train": ["a"], "valid": ["b"], "test": ["c"]}

    def test_is_case_insensitive(self):
        splits = split_from_subsets({"a": "TRAIN", "b": "test"})

        assert splits["train"] == ["a"]
        assert splits["test"] == ["b"]

    def test_several_clips_per_split_come_out_sorted(self):
        splits = split_from_subsets({"c": "Train", "a": "Train", "b": "Train"})

        assert splits["train"] == ["a", "b", "c"]

    def test_an_empty_split_is_allowed(self):
        splits = split_from_subsets({"a": "Train"})

        assert splits["test"] == []
        assert splits["valid"] == []

    def test_a_clip_with_no_subset_is_refused(self):
        # Defaulting it to train would train on a clip somebody meant to hold
        # out, and say nothing about having done so.
        with pytest.raises(SplitError, match="no subset in CVAT"):
            split_from_subsets({"a": "Train", "b": ""})

    def test_a_subset_of_whitespace_is_still_no_subset(self):
        with pytest.raises(SplitError, match="no subset in CVAT"):
            split_from_subsets({"a": "Train", "b": "   "})

    def test_a_subset_nobody_defined_is_refused(self):
        with pytest.raises(SplitError, match="does not recognise"):
            split_from_subsets({"a": "Train", "b": "Holdout"})

    def test_the_refusal_names_every_offending_clip(self):
        with pytest.raises(SplitError) as caught:
            split_from_subsets({"a": "", "b": "", "c": "Train"})

        assert "a" in str(caught.value)
        assert "b" in str(caught.value)

    def test_nothing_to_tune_on_is_refused(self):
        with pytest.raises(SplitError, match="nothing to tune"):
            split_from_subsets({"a": "Test", "b": "Validation"})


class TestModeCoverage:
    """
    With one clip per camera angle, a split can quietly ask the impossible.
    These warnings are the point of recording camera_mode at all.
    """

    def test_a_mode_in_both_train_and_test_is_fine(self):
        splits = {"train": ["a"], "valid": [], "test": ["b"]}
        modes = {"a": "left_corridor", "b": "left_corridor"}

        assert mode_coverage(splits, modes) == []

    def test_a_mode_only_in_test_is_flagged(self):
        # Nothing trained on this angle, so the figure measures the gap.
        splits = {"train": ["a"], "valid": [], "test": ["b"]}
        modes = {"a": "left_corridor", "b": "top_corridor"}

        warnings = mode_coverage(splits, modes)

        assert any("in test but not in train" in w for w in warnings)
        assert any("top_corridor" in w for w in warnings)

    def test_a_mode_only_in_train_is_reported_as_untested(self):
        splits = {"train": ["a", "b"], "valid": [], "test": ["b"]}
        modes = {"a": "top_corridor", "b": "left_corridor"}

        warnings = mode_coverage(splits, modes)

        assert any("never tested" in w and "top_corridor" in w for w in warnings)

    def test_a_clip_with_no_known_mode_claims_no_mode(self):
        # The test clip's mode is unknown, so it cannot be said to extrapolate
        # to anything; the train mode being untested is still true and said.
        splits = {"train": ["a"], "valid": [], "test": ["b"]}

        warnings = mode_coverage(splits, {"a": "left_corridor"})

        assert not any("in test but not in train" in w for w in warnings)
        assert any("never tested" in w for w in warnings)
