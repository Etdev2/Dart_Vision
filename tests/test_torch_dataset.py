"""Tests for the training dataset."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from dartvision.data import Annotation, Origin, image_filename  # noqa: E402
from dartvision.data.torch_dataset import (  # noqa: E402
    PhotometricJitter,
    SceneDataset,
    collate,
)

LANDMARKS_8 = tuple((0.2 + 0.07 * i, 0.25 + 0.06 * i) for i in range(8))


def annotation(image_id="s0/sess0/img-0000", tips=((0.5, 0.5),), landmarks=LANDMARKS_8):
    return Annotation(
        image_id=image_id, setup_id="s0", session_id="s0/sess0",
        origin=Origin.SYNTHETIC, landmarks=landmarks, tips=tips,
        image_size=(1200, 1600),
    )


@pytest.fixture
def rendered(tmp_path):
    """A manifest with matching image files on disk."""
    annotations = [
        annotation(f"s0/sess0/img-{i:04d}", tips=tuple((0.4 + 0.1 * j, 0.5) for j in range(i % 4)))
        for i in range(6)
    ]
    for a in annotations:
        rng = np.random.default_rng(len(a.image_id))
        Image.fromarray(
            (rng.random((160, 120, 3)) * 255).astype("uint8")
        ).save(tmp_path / image_filename(a.image_id))
    return annotations, tmp_path


def test_filename_derivation_is_shared_with_the_renderer():
    """Both sides must agree, or the failure looks like missing data."""
    assert image_filename("a/b/c-0001") == "a_b_c-0001.jpg"
    assert image_filename("x", suffix=".png") == "x.png"


def test_sample_shapes_follow_input_size_and_stride(rendered):
    annotations, root = rendered
    dataset = SceneDataset(annotations, root, input_size=(256, 256), stride=4)

    assert dataset.grid == (64, 64)
    sample = dataset[0]
    assert sample.image.shape == (3, 256, 256)
    assert sample.landmark_heat.shape == (8, 64, 64)
    assert sample.tip_heat.shape == (1, 64, 64)
    assert sample.tip_offset.shape == (2, 64, 64)
    assert sample.tip_mask.shape == (1, 64, 64)


def test_images_are_normalized_to_unit_range(rendered):
    annotations, root = rendered
    image = SceneDataset(annotations, root, input_size=(128, 128))[0].image
    assert image.dtype == torch.float32
    assert 0.0 <= image.min().item() and image.max().item() <= 1.0


def test_resizing_needs_no_label_change_because_labels_are_normalized(rendered):
    """The same annotation at two input sizes must encode the same peak cell."""
    annotations, root = rendered
    small = SceneDataset(annotations, root, input_size=(128, 128), stride=4)[0]
    large = SceneDataset(annotations, root, input_size=(256, 256), stride=4)[0]

    for channel in range(8):
        peak_small = np.unravel_index(int(small.landmark_heat[channel].argmax()), (32, 32))
        peak_large = np.unravel_index(int(large.landmark_heat[channel].argmax()), (64, 64))
        assert peak_small[0] == pytest.approx(peak_large[0] / 2, abs=1)
        assert peak_small[1] == pytest.approx(peak_large[1] / 2, abs=1)


def test_tip_count_matches_the_annotation(rendered):
    annotations, root = rendered
    dataset = SceneDataset(annotations, root, input_size=(256, 256))
    for index, a in enumerate(annotations):
        assert dataset[index].tip_mask.sum().item() == len(a.tips)


def test_an_empty_board_yields_empty_tip_targets(rendered):
    annotations, root = rendered
    dataset = SceneDataset([annotations[0].__class__(**{**annotations[0].to_dict(),
        "origin": Origin.SYNTHETIC, "landmarks": LANDMARKS_8, "tips": ()})], root)
    sample = dataset[0]
    assert sample.tip_mask.sum().item() == 0
    assert sample.tip_heat.max().item() == 0.0


def test_photometric_jitter_changes_pixels_but_stays_in_range(rendered):
    annotations, root = rendered
    plain = SceneDataset(annotations, root, input_size=(128, 128))[0].image
    jittered = SceneDataset(
        annotations, root, input_size=(128, 128), jitter=PhotometricJitter()
    )[0].image

    assert not torch.allclose(plain, jittered)
    assert 0.0 <= jittered.min().item() and jittered.max().item() <= 1.0


def test_jitter_leaves_targets_untouched(rendered):
    """Label-safe means label-identical, which is the point of separating it."""
    annotations, root = rendered
    plain = SceneDataset(annotations, root, input_size=(128, 128))[0]
    jittered = SceneDataset(
        annotations, root, input_size=(128, 128), jitter=PhotometricJitter()
    )[0]
    assert torch.equal(plain.landmark_heat, jittered.landmark_heat)
    assert torch.equal(plain.tip_offset, jittered.tip_offset)


def test_a_missing_image_says_so_plainly(tmp_path):
    with pytest.raises(FileNotFoundError, match="render the manifest first"):
        SceneDataset([annotation()], tmp_path)[0]


def test_a_mixed_landmark_manifest_is_rejected(rendered):
    annotations, root = rendered
    mixed = list(annotations) + [annotation("s0/sess0/img-9999", landmarks=LANDMARKS_8[:4])]
    with pytest.raises(ValueError, match="same number of landmarks"):
        SceneDataset(mixed, root)


def test_input_size_must_be_divisible_by_stride(rendered):
    annotations, root = rendered
    with pytest.raises(ValueError, match="divisible by stride"):
        SceneDataset(annotations, root, input_size=(250, 250), stride=4)


def test_empty_annotations_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="no annotations"):
        SceneDataset([], tmp_path)


def test_collate_batches_tensors_and_keeps_ids(rendered):
    annotations, root = rendered
    dataset = SceneDataset(annotations, root, input_size=(128, 128))
    batch = collate([dataset[i] for i in range(3)])

    assert batch["image"].shape == (3, 3, 128, 128)
    assert batch["landmark_heat"].shape == (3, 8, 32, 32)
    assert len(batch["image_id"]) == 3
    assert batch["session_id"] == ["s0/sess0"] * 3


def test_a_batch_feeds_the_model_without_reshaping(rendered):
    """Dataset output and model input must line up with nothing in between."""
    pytest.importorskip("timm")
    from dartvision.model.net import DartVisionBrain

    annotations, root = rendered
    dataset = SceneDataset(annotations, root, input_size=(128, 128), stride=4)
    batch = collate([dataset[i] for i in range(2)])

    model = DartVisionBrain(backbone="resnet18", landmarks=8, width=16, stride=4)
    out = model(batch["image"])
    assert out.landmark_heat.shape == batch["landmark_heat"].shape
    assert out.tip_heat.shape == batch["tip_heat"].shape
    assert out.tip_offset.shape == batch["tip_offset"].shape
