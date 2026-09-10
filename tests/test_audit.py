"""Tests for #14 section 7's dataset verification checks.

The checks exist to stop a corpus from producing a confident, meaningless
number, so the tests are mostly about the failure cases: a corpus that is
subtly wrong must be reported as wrong, and a corpus that is merely weak must
be reported as weak rather than as broken.

Corpora here are built with the synthetic generator, so margins to the nearest
wire are exact rather than assumed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dartvision.audit import audit, check_darts_per_image, check_schema
from dartvision.audit.checks import (
    FAIL,
    PASS,
    SKIPPED,
    WARN,
    check_cross_session_duplicates,
    check_margin_distribution,
    check_split_grouping,
    check_within_session_duplication,
)
from dartvision.audit.hashing import Pair, cluster, dhash, hamming, near_duplicate_pairs
from dartvision.data.labels import Annotation, Origin, write_jsonl
from dartvision.data.splits import SplitAssignment, build_session_split, split_for_tier
from dartvision.geometry.camera import CameraPose
from dartvision.synthetic import build_scene, sample_tip_near_boundary, sample_tip_uniform

POSE = CameraPose(
    elevation_deg=55.0, azimuth_deg=20.0, distance_mm=1800.0, focal_px=4200.0
)


def scene(image_id, tips_mm, session_id="s0/sess0", setup_id="s0"):
    return build_scene(
        POSE, tips_mm, image_id=image_id, setup_id=setup_id, session_id=session_id
    ).annotation


def corpus(sessions=3, per_session=4, near_wire=True, seed=0):
    """A synthetic corpus with a controlled margin distribution."""
    rng = np.random.default_rng(seed)
    annotations = []
    for s in range(sessions):
        session_id = f"set{s % 2}/sess{s}"
        for i in range(per_session):
            tips = [
                sample_tip_near_boundary(rng, 0.4) if near_wire
                else sample_tip_uniform(rng)
                for _ in range(i % 4)
            ]
            annotations.append(
                scene(
                    f"{session_id}/img-{s}{i:02d}", tips,
                    session_id=session_id, setup_id=f"set{s % 2}",
                )
            )
    return annotations


def status_of(report, name):
    return next(c for c in report.checks if c.name == name).status


def detail(report, name, key):
    return next(c for c in report.checks if c.name == name).details[key]


# --------------------------------------------------------------------------
# Perceptual hashing
# --------------------------------------------------------------------------

def images(tmp_path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    from PIL import Image

    paths = {}
    for name, array in arrays.items():
        path = tmp_path / f"{name}.png"
        Image.fromarray(array.astype("uint8")).save(path)
        paths[name] = path
    return paths


def test_a_hash_is_stable_and_an_unrelated_image_is_far(tmp_path):
    pytest.importorskip("PIL")
    rng = np.random.default_rng(0)
    gradient = np.tile(np.linspace(0, 255, 64), (64, 1))
    paths = images(tmp_path, {
        "a": gradient,
        "a_again": gradient,
        "noise": rng.random((64, 64)) * 255,
    })

    assert dhash(paths["a"]) == dhash(paths["a_again"])
    assert hamming(dhash(paths["a"]), dhash(paths["noise"])) > 10


def test_a_slightly_altered_frame_stays_near(tmp_path):
    """The point of a perceptual hash: a burst frame must collide."""
    pytest.importorskip("PIL")
    rng = np.random.default_rng(1)
    base = rng.random((64, 64)) * 200
    paths = images(tmp_path, {"base": base, "jittered": base + 6.0})

    assert hamming(dhash(paths["base"]), dhash(paths["jittered"])) <= 2


def test_pairs_are_unique_unordered_and_thresholded():
    hashes = {"a": 0b1111, "b": 0b1110, "c": 0b0001}
    close = near_duplicate_pairs(hashes, threshold=1)

    assert [(p.left, p.right) for p in close] == [("a", "b")]
    assert near_duplicate_pairs(hashes, threshold=0) == []
    assert len(near_duplicate_pairs(hashes, threshold=64)) == 3


def test_clusters_are_transitive():
    """A burst where each frame is close only to its neighbour is still one
    moment, and must count as one."""
    pairs = [Pair("a", "b", 1), Pair("b", "c", 1)]
    assert cluster(["a", "b", "c", "d"], pairs) == [["a", "b", "c"], ["d"]]


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(3)
    hashes = {f"i{i}": int(v) for i, v in enumerate(rng.integers(0, 2**63, 40))}
    assert near_duplicate_pairs(hashes, 20, chunk=7) == near_duplicate_pairs(
        hashes, 20, chunk=1000
    )


# --------------------------------------------------------------------------
# Schema conformance
# --------------------------------------------------------------------------

def test_a_clean_manifest_passes():
    result = check_schema(corpus())
    assert result.status == PASS
    assert result.details["below_four_landmarks"] == 0


def test_an_empty_manifest_fails():
    assert check_schema([]).status == FAIL


def test_duplicate_image_ids_fail():
    annotations = corpus(sessions=1, per_session=2)
    result = check_schema([*annotations, annotations[0]])
    assert result.status == FAIL
    assert "duplicate" in result.summary


def test_a_mixed_landmark_count_fails():
    """A manifest with both 4- and 8-landmark rows silently changes the head's
    channel count depending on which row the loader saw first."""
    eight = corpus(sessions=1, per_session=2)
    four = build_scene(
        POSE, [], image_id="s0/sess0/img-four", setup_id="s0",
        session_id="s0/sess0", landmark_count=4,
    ).annotation

    result = check_schema([*eight, four])
    assert result.status == FAIL
    assert "mixed landmark counts" in result.summary


def test_an_id_that_disagrees_with_its_session_fails():
    """Image ids carry the session; the renderer and the loader both rely on
    that, so a disagreement is a broken group, not a cosmetic problem."""
    good = corpus(sessions=1, per_session=1)[0]
    bad = Annotation(
        image_id="somewhere/else/img-0", setup_id=good.setup_id,
        session_id=good.session_id, origin=good.origin,
        landmarks=good.landmarks, tips=good.tips, image_size=good.image_size,
    )
    assert check_schema([good, bad]).status == FAIL


def test_images_below_four_landmarks_are_counted_and_warned():
    """#13's cliff: these images cannot be scored at all, at any precision."""
    annotations = corpus(sessions=1, per_session=4)
    blinded = list(annotations[0].landmarks)
    for i in range(6):
        blinded[i] = None
    annotations[0] = Annotation(
        image_id=annotations[0].image_id, setup_id=annotations[0].setup_id,
        session_id=annotations[0].session_id, origin=annotations[0].origin,
        landmarks=tuple(blinded), tips=annotations[0].tips,
        image_size=annotations[0].image_size,
    )

    result = check_schema(annotations)
    assert result.status == WARN
    assert result.details["below_four_landmarks"] == 1
    assert result.details["below_four_landmarks_rate"] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def test_split_grouping_is_skipped_without_a_tier():
    assert check_split_grouping(corpus(), None).status == SKIPPED


def test_a_valid_split_passes_and_records_its_digest():
    annotations = corpus(sessions=4)
    assignment = split_for_tier(annotations, "session", seed=0)
    result = check_split_grouping(annotations, assignment)

    assert result.status == PASS
    assert result.details["digest"] == assignment.digest


def test_a_session_spanning_two_partitions_fails():
    """The single rule that prevents the most likely leak."""
    annotations = corpus(sessions=3)
    sound = build_session_split(annotations, ["set0/sess2"], [])
    spanning = SplitAssignment(
        tier=sound.tier,
        partitions=sound.partitions,
        sessions={"train": ("set0/sess0", "set1/sess1"),
                  "val": ("set0/sess2", "set1/sess1"), "test": ()},
        setups=sound.setups,
    )

    result = check_split_grouping(annotations, spanning)
    assert result.status == FAIL
    assert "set1/sess1" in result.summary


# --------------------------------------------------------------------------
# Near-duplicates
# --------------------------------------------------------------------------

def test_duplicate_checks_are_skipped_without_images():
    annotations = corpus()
    assert check_cross_session_duplicates(annotations, None).status == SKIPPED
    assert check_within_session_duplication(annotations, None).status == SKIPPED


def test_a_duplicate_inside_one_session_is_not_a_leak():
    annotations = corpus(sessions=2, per_session=2)
    same_session = [a.image_id for a in annotations if a.session_id == "set0/sess0"]
    pairs = [Pair(same_session[0], same_session[1], 1)]

    assert check_cross_session_duplicates(annotations, pairs).status == PASS


def test_a_duplicate_spanning_sessions_fails_and_names_the_merge():
    annotations = corpus(sessions=2, per_session=2)
    first = next(a for a in annotations if a.session_id == "set0/sess0")
    second = next(a for a in annotations if a.session_id == "set1/sess1")

    result = check_cross_session_duplicates(
        annotations, [Pair(first.image_id, second.image_id, 2)]
    )
    assert result.status == FAIL
    assert result.details["sessions_to_merge"] == [("set0/sess0", "set1/sess1")]


def test_effective_size_reports_the_smaller_number():
    annotations = corpus(sessions=1, per_session=4)
    ids = [a.image_id for a in annotations]
    clusters = [ids[:3], ids[3:]]      # four images, two moments

    result = check_within_session_duplication(annotations, clusters)
    assert result.status == WARN
    assert result.details["effective_images"] == 2
    assert result.details["effective_ratio"] == pytest.approx(0.5)
    assert result.details["largest_cluster_size"] == 3


def test_a_distinct_corpus_passes():
    annotations = corpus(sessions=1, per_session=4)
    clusters = [[a.image_id] for a in annotations]
    assert check_within_session_duplication(annotations, clusters).status == PASS


# --------------------------------------------------------------------------
# Balance
# --------------------------------------------------------------------------

def test_a_balanced_corpus_passes_on_darts_per_image():
    result = check_darts_per_image(corpus(sessions=4, per_session=4))
    assert result.status == PASS
    assert set(result.details["distribution"]) == {0, 1, 2, 3}


def test_a_corpus_of_only_full_boards_is_warned():
    """The empty board is the state the app is in before every visit."""
    rng = np.random.default_rng(0)
    annotations = [
        scene(f"s0/sess0/img-{i:03d}", [sample_tip_uniform(rng) for _ in range(3)])
        for i in range(8)
    ]
    result = check_darts_per_image(annotations)

    assert result.status == WARN
    assert result.details["distribution"][0] == 0
    assert result.details["total_darts"] == 24


# --------------------------------------------------------------------------
# Margins
# --------------------------------------------------------------------------

def test_near_wire_darts_make_the_corpus_able_to_measure_precision():
    result = check_margin_distribution(corpus(near_wire=True, per_session=4))
    assert result.status == PASS
    assert result.details["near_wire_share"] == pytest.approx(1.0)
    assert result.details["median_mm"] < 3.0


def test_a_corpus_with_no_near_wire_darts_is_warned_not_failed():
    """It is a sound corpus that cannot measure the thing that breaks scoring.
    That is a weakness, not a defect."""
    rng = np.random.default_rng(0)
    annotations = [
        scene(
            f"s0/sess0/img-{i:03d}",
            # Mid-bed, far from every wire: correct for a precise model and an
            # imprecise one alike.
            [(0.0, 120.0), (0.0, -120.0)],
        )
        for i in range(6)
    ]
    result = check_margin_distribution(annotations)

    assert result.status == WARN
    assert result.details["near_wire_share"] == 0.0
    assert result.details["min_mm"] > 3.0


def test_darts_on_an_uncalibratable_image_are_counted_as_unplaceable():
    good = corpus(sessions=1, per_session=4)
    blind = good[3]
    annotations = [
        *good[:3],
        Annotation(
            image_id=blind.image_id, setup_id=blind.setup_id,
            session_id=blind.session_id, origin=blind.origin,
            landmarks=(None,) * 8, tips=blind.tips, image_size=blind.image_size,
        ),
    ]
    result = check_margin_distribution(annotations)
    assert result.details["unplaceable_darts"] == len(blind.tips) > 0


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------

def test_the_report_takes_the_worst_status():
    annotations = corpus(sessions=3, per_session=4)
    report = audit(annotations, assignment=split_for_tier(annotations, "session"))

    assert report.status in (PASS, WARN)
    assert report.sound
    assert [c.name for c in report.checks] == [
        "schema", "split_grouping", "cross_session_duplicates",
        "effective_size", "darts_per_image", "margin_distribution",
    ]


def test_a_failing_check_makes_the_corpus_unsound():
    annotations = corpus(sessions=1, per_session=2)
    report = audit([*annotations, annotations[0]])

    assert report.status == FAIL
    assert not report.sound
    assert status_of(report, "schema") == FAIL


def test_checks_needing_images_are_skipped_not_failed():
    report = audit(corpus())
    assert status_of(report, "cross_session_duplicates") == SKIPPED
    assert status_of(report, "effective_size") == SKIPPED
    assert report.sound


# --------------------------------------------------------------------------
# End to end, with images on disk
# --------------------------------------------------------------------------

@pytest.fixture
def rendered(tmp_path):
    """A corpus with images, where each session is a burst of near-identical
    frames -- a smaller effective size, but no leak."""
    from PIL import Image

    from dartvision.data.labels import image_filename

    annotations = corpus(sessions=3, per_session=4)
    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(0)
    distinct = {
        a.session_id: rng.random((64, 64)) * 200
        for a in annotations
    }
    for index, annotation in enumerate(annotations):
        # Every frame in a session is a near-copy of that session's base frame.
        array = distinct[annotation.session_id] + rng.random((64, 64)) * 2.0
        Image.fromarray(array.astype("uint8")).save(
            root / image_filename(annotation.image_id)
        )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(annotations, manifest)
    return annotations, manifest, root


def test_a_bursty_corpus_reports_a_smaller_effective_size(rendered):
    pytest.importorskip("PIL")
    annotations, _, root = rendered
    report = audit(annotations, image_root=root)

    assert status_of(report, "effective_size") == WARN
    assert detail(report, "effective_size", "effective_images") < len(annotations)


def test_a_frame_shared_between_two_sessions_is_caught_end_to_end(rendered):
    """The leak that session-level grouping cannot survive. Copying one
    session's frame into another is exactly what a mis-sorted capture folder
    does, and nothing else in the pipeline would notice."""
    pytest.importorskip("PIL")
    from dartvision.data.labels import image_filename

    annotations, _, root = rendered
    source = next(a for a in annotations if a.session_id == "set0/sess0")
    target = next(a for a in annotations if a.session_id == "set1/sess1")
    (root / image_filename(target.image_id)).write_bytes(
        (root / image_filename(source.image_id)).read_bytes()
    )

    report = audit(annotations, image_root=root)
    assert status_of(report, "cross_session_duplicates") == FAIL
    assert not report.sound
    assert ("set0/sess0", "set1/sess1") in detail(
        report, "cross_session_duplicates", "sessions_to_merge"
    )


def test_missing_image_files_are_counted_not_fatal(rendered):
    pytest.importorskip("PIL")
    annotations, _, root = rendered
    for path in sorted(root.iterdir())[:2]:
        path.unlink()

    report = audit(annotations, image_root=root)
    assert detail(report, "cross_session_duplicates", "unhashed_images") == 2
    assert report.sound


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------

def test_cli_exits_zero_on_a_sound_corpus(rendered, tmp_path, capsys):
    from dartvision.audit.__main__ import main

    _, manifest, root = rendered
    out = tmp_path / "audit.json"
    code = main([
        "--manifest", str(manifest), "--image-root", str(root),
        "--split", "session", "--out", str(out),
    ])

    assert code == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["sound"] is True
    assert len(written["checks"]) == 6
    assert "margin_distribution" in capsys.readouterr().out


def test_cli_exits_nonzero_when_the_corpus_is_unsound(tmp_path, capsys):
    from dartvision.audit.__main__ import main

    annotations = corpus(sessions=1, per_session=2)
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl([*annotations, annotations[0]], manifest)

    assert main(["--manifest", str(manifest)]) == 1
    assert "not trustworthy" in capsys.readouterr().out


def test_cli_can_be_quiet(rendered, capsys):
    from dartvision.audit.__main__ import main

    _, manifest, _ = rendered
    assert main(["--manifest", str(manifest), "--quiet"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_reports_an_unbuildable_tier_as_a_failed_check(rendered, capsys):
    """A corpus with no real sessions cannot support sim-to-real. That is a
    finding about the corpus, not a crash -- and the other checks still run."""
    from dartvision.audit.__main__ import main

    _, manifest, _ = rendered
    assert main(["--manifest", str(manifest), "--split", "sim_to_real"]) == 1

    printed = capsys.readouterr().out
    assert "cannot build the 'sim_to_real' tier" in printed
    assert "margin_distribution" in printed


def test_a_split_construction_error_is_reported_without_an_assignment():
    result = check_split_grouping(corpus(), None, error="needs three setups")
    assert result.status == FAIL
    assert result.summary == "needs three setups"
