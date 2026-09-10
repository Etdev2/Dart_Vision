"""#14's verification checks, as a thing that can be run rather than remembered.

Section 7 of the leakage-safe benchmark lists checks a dataset must pass before
any number computed on it means anything, and hands them to #17 to execute. The
data-strategy decision retargeted them from DeepDarts to our own corpus:
DeepDarts' `d2_03_03_2020` anomaly is gone with the dataset, and near-duplicate
hashing now has to cover synthetic output as well as captures, because a
renderer sampling poses at random can draw two nearly identical cameras.

Each check reports a status rather than raising:

- ``fail``    -- the benchmark is not sound. A number computed here is not
                 trustworthy: sessions span splits, the schema is violated, or
                 a near-duplicate spans two sessions.
- ``warn``    -- sound but weak. The corpus cannot measure something #21 needs,
                 or is smaller than it looks.
- ``pass``    -- checked, nothing found.
- ``skipped`` -- not checkable from what was supplied (usually: no images).

A warning is not a nuisance to be silenced. "Almost no darts near a wire" is a
passing corpus by every other measure and is still unable to measure the thing
that actually breaks scoring.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from dartvision.data.labels import Annotation, image_filename
from dartvision.data.splits import LeakageError, SplitAssignment, validate_split
from dartvision.geometry.board import BDO_BOARD, BoardSpec, margin_to_nearest_boundary
from dartvision.geometry.calibration import estimate_homography

__all__ = [
    "CheckResult",
    "AuditReport",
    "check_schema",
    "check_split_grouping",
    "check_cross_session_duplicates",
    "check_within_session_duplication",
    "check_darts_per_image",
    "check_margin_distribution",
    "audit",
]

PASS, WARN, FAIL, SKIPPED = "pass", "warn", "fail", "skipped"
_SEVERITY = {SKIPPED: 0, PASS: 1, WARN: 2, FAIL: 3}

# A dart this close to a wire cannot be scored reliably at any precision the
# system will reach; #14 makes the distribution of these a first-class number.
NEAR_WIRE_MM = 3.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    summary: str
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "status": self.status,
            "summary": self.summary, "details": self.details,
        }


@dataclass(frozen=True)
class AuditReport:
    checks: tuple[CheckResult, ...]

    @property
    def status(self) -> str:
        """The worst status any check reached."""
        return max((c.status for c in self.checks), key=lambda s: _SEVERITY[s])

    @property
    def sound(self) -> bool:
        """Whether a number computed on this corpus can be trusted at all."""
        return self.status != FAIL

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "sound": self.sound,
            "checks": [c.to_dict() for c in self.checks],
        }


# --------------------------------------------------------------------------
# 1. Schema conformance and the calibration cliff
# --------------------------------------------------------------------------

def check_schema(annotations: Sequence[Annotation]) -> CheckResult:
    """Contract conformance, plus the count of images that cannot be scored.

    Parsing a manifest already enforces most of the contract -- ``Annotation``
    validates on construction. What survives that and still breaks training is
    structural: duplicate image ids, a landmark count that varies across the
    manifest (which would silently change the head's channel count), an id
    whose session prefix disagrees with its session field, or a point outside
    the frame.
    """
    if not annotations:
        return CheckResult("schema", FAIL, "the manifest is empty")

    ids = Counter(a.image_id for a in annotations)
    duplicates = sorted(i for i, n in ids.items() if n > 1)
    counts = Counter(len(a.landmarks) for a in annotations)
    mismatched = sorted(
        a.image_id for a in annotations
        if not a.image_id.startswith(f"{a.session_id}/")
    )
    out_of_frame = sorted(
        a.image_id for a in annotations
        if any(
            not (0.0 <= v <= 1.0)
            for point in (*a.landmarks, *a.tips) if point is not None
            for v in point
        )
    )
    no_size = sum(1 for a in annotations if a.image_size is None)
    uncalibratable = sorted(a.image_id for a in annotations if not a.is_calibratable)

    details: dict[str, object] = {
        "images": len(annotations),
        "landmark_counts": dict(sorted(counts.items())),
        "duplicate_image_ids": duplicates[:20],
        "id_session_mismatches": mismatched[:20],
        "points_outside_the_frame": out_of_frame[:20],
        "missing_image_size": no_size,
        # The cliff from #13: below four landmarks nothing in the image scores.
        "below_four_landmarks": len(uncalibratable),
        "below_four_landmarks_rate": round(len(uncalibratable) / len(annotations), 5),
        "below_four_examples": uncalibratable[:20],
    }

    problems = []
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate image ids")
    if len(counts) > 1:
        problems.append(f"mixed landmark counts {sorted(counts)}")
    if mismatched:
        problems.append(f"{len(mismatched)} ids disagree with their session")
    if problems:
        return CheckResult("schema", FAIL, "; ".join(problems), details)

    warnings = []
    if out_of_frame:
        warnings.append(f"{len(out_of_frame)} images have a point outside the frame")
    if uncalibratable:
        warnings.append(
            f"{len(uncalibratable)} of {len(annotations)} images "
            f"({100 * len(uncalibratable) / len(annotations):.2f}%) have fewer than "
            "four landmarks and cannot be scored at all"
        )
    if no_size:
        warnings.append(f"{no_size} images carry no image_size")
    if warnings:
        return CheckResult("schema", WARN, "; ".join(warnings), details)

    return CheckResult(
        "schema", PASS,
        f"{len(annotations)} images conform; all are calibratable", details,
    )


# --------------------------------------------------------------------------
# 2. No session spans splits
# --------------------------------------------------------------------------

def check_split_grouping(
    annotations: Sequence[Annotation],
    assignment: SplitAssignment | None,
    error: str | None = None,
) -> CheckResult:
    """#14's single most important rule: no session may span partitions.

    ``error`` carries a failure from *building* the split. A tier that cannot
    be constructed on this corpus is a real finding -- cross-setup needs three
    setups, sim-to-real needs both kinds of session -- and reporting it as a
    failed check keeps the rest of the audit running.
    """
    if error is not None:
        return CheckResult("split_grouping", FAIL, error)
    if assignment is None:
        return CheckResult(
            "split_grouping", SKIPPED,
            "no tier given; pass --split to check the grouping rule",
        )
    details = {
        "tier": assignment.tier,
        "digest": assignment.digest,
        "counts": assignment.counts(),
        "sessions": {k: list(v) for k, v in sorted(assignment.sessions.items())},
    }
    try:
        validate_split(annotations, assignment)
    except LeakageError as error:
        return CheckResult("split_grouping", FAIL, str(error), details)

    sessions = sum(len(v) for v in assignment.sessions.values())
    return CheckResult(
        "split_grouping", PASS,
        f"{sessions} sessions, none spanning partitions (digest {assignment.digest})",
        details,
    )


# --------------------------------------------------------------------------
# 3. Cross-session near-duplicates are a real leak
# --------------------------------------------------------------------------

def check_cross_session_duplicates(
    annotations: Sequence[Annotation], pairs, unhashed: int = 0
) -> CheckResult:
    """A near-duplicate spanning two sessions defeats session-level grouping.

    Grouping assumes sessions are independent. A frame that appears in two of
    them puts a sibling of a training image into the test set, and the number
    that comes out the other end is a within-session number wearing a
    cross-session label. The fix is to merge the affected sessions into one
    group, not to delete a frame.
    """
    if pairs is None:
        return CheckResult(
            "cross_session_duplicates", SKIPPED,
            "no images available to hash; pass --image-root",
        )
    session_of = {a.image_id: a.session_id for a in annotations}
    crossing = [
        p for p in pairs
        if session_of.get(p.left) != session_of.get(p.right)
    ]
    details = {
        "pairs_examined": len(pairs),
        "cross_session_pairs": len(crossing),
        "unhashed_images": unhashed,
        "sessions_to_merge": sorted(
            {
                tuple(sorted((session_of[p.left], session_of[p.right])))
                for p in crossing
            }
        )[:20],
        "examples": [
            {"left": p.left, "right": p.right, "distance": p.distance}
            for p in crossing[:10]
        ],
    }
    if crossing:
        return CheckResult(
            "cross_session_duplicates", FAIL,
            f"{len(crossing)} near-duplicate pairs span sessions; merge those "
            "sessions into one group before splitting",
            details,
        )
    return CheckResult(
        "cross_session_duplicates", PASS,
        "no near-duplicate frame spans two sessions", details,
    )


# --------------------------------------------------------------------------
# 4. Effective dataset size
# --------------------------------------------------------------------------

def check_within_session_duplication(
    annotations: Sequence[Annotation], clusters, minimum_ratio: float = 0.6
) -> CheckResult:
    """How much of the corpus is distinct. Not a leak -- an inflated epoch.

    Near-duplicates within a session are legitimate data, but ten views of one
    moment carry roughly one moment's information while costing ten forward
    passes. The effective size is the number to compare against an epoch
    budget, and the number that says whether "3,000 images" means anything.

    Clusters are formed across the whole corpus, not per session. When the
    cross-session check passes there is no difference; when it fails, the
    clusters show exactly which sessions have to be merged.
    """
    if clusters is None:
        return CheckResult(
            "effective_size", SKIPPED,
            "no images available to hash; pass --image-root",
        )
    nominal = len(annotations)
    effective = len(clusters)
    ratio = effective / nominal if nominal else 0.0
    largest = clusters[0] if clusters else []
    details = {
        "images": nominal,
        "effective_images": effective,
        "effective_ratio": round(ratio, 4),
        "clusters_larger_than_one": sum(1 for c in clusters if len(c) > 1),
        "largest_cluster_size": len(largest),
        "largest_cluster": largest[:10],
    }
    if ratio < minimum_ratio:
        return CheckResult(
            "effective_size", WARN,
            f"{nominal} images carry about {effective} images' worth of "
            f"information ({100 * ratio:.1f}%); epoch budgets and dataset-size "
            "claims should use the smaller number",
            details,
        )
    return CheckResult(
        "effective_size", PASS,
        f"{effective} of {nominal} images are distinct ({100 * ratio:.1f}%)",
        details,
    )


# --------------------------------------------------------------------------
# 5. Darts per image
# --------------------------------------------------------------------------

def check_darts_per_image(
    annotations: Sequence[Annotation], minimum_share: float = 0.05
) -> CheckResult:
    """The balance that matters for a detector: how many darts are in frame.

    A corpus that is almost entirely three-dart images teaches the tip head
    that three is the answer, and the empty board -- the state the app is in
    before every visit -- becomes the case it has never practised.
    """
    counts = Counter(len(a.tips) for a in annotations)
    total = len(annotations)
    distribution = {n: counts.get(n, 0) for n in range(4)}
    details = {
        "distribution": distribution,
        "shares": {n: round(c / total, 4) for n, c in distribution.items()} if total else {},
        "total_darts": sum(n * c for n, c in counts.items()),
    }
    thin = [n for n, c in distribution.items() if c / total < minimum_share] if total else []
    if thin:
        return CheckResult(
            "darts_per_image", WARN,
            "under-represented dart counts: "
            + ", ".join(f"{n} darts {100 * distribution[n] / total:.1f}%" for n in thin),
            details,
        )
    return CheckResult(
        "darts_per_image", PASS,
        "every dart count from 0 to 3 is represented", details,
    )


# --------------------------------------------------------------------------
# 6. Margin to the nearest boundary
# --------------------------------------------------------------------------

_BUCKETS = ((0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, math.inf))


def check_margin_distribution(
    annotations: Sequence[Annotation],
    board: BoardSpec = BDO_BOARD,
    minimum_near_wire_share: float = 0.05,
) -> CheckResult:
    """The distribution #14 calls first-class, and #21 cannot gate without.

    A dart three millimetres inside the treble bed is scored correctly by a
    model with three millimetres of error and by one with none. Only darts near
    a wire distinguish them, so a corpus with almost no near-wire darts reports
    a high accuracy that says nothing about the failure mode that costs points.
    """
    margins: list[float] = []
    unscoreable = 0
    for annotation in annotations:
        if not annotation.tips:
            continue
        if not annotation.is_calibratable:
            unscoreable += len(annotation.tips)
            continue
        try:
            homography = estimate_homography(annotation.landmarks, board=board)
            for x, y in homography.to_board(annotation.tips):
                margins.append(margin_to_nearest_boundary(float(x), float(y), board))
        except ValueError:
            unscoreable += len(annotation.tips)

    if not margins:
        return CheckResult(
            "margin_distribution", WARN,
            "no dart in the corpus could be placed on the board",
            {"darts": 0, "unplaceable_darts": unscoreable},
        )

    ordered = sorted(margins)
    buckets = {
        f"{low:g}-{high:g}mm" if high != math.inf else f">{low:g}mm":
        sum(1 for m in margins if low <= m < high)
        for low, high in _BUCKETS
    }
    near_wire = sum(1 for m in margins if m < NEAR_WIRE_MM)
    share = near_wire / len(margins)
    details = {
        "darts": len(margins),
        "unplaceable_darts": unscoreable,
        "buckets_mm": buckets,
        "near_wire_darts": near_wire,
        "near_wire_share": round(share, 4),
        "median_mm": round(_percentile(ordered, 50), 3),
        "p05_mm": round(_percentile(ordered, 5), 3),
        "min_mm": round(ordered[0], 3),
    }
    if share < minimum_near_wire_share:
        return CheckResult(
            "margin_distribution", WARN,
            f"only {100 * share:.1f}% of darts land within {NEAR_WIRE_MM:g} mm of a "
            "wire; accuracy measured here cannot distinguish a precise model from "
            "an imprecise one",
            details,
        )
    return CheckResult(
        "margin_distribution", PASS,
        f"{100 * share:.1f}% of darts land within {NEAR_WIRE_MM:g} mm of a wire "
        f"(median margin {details['median_mm']} mm)",
        details,
    )


def _percentile(ordered: Sequence[float], percent: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percent / 100.0
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


# --------------------------------------------------------------------------
# Running the lot
# --------------------------------------------------------------------------

def audit(
    annotations: Sequence[Annotation],
    image_root: str | Path | None = None,
    assignment: SplitAssignment | None = None,
    hash_threshold: int = 5,
    board: BoardSpec = BDO_BOARD,
    split_error: str | None = None,
) -> AuditReport:
    """Run every check #14 §7 asks for, in one pass over the corpus."""
    pairs = clusters = None
    unhashed = 0
    if image_root is not None:
        from dartvision.audit.hashing import cluster, hash_images, near_duplicate_pairs

        root = Path(image_root)
        hashes = hash_images(
            {a.image_id: root / image_filename(a.image_id) for a in annotations}
        )
        unhashed = len(annotations) - len(hashes)
        pairs = near_duplicate_pairs(hashes, threshold=hash_threshold)
        clusters = cluster(hashes, pairs)

    return AuditReport(
        checks=(
            check_schema(annotations),
            check_split_grouping(annotations, assignment, split_error),
            check_cross_session_duplicates(annotations, pairs, unhashed),
            check_within_session_duplication(annotations, clusters),
            check_darts_per_image(annotations),
            check_margin_distribution(annotations, board),
        )
    )
