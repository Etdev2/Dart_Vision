"""Leakage-safe splits, and the discipline that keeps a holdout a holdout.

#14 established that the failure which invalidates every downstream number
while looking perfect is leakage: near-duplicate frames from one session
appearing on both sides of a split. The defence is structural rather than
statistical -- **group by session, and never let a session span a boundary** --
so this module refuses to build a split that violates it rather than warning
about it afterwards.

The tiers follow the revision in ``docs/architecture/data-strategy-decision.md``:

* ``synthetic``   -- train and evaluate on synthetic only. A pipeline sanity
  check, never a capability claim.
* ``sim_to_real`` -- train on synthetic, evaluate on real. The transfer gap.
* ``cross_setup`` -- hold out a whole physical setup. Unseen-board
  generalization, on data we own.
* ``holdout``     -- the untouchable set, read once at the end.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from dartvision.data.labels import Annotation, Origin

__all__ = [
    "Partition",
    "SplitAssignment",
    "LeakageError",
    "build_session_split",
    "build_cross_setup_split",
    "build_sim_to_real_split",
    "validate_split",
    "HoldoutLedger",
]

TRAIN, VAL, TEST = "train", "val", "test"
Partition = str


class LeakageError(ValueError):
    """A split would place one session, or one setup, on both sides."""


@dataclass(frozen=True)
class SplitAssignment:
    """Which partition every image belongs to, plus enough provenance to pin it.

    ``digest`` is what a training run records. Two runs quoting different
    digests are not comparable and must not appear in the same table.
    """

    tier: str
    partitions: dict[str, Partition]
    sessions: dict[Partition, tuple[str, ...]]
    setups: dict[Partition, tuple[str, ...]]

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {"tier": self.tier, "partitions": dict(sorted(self.partitions.items()))},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def counts(self) -> dict[Partition, int]:
        counts: dict[str, int] = {}
        for partition in self.partitions.values():
            counts[partition] = counts.get(partition, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier,
            "digest": self.digest,
            "counts": self.counts(),
            "sessions": {k: list(v) for k, v in sorted(self.sessions.items())},
            "setups": {k: list(v) for k, v in sorted(self.setups.items())},
            "partitions": dict(sorted(self.partitions.items())),
        }

    def write(self, path: str | Path) -> str:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return self.digest


def _group(annotations: Sequence[Annotation]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """session -> setup, and session -> image ids."""
    setup_of: dict[str, str] = {}
    images: dict[str, list[str]] = {}
    for a in annotations:
        if setup_of.setdefault(a.session_id, a.setup_id) != a.setup_id:
            raise LeakageError(
                f"session {a.session_id!r} appears under two setups "
                f"({setup_of[a.session_id]!r} and {a.setup_id!r}); session ids must be "
                "unique across setups or grouping cannot be trusted"
            )
        images.setdefault(a.session_id, []).append(a.image_id)
    return setup_of, images


def _assemble(
    tier: str,
    annotations: Sequence[Annotation],
    by_partition: dict[Partition, Iterable[str]],
) -> SplitAssignment:
    setup_of, images = _group(annotations)
    partitions: dict[str, str] = {}
    sessions: dict[str, tuple[str, ...]] = {}
    setups: dict[str, tuple[str, ...]] = {}

    for partition, session_ids in by_partition.items():
        ordered = tuple(sorted(set(session_ids)))
        sessions[partition] = ordered
        setups[partition] = tuple(sorted({setup_of[s] for s in ordered}))
        for session_id in ordered:
            for image_id in images[session_id]:
                partitions[image_id] = partition

    assignment = SplitAssignment(
        tier=tier, partitions=partitions, sessions=sessions, setups=setups
    )
    validate_split(annotations, assignment)
    return assignment


def build_session_split(
    annotations: Sequence[Annotation],
    val_sessions: Sequence[str],
    test_sessions: Sequence[str],
    tier: str = "synthetic",
) -> SplitAssignment:
    """Hold out named sessions. Everything else trains."""
    _, images = _group(annotations)
    val, test = set(val_sessions), set(test_sessions)
    unknown = (val | test) - set(images)
    if unknown:
        raise LeakageError(f"unknown sessions: {sorted(unknown)}")

    return _assemble(
        tier,
        annotations,
        {
            TRAIN: [s for s in images if s not in val and s not in test],
            VAL: sorted(val),
            TEST: sorted(test),
        },
    )


def build_cross_setup_split(
    annotations: Sequence[Annotation],
    holdout_setup: str,
    val_sessions: Sequence[str] = (),
) -> SplitAssignment:
    """Hold out one entire physical setup -- the unseen-board measurement.

    Requires at least two other setups to train on: a model trained on a single
    setup and evaluated on one other produces a number with no notion of
    variance, which #14 warns is a point estimate rather than a claim.
    """
    setup_of, images = _group(annotations)
    setups = set(setup_of.values())
    if holdout_setup not in setups:
        raise LeakageError(f"unknown setup {holdout_setup!r}; have {sorted(setups)}")
    if len(setups) < 3:
        raise LeakageError(
            f"cross-setup evaluation needs at least 3 setups (2 to train on, 1 held "
            f"out); got {len(setups)}. See #26 -- this is why the capture plan "
            "requires three."
        )

    val = set(val_sessions)
    held = {s for s, setup in setup_of.items() if setup == holdout_setup}
    if val & held:
        raise LeakageError("validation sessions cannot come from the held-out setup")

    return _assemble(
        "cross_setup",
        annotations,
        {
            TRAIN: [s for s in images if s not in held and s not in val],
            VAL: sorted(val),
            TEST: sorted(held),
        },
    )


def build_sim_to_real_split(
    annotations: Sequence[Annotation], val_sessions: Sequence[str] = ()
) -> SplitAssignment:
    """Train on synthetic, evaluate on real. The transfer gap (#25)."""
    setup_of, images = _group(annotations)
    origin_of: dict[str, Origin] = {}
    for a in annotations:
        if origin_of.setdefault(a.session_id, a.origin).is_real != a.origin.is_real:
            raise LeakageError(
                f"session {a.session_id!r} mixes synthetic and real images; a session "
                "must be wholly one or the other for the transfer gap to mean anything"
            )

    synthetic = [s for s in images if not origin_of[s].is_real]
    real = [s for s in images if origin_of[s].is_real]
    if not synthetic or not real:
        raise LeakageError("sim-to-real needs both synthetic and real sessions")

    val = set(val_sessions)
    return _assemble(
        "sim_to_real",
        annotations,
        {
            TRAIN: [s for s in synthetic if s not in val],
            VAL: sorted(val & set(synthetic)),
            TEST: sorted(real),
        },
    )


def validate_split(
    annotations: Sequence[Annotation], assignment: SplitAssignment
) -> None:
    """Refuse a split that leaks. Called by every builder; call it on loaded ones too."""
    setup_of, images = _group(annotations)

    seen: dict[str, str] = {}
    for partition, session_ids in assignment.sessions.items():
        for session_id in session_ids:
            if session_id in seen:
                raise LeakageError(
                    f"session {session_id!r} is in both {seen[session_id]!r} and "
                    f"{partition!r}; near-duplicate frames would appear on both sides"
                )
            seen[session_id] = partition

    missing = set(images) - set(seen)
    if missing:
        raise LeakageError(f"sessions assigned to no partition: {sorted(missing)}")

    unassigned = {a.image_id for a in annotations} - set(assignment.partitions)
    if unassigned:
        raise LeakageError(f"{len(unassigned)} images assigned to no partition")

    if assignment.tier == "cross_setup":
        train_setups = set(assignment.setups.get(TRAIN, ()))
        test_setups = set(assignment.setups.get(TEST, ()))
        overlap = train_setups & test_setups
        if overlap:
            raise LeakageError(
                f"setup(s) {sorted(overlap)} appear in both train and test; a "
                "cross-setup split must hold out whole setups"
            )


@dataclass
class HoldoutLedger:
    """Mechanical test-set discipline.

    #14 requires that the test set be read once per candidate, and that this be
    enforced rather than intended. The ledger records every read of a
    ``(checkpoint, split)`` pair and refuses a repeat without an explicit,
    logged override. A test set read fifty times during tuning is a validation
    set, and the project has no holdout.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def _entries(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def reads(self, checkpoint: str, split_digest: str) -> int:
        return sum(
            1
            for e in self._entries()
            if e["checkpoint"] == checkpoint and e["split_digest"] == split_digest
        )

    def record(
        self, checkpoint: str, split_digest: str, *, override: bool = False,
        reason: str | None = None,
    ) -> None:
        previous = self.reads(checkpoint, split_digest)
        if previous and not override:
            raise LeakageError(
                f"test set {split_digest} has already been read {previous}x for "
                f"checkpoint {checkpoint}. Reading it again turns the holdout into a "
                "validation set. Pass override=True with a reason if this is "
                "deliberate -- it will be logged."
            )
        if override and not reason:
            raise ValueError("an override must state a reason; it goes in the ledger")

        entries = self._entries()
        entries.append({
            "checkpoint": checkpoint,
            "split_digest": split_digest,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "override": bool(override),
            "reason": reason,
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
