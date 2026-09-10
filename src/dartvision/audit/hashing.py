"""Perceptual hashing, for finding near-duplicate frames.

#14 wants two different things out of this, and they are not the same question.

A near-duplicate pair *spanning two sessions* is a leak: session-level grouping
assumes sessions are independent, and a frame that appears in two of them puts a
sibling of a training image into the test set. That is a hard failure.

A near-duplicate pair *inside* one session is not a leak -- whole sessions move
together -- but it inflates epoch length without adding information. Counting
those gives the effective dataset size, which is the number that should be
compared against #20's epoch budget rather than the raw file count.

The hash is a difference hash: greyscale, 9x8, compare each pixel with its
right-hand neighbour, one bit per comparison. It is deliberately not a
cryptographic hash -- the whole point is that similar images collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "HASH_BITS",
    "dhash",
    "hash_images",
    "hamming",
    "near_duplicate_pairs",
    "cluster",
]

HASH_BITS = 64


def dhash(path: str | Path, size: int = 8) -> int:
    """Difference hash of the image at ``path``, as a ``size * size``-bit int."""
    from PIL import Image

    with Image.open(path) as handle:
        grey = handle.convert("L").resize((size + 1, size), Image.BILINEAR)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_images(paths: dict[str, Path], size: int = 8) -> dict[str, int]:
    """Hash a mapping of id to path, skipping ids whose file is missing."""
    return {
        key: dhash(path, size) for key, path in paths.items() if Path(path).exists()
    }


def hamming(left: int, right: int) -> int:
    """Number of differing bits."""
    return bin(left ^ right).count("1")


def _bit_matrix(values: Sequence[int], bits: int = HASH_BITS) -> np.ndarray:
    packed = np.array(
        [[(v >> shift) & 1 for shift in range(bits - 1, -1, -1)] for v in values],
        dtype=np.int16,
    )
    return packed


@dataclass(frozen=True)
class Pair:
    """Two ids whose images are within ``distance`` bits of each other."""

    left: str
    right: str
    distance: int


def near_duplicate_pairs(
    hashes: dict[str, int], threshold: int = 5, chunk: int = 512
) -> list[Pair]:
    """Every pair of ids within ``threshold`` bits, closest first.

    Brute force, in row chunks. A metric-tree index would scale further, but at
    the corpus sizes this project will reach (thousands, not millions) the
    chunked matrix product is both faster and much easier to trust.
    """
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    keys = sorted(hashes)
    if len(keys) < 2:
        return []

    bits = _bit_matrix([hashes[k] for k in keys])
    complement = 1 - bits

    pairs: list[Pair] = []
    for start in range(0, len(keys), chunk):
        stop = min(start + chunk, len(keys))
        block = bits[start:stop]
        # Hamming distance as two matrix products: ones where they are zero,
        # plus zeros where they are one.
        distances = block @ complement.T + (1 - block) @ bits.T
        rows, columns = np.nonzero(distances <= threshold)
        for row, column in zip(rows, columns):
            index = start + int(row)
            other = int(column)
            if other <= index:      # upper triangle only, and never self
                continue
            pairs.append(
                Pair(keys[index], keys[other], int(distances[row, other]))
            )
    pairs.sort(key=lambda p: (p.distance, p.left, p.right))
    return pairs


def cluster(keys: Iterable[str], pairs: Sequence[Pair]) -> list[list[str]]:
    """Group ids into transitive near-duplicate clusters.

    Transitivity is the right choice even though perceptual similarity is not
    transitive: a burst of twenty frames where each is close only to its
    neighbours is still twenty views of one moment, and counting it as one
    cluster is what makes the effective-size number honest.
    """
    parent: dict[str, str] = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for pair in pairs:
        if pair.left not in parent or pair.right not in parent:
            continue
        a, b = find(pair.left), find(pair.right)
        if a != b:
            parent[a] = b

    groups: dict[str, list[str]] = {}
    for key in parent:
        groups.setdefault(find(key), []).append(key)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))
