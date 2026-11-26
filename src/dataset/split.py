"""Held-out split of the Trompa pages by piece (Analysis 005, Exp 035, Exp 038).

Pages of one piece share their engraving and musical content, so they are kept on the same side:
a piece is the file name without its trailing page number.
"""

import random
import re
from collections.abc import Sequence
from pathlib import Path

from .coco import CocoMetadata


def piece_name(file_name: str) -> str:
    return re.sub(r"_\d+$", "", Path(file_name).stem)


def held_out_pieces(pieces: Sequence[str], seed: int, fraction: float) -> set[str]:
    """Same split as Analysis 005: sorted pieces, shuffled with the seed, first fraction held out."""
    order = list(range(len(pieces)))
    random.Random(seed).shuffle(order)
    num_holdout = max(1, round(fraction * len(pieces)))
    return {pieces[k] for k in order[:num_holdout]}


def split_by_piece(
    images: Sequence[CocoMetadata], seed: int, fraction: float
) -> tuple[list[CocoMetadata], list[CocoMetadata]]:
    """(train, held-out) pages; with all pages as input, the held-out pieces are exactly those
    of the Exp 035 refiner experiments for the same seed and fraction."""
    pieces = sorted({piece_name(img.file_name) for img in images})
    holdout = held_out_pieces(pieces, seed, fraction)
    train = [img for img in images if piece_name(img.file_name) not in holdout]
    held_out = [img for img in images if piece_name(img.file_name) in holdout]
    return train, held_out
