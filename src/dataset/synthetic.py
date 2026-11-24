"""Synthetic music-score page generation for detector pretraining.

Self-contained companion to ``dataset.coco``: it renders fresh full-page music
scores from SMuFL glyphs (via Pillow) with exact ground-truth boxes (symbols)
and keypoint lines (lines), exposing the same loader interface as
``load_coco_sample`` so it can later be dropped into the training pipeline.

The class names and ids below mirror the real Trompa-COCO categories exactly,
so the synthetic output is a drop-in *virtual twin* of the real dataset.
"""

from __future__ import annotations

import json
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dataset.coco import CocoMetadata
from music_types import (
    Absolute,
    BoxDim,
    BoundingBoxes,
    CHW,
    ClassLabels,
    Data,
    DetectionSample,
    Int255,
    KeypointDim,
    Keypoints,
    NumBoxes,
    NumKeypoints,
    NumLineClasses,
    NumSymbolClasses,
    RGB,
    TensorImage,
    TopLeft,
    X1Y1X2Y2,
    XYXY,
)

DEFAULT_VEROVIO_ROOT = Path("/home/kwon-young/prog/verovio")

_FONT_FILES: dict[str, str] = {
    "Bravura": "Bravura.otf",
    "Leland": "Leland.otf",
    "Leipzig": "Leipzig.ttf",
    "Gootville": "Gootville.otf",
}

# (COCO id, name) for the 83 symbol classes, matching Trompa-COCO order (by id).
SYMBOL_CATEGORIES: tuple[tuple[int, str], ...] = (
    (1, "accidentalDoubleFlat"),
    (2, "accidentalDoubleSharp"),
    (3, "accidentalFlat"),
    (4, "accidentalNatural"),
    (5, "accidentalNaturalFlat"),
    (6, "accidentalNaturalSharp"),
    (7, "accidentalParensLeft"),
    (8, "accidentalParensRight"),
    (9, "accidentalSharp"),
    (10, "articAccentAbove"),
    (11, "articAccentBelow"),
    (12, "articMarcatoAbove"),
    (13, "articStaccatissimoWedgeBelow"),
    (14, "articStaccatoAbove"),
    (15, "articStaccatoBelow"),
    (16, "articTenutoAbove"),
    (17, "articTenutoBelow"),
    (20, "brace"),
    (21, "breathMarkComma"),
    (22, "dots"),
    (23, "dynamicFF"),
    (24, "dynamicForte"),
    (25, "dynamicFortePiano"),
    (26, "dynamicMF"),
    (27, "dynamicMP"),
    (28, "dynamicPP"),
    (29, "dynamicPPP"),
    (30, "dynamicPiano"),
    (31, "dynamicRinforzando1"),
    (32, "dynamicSforzando1"),
    (33, "dynamicSforzandoPiano"),
    (34, "fClef"),
    (35, "fClefChange"),
    (36, "fermataAbove"),
    (37, "fermataBelow"),
    (38, "flag16thDown"),
    (39, "flag16thUp"),
    (40, "flag32ndDown"),
    (41, "flag32ndUp"),
    (42, "flag64thDown"),
    (43, "flag64thUp"),
    (44, "flag8thDown"),
    (45, "flag8thUp"),
    (46, "gClef"),
    (47, "gClefChange"),
    (48, "keyboardPedalPed"),
    (49, "keyboardPedalUp"),
    (51, "noteheadBlack"),
    (52, "noteheadHalf"),
    (53, "noteheadWhole"),
    (55, "ornamentShortTrill"),
    (56, "ornamentTrill"),
    (57, "ornamentTurn"),
    (58, "ornamentZigZagLineNoRightEnd"),
    (59, "ornamentZigZagLineWithRightEnd"),
    (60, "rest128th"),
    (61, "rest16th"),
    (62, "rest32nd"),
    (63, "rest64th"),
    (64, "rest8th"),
    (65, "restHalf"),
    (66, "restQuarter"),
    (67, "restWhole"),
    (68, "slur"),
    (72, "tie"),
    (381, "timeSig1"),
    (382, "timeSig2"),
    (383, "timeSig3"),
    (384, "timeSig4"),
    (385, "timeSig6"),
    (386, "timeSig8"),
    (387, "timeSig9"),
    (388, "timeSigCommon"),
    (389, "timeSigCutCommon"),
    (390, "tuplet1"),
    (391, "tuplet2"),
    (392, "tuplet3"),
    (393, "tuplet5"),
    (394, "tuplet6"),
    (395, "tuplet7"),
    (396, "tuplet8"),
    (397, "tuplet9"),
    (399, "wiggleArpeggiatoUp"),
)

# (COCO id, name) for the 8 line classes, mirroring Trompa-COCO order (by id).
LINE_CATEGORIES: tuple[tuple[int, str], ...] = (
    (18, "barLine"),
    (19, "beam"),
    (50, "ledgerLines"),
    (54, "octave"),
    (69, "staff"),
    (70, "stem"),
    (71, "system"),
    (398, "voltaBracket"),
)

SYMBOL_NAMES: tuple[str, ...] = tuple(name for _, name in SYMBOL_CATEGORIES)
LINE_NAMES: tuple[str, ...] = tuple(name for _, name in LINE_CATEGORIES)

# Symbol classes that are NOT rasterized from a font glyph.
_SPECIAL_SYMBOLS: frozenset[str] = frozenset({"dots", "slur", "tie"})
_GLYPH_SYMBOLS: tuple[str, ...] = tuple(
    name for name in SYMBOL_NAMES if name not in _SPECIAL_SYMBOLS
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Distribution knobs for the synthetic page generator."""

    page_size: int = 2048
    staff_space_min: float = 24.0
    staff_space_max: float = 96.0
    objects_per_page: int = 240
    lines_per_page: int = 200
    symbol_height_ss_min: float = 0.5
    symbol_height_ss_max: float = 3.0
    fonts: tuple[str, ...] = ("Bravura",)
    background: int = 247
    noise: float = 2.0
    ink_min: int = 5
    ink_max: int = 45
    margin: int = 4
    symbol_max_pair_overlap: float = 0.25
    symbol_max_total_overlap: float = 0.40
    line_max_pair_overlap: float = 0.50
    line_max_total_overlap: float | None = None
    seed: int = 0


@dataclass(frozen=True)
class GlyphSprite:
    image: Image.Image
    ink_width: int
    ink_height: int
    font_index: int
    name: str


class GlyphAtlas:
    """Maps glyph names to pre-rendered ink-cropped RGBA sprites, per font."""

    def __init__(
        self,
        verovio_root: Path = DEFAULT_VEROVIO_ROOT,
        fonts: tuple[str, ...] = ("Bravura",),
        nominal_size: int = 512,
    ) -> None:
        self._root = verovio_root
        self._fonts = fonts
        self._nominal_size = nominal_size
        self._codepoints: list[dict[str, int]] = [
            self._load_codepoints(font) for font in fonts
        ]
        self._defaults: dict[str, float] = self._load_defaults(fonts[0])
        self._font_cache: list[ImageFont.FreeTypeFont | None] = [None] * len(fonts)
        self._sprite_cache: dict[str, list[GlyphSprite | None]] = {}

    @property
    def num_fonts(self) -> int:
        return len(self._fonts)

    def thickness(self, key: str, default: float) -> float:
        return self._defaults.get(key, default)

    def sprite(self, name: str, font_index: int) -> GlyphSprite:
        """Returns the sprite for ``name``, trying ``font_index`` then font 0."""
        for idx in (font_index, 0):
            if name in self._codepoints[idx]:
                cached = self._sprite_cache.get(name)
                if cached is not None:
                    entry = cached[idx]
                    if entry is not None:
                        return entry
                return self._build_sprite(name, idx)
        raise ValueError(
            f"glyph {name!r} not present in any font of {self._fonts}"
        )

    def prerender(self, names: tuple[str, ...]) -> None:
        """Builds every sprite up front. FreeType fonts are not thread-safe, so
        an atlas shared by worker threads must be fully rendered before use."""
        for name in names:
            for idx in range(self.num_fonts):
                if name in self._codepoints[idx]:
                    self.sprite(name, idx)

    def _load_codepoints(self, font: str) -> dict[str, int]:
        cp: dict[str, int] = {}
        meta_path = self._root / "fonts" / font / f"{font.lower()}_metadata.json"
        meta = json.loads(meta_path.read_text())
        for name, info in meta.get("optionalGlyphs", {}).items():
            codepoint = info.get("codepoint")
            if isinstance(codepoint, str) and codepoint.startswith("U+"):
                cp[name] = int(codepoint[2:], 16)
        xml_path = self._root / "data" / f"{font}.xml"
        if xml_path.exists():
            tree = ET.parse(xml_path)
            for g in tree.getroot().findall("g"):
                n = g.get("n")
                c = g.get("c")
                if n is not None and c is not None:
                    cp[n] = int(c, 16)
        return cp

    def _load_defaults(self, font: str) -> dict[str, float]:
        meta_path = self._root / "fonts" / font / f"{font.lower()}_metadata.json"
        meta = json.loads(meta_path.read_text())
        out: dict[str, float] = {}
        for key, value in meta.get("engravingDefaults", {}).items():
            if isinstance(value, (int, float)):
                out[key] = float(value)
        return out

    def _load_font(self, index: int) -> ImageFont.FreeTypeFont:
        cached = self._font_cache[index]
        if cached is not None:
            return cached
        font_file = self._root / "fonts" / self._fonts[index] / _FONT_FILES[self._fonts[index]]
        loaded = ImageFont.truetype(str(font_file), self._nominal_size)
        self._font_cache[index] = loaded
        return loaded

    def _build_sprite(self, name: str, index: int) -> GlyphSprite:
        font = self._load_font(index)
        n = self._nominal_size
        canvas = Image.new("RGBA", (n * 3, n * 3), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text(
            (n * 1.5, n * 1.5), chr(self._codepoints[index][name]), font=font,
            fill=(0, 0, 0, 255), anchor="mm",
        )
        bbox = canvas.getbbox()
        if bbox is None:
            sprite = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
            ink_w = ink_h = 1
        else:
            sprite = canvas.crop(bbox)
            ink_w = bbox[2] - bbox[0]
            ink_h = bbox[3] - bbox[1]
        entry = GlyphSprite(
            image=sprite, ink_width=ink_w, ink_height=ink_h,
            font_index=index, name=name,
        )
        self._sprite_cache.setdefault(name, [None] * self.num_fonts)[index] = entry
        return entry


@dataclass
class SyntheticPage:
    image: np.ndarray
    boxes: list[list[float]]
    box_labels: list[int]
    keypoints: list[list[float]]
    keypoint_labels: list[int]
    keypoint_thicknesses: list[float]
    width: int
    height: int


@dataclass(frozen=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def _intersection_area(a: Rect, b: Rect) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if w <= 0.0 or h <= 0.0:
        return 0.0
    return w * h


class OcclusionTracker:
    """Tracks placed ink rects to reject placements that would occlude too much.

    A candidate is accepted when no single existing object covers (or is covered
    by) more than ``max_pair`` of the smaller rect's area, and the total covered
    fraction of the candidate stays below ``max_total``.
    """

    def __init__(self) -> None:
        self._rects: list[Rect] = []

    @property
    def count(self) -> int:
        return len(self._rects)

    def accepts(self, r: Rect, max_pair: float, max_total: float | None) -> bool:
        area = r.area
        if area <= 0.0:
            return False
        total = 0.0
        for o in self._rects:
            inter = _intersection_area(r, o)
            if inter <= 0.0:
                continue
            if inter / min(area, o.area) > max_pair:
                return False
            total += inter
        if max_total is None:
            return True
        return total / area <= max_total

    def add(self, r: Rect) -> None:
        self._rects.append(r)


class SyntheticPageGenerator:
    """Renders fresh full-page scores with exact ground truth.

    The canvas is a numpy uint8 array; symbols are alpha-composited and lines /
    dots rasterized with vectorized numpy ops (no rotation, single placement
    attempt per object: rejected objects are simply dropped).
    """

    def __init__(
        self,
        atlas: GlyphAtlas,
        config: SyntheticConfig = SyntheticConfig(),
    ) -> None:
        self.atlas = atlas
        self.config = config
        self._base_seed = config.seed
        self._counter = 0
        self._canvas_np: np.ndarray = np.zeros(
            (config.page_size, config.page_size, 3), dtype=np.uint8
        )

    def generate(self) -> SyntheticPage:
        seed = self._base_seed + self._counter
        self._counter += 1
        rng = random.Random(seed)
        nprng = np.random.default_rng(seed)
        cfg = self.config
        size = cfg.page_size

        bg = np.full((size, size, 3), cfg.background, dtype=np.float32)
        if cfg.noise > 0:
            bg += nprng.normal(0.0, cfg.noise, (size, size, 1))
        self._canvas_np = np.clip(bg, 0, 255).astype(np.uint8)

        ink_val = rng.randrange(cfg.ink_min, cfg.ink_max + 1)
        ink: tuple[int, int, int] = (ink_val, ink_val, ink_val)

        boxes: list[list[float]] = []
        box_labels: list[int] = []
        keypoints: list[list[float]] = []
        keypoint_labels: list[int] = []
        keypoint_thicknesses: list[float] = []
        self._occupied = OcclusionTracker()
        sym_pair, sym_total = cfg.symbol_max_pair_overlap, cfg.symbol_max_total_overlap

        num_symbols = int(cfg.objects_per_page * rng.uniform(0.7, 1.3))
        for _ in range(num_symbols):
            sym_idx = rng.randrange(len(SYMBOL_NAMES))
            name = SYMBOL_NAMES[sym_idx]
            ss = self._sample_staff_space(rng)
            box: list[float] | None
            if name == "dots":
                box = self._draw_dots(rng, ss, ink, sym_pair, sym_total)
            elif name in ("slur", "tie"):
                tile, iw, ih = self._bezier_tile(rng, name, ss)
                size_ss = math.exp(rng.uniform(math.log(0.4), math.log(2.0)))
                box = self._paste_tile(rng, tile, round(size_ss * ss), sym_pair, sym_total)
            else:
                font_idx = rng.randrange(self.atlas.num_fonts)
                sprite = self.atlas.sprite(name, font_idx)
                size_ss = math.exp(
                    rng.uniform(
                        math.log(cfg.symbol_height_ss_min),
                        math.log(cfg.symbol_height_ss_max),
                    )
                )
                box = self._paste_tile(
                    rng, sprite.image, round(size_ss * ss), sym_pair, sym_total
                )
            if box is not None:
                boxes.append(box)
                box_labels.append(sym_idx)

        for _ in range(cfg.lines_per_page):
            line_idx = rng.randrange(len(LINE_NAMES))
            name = LINE_NAMES[line_idx]
            ss = self._sample_staff_space(rng)
            for kp, thickness in self._render_line(rng, name, ss, ink):
                keypoints.append(kp)
                keypoint_labels.append(line_idx)
                keypoint_thicknesses.append(thickness)

        return SyntheticPage(
            image=self._canvas_np,
            boxes=boxes,
            box_labels=box_labels,
            keypoints=keypoints,
            keypoint_labels=keypoint_labels,
            keypoint_thicknesses=keypoint_thicknesses,
            width=size,
            height=size,
        )

    # --- placement helpers -------------------------------------------------

    def _sample_staff_space(self, rng: random.Random) -> float:
        """Per-object staff space (log-uniform), decoupled from the page."""
        return math.exp(
            rng.uniform(
                math.log(self.config.staff_space_min),
                math.log(self.config.staff_space_max),
            )
        )

    def _sample_center(
        self, rng: random.Random, mx: float, my: float
    ) -> tuple[float, float] | None:
        lo_x, hi_x = mx, self.config.page_size - mx
        lo_y, hi_y = my, self.config.page_size - my
        if hi_x <= lo_x or hi_y <= lo_y:
            return None
        return (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))

    def _paste_tile(
        self,
        rng: random.Random,
        tile: Image.Image,
        target_h: int,
        max_pair: float,
        max_total: float,
    ) -> list[float] | None:
        """Resize an RGBA ink tile, alpha-composite it, return the ink bbox."""
        w, h = tile.size
        if w <= 0 or h <= 0:
            return None
        scale = target_h / h
        nw = max(1, round(w * scale))
        nh = max(1, round(h * scale))
        if nw > self.config.page_size - 2 * self.config.margin or nh > self.config.page_size - 2 * self.config.margin:
            return None
        scaled = tile.resize((nw, nh), Image.Resampling.LANCZOS)
        center = self._sample_center(
            rng, nw / 2 + self.config.margin, nh / 2 + self.config.margin
        )
        if center is None:
            return None
        left = round(center[0] - nw / 2)
        top = round(center[1] - nh / 2)
        rect = Rect(float(left), float(top), float(left + nw), float(top + nh))
        if not self._occupied.accepts(rect, max_pair, max_total):
            return None
        self._composite_rgba(np.asarray(scaled), left, top)
        self._occupied.add(rect)
        return [rect.x1, rect.y1, rect.x2, rect.y2]

    def _composite_rgba(self, rgba: np.ndarray, x: int, y: int) -> None:
        """Alpha-blends an (H, W, 4) uint8 RGBA tile onto the numpy canvas."""
        h, w = rgba.shape[0], rgba.shape[1]
        region = self._canvas_np[y : y + h, x : x + w]
        rgb = rgba[..., :3].astype(np.uint16)
        alpha = rgba[..., 3:4].astype(np.uint16)
        inv = np.uint16(255) - alpha
        out = (rgb * alpha + region.astype(np.uint16) * inv + np.uint16(127)) // np.uint16(255)
        region[...] = out.astype(np.uint8)

    # --- symbol renderers --------------------------------------------------

    def _draw_dots(
        self,
        rng: random.Random,
        ss: float,
        ink: tuple[int, int, int],
        max_pair: float,
        max_total: float,
    ) -> list[float] | None:
        num = rng.choice((1, 2, 2))
        r = ss * rng.uniform(0.10, 0.20)
        spacing = 3.0 * r
        total_w = (num - 1) * spacing + 2 * r
        center = self._sample_center(
            rng, total_w / 2 + r + self.config.margin, r + self.config.margin
        )
        if center is None:
            return None
        left = center[0] - total_w / 2
        top = center[1] - r
        rect = Rect(left, top, left + total_w, top + 2 * r)
        if not self._occupied.accepts(rect, max_pair, max_total):
            return None
        for i in range(num):
            self._draw_disk_np(left + r + i * spacing, top + r, r, ink)
        self._occupied.add(rect)
        return [left, top, left + total_w, top + 2 * r]

    def _bezier_tile(
        self, rng: random.Random, kind: str, ss: float
    ) -> tuple[Image.Image, int, int]:
        """Quadratic bezier (3 points) on a transparent tile; returns ink image."""
        t = 512
        tile = Image.new("RGBA", (t, t), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        x1 = t * rng.uniform(0.05, 0.25)
        x2 = t * rng.uniform(0.75, 0.95)
        y0 = t * rng.uniform(0.35, 0.65)
        curv = t * rng.uniform(0.08, 0.35)
        yc = y0 - curv if kind == "slur" else y0 + curv
        width = max(1, round(self.atlas.thickness("slurMidpointThickness", 0.22) * ss))
        points = _quad_bezier_points((x1, y0), ((x1 + x2) / 2, yc), (x2, y0), 48)
        td.line(points, fill=(0, 0, 0, 255), width=width, joint="curve")
        bbox = tile.getbbox()
        if bbox is None:
            return Image.new("RGBA", (2, 2), (0, 0, 0, 0)), 1, 1
        ink_img = tile.crop(bbox)
        return ink_img, bbox[2] - bbox[0], bbox[3] - bbox[1]

    # --- line renderers ----------------------------------------------------

    def _render_line(
        self,
        rng: random.Random,
        name: str,
        ss: float,
        ink: tuple[int, int, int],
    ) -> list[tuple[list[float], float]]:
        """Draws one line object; returns its (start, end) keypoints and thickness.

        Endpoints follow the Trompa-COCO convention: start is the top end of a
        mostly vertical line, the left end otherwise. Real lines are vertical or
        within ±20° of horizontal, so the flip at 45° never splits real data.
        """
        cfg = self.config
        length = self._line_length(rng, name, ss)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        thickness = self._line_thickness(rng, name, ss)
        half = self._clamp_half(length / 2, thickness)
        if half is None:
            return []
        center = self._sample_center(
            rng, half + thickness / 2 + self.config.margin, half + thickness / 2 + self.config.margin
        )
        if center is None:
            return []
        x1 = center[0] - half * math.cos(angle)
        y1 = center[1] - half * math.sin(angle)
        x2 = center[0] + half * math.cos(angle)
        y2 = center[1] + half * math.sin(angle)
        rect = self._segment_rect(x1, y1, x2, y2, thickness)
        if not self._occupied.accepts(
            rect, cfg.line_max_pair_overlap, cfg.line_max_total_overlap
        ):
            return []
        if name == "octave":
            self._draw_dashes_np((x1, y1), (x2, y2), ss, thickness, ink)
        elif name == "voltaBracket":
            self._draw_segment_np(x1, y1, x2, y2, thickness, ink)
            hook = ss * 0.9
            hx, hy = -math.sin(angle), math.cos(angle)
            self._draw_segment_np(x2, y2, x2 + hx * hook, y2 + hy * hook, thickness, ink)
        else:
            self._draw_segment_np(x1, y1, x2, y2, thickness, ink)
        self._occupied.add(rect)
        steep = abs(y2 - y1) > abs(x2 - x1)
        if (steep and y2 < y1) or (not steep and x2 < x1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        return [([x1, y1, x2, y2], float(thickness))]

    def _segment_rect(
        self, x1: float, y1: float, x2: float, y2: float, thickness: int
    ) -> Rect:
        pad = thickness / 2
        return Rect(
            min(x1, x2) - pad,
            min(y1, y2) - pad,
            max(x1, x2) + pad,
            max(y1, y2) + pad,
        )

    def _clamp_half(
        self, requested: float, thickness: float
    ) -> float | None:
        """Clamps a half-length so the segment fits inside the page; None if none fits."""
        budget = (
            self.config.page_size / 2.0 - thickness / 2.0 - self.config.margin - 1.0
        )
        if budget <= 0.0:
            return None
        return min(requested, budget)

    def _line_length(self, rng: random.Random, name: str, ss: float) -> float:
        if name == "staff":
            return ss * rng.uniform(12.0, 30.0)
        if name == "stem":
            return ss * rng.uniform(2.5, 7.0)
        if name == "beam":
            return ss * rng.uniform(4.0, 12.0)
        if name == "system":
            return ss * rng.uniform(10.0, 24.0)
        if name == "barLine":
            return ss * rng.uniform(4.0, 8.0)
        if name == "ledgerLines":
            return ss * rng.uniform(2.5, 5.0)
        if name == "octave":
            return ss * rng.uniform(8.0, 20.0)
        return ss * rng.uniform(6.0, 12.0)  # voltaBracket

    def _line_thickness(self, rng: random.Random, name: str, ss: float) -> int:
        at = self.atlas
        if name == "staff":
            return max(1, round(at.thickness("staffLineThickness", 0.13) * ss))
        if name == "stem":
            return max(1, round(at.thickness("stemThickness", 0.12) * ss))
        if name == "beam":
            return max(1, round(at.thickness("beamThickness", 0.5) * ss * rng.uniform(0.8, 1.6)))
        if name == "system":
            return max(1, round(at.thickness("thinBarlineThickness", 0.16) * ss))
        if name == "barLine":
            thin = at.thickness("thinBarlineThickness", 0.16) * ss
            thick = at.thickness("thickBarlineThickness", 0.5) * ss
            return max(1, round(rng.choice((thin, thick))))
        if name == "ledgerLines":
            return max(1, round(at.thickness("legerLineThickness", 0.16) * ss))
        if name == "octave":
            return max(1, round(at.thickness("octaveLineThickness", 0.16) * ss))
        return max(1, round(at.thickness("repeatEndingLineThickness", 0.16) * ss))  # voltaBracket

    def _draw_segment_np(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        thickness: int,
        ink: tuple[int, int, int],
    ) -> None:
        """Rasterizes a thick segment onto the numpy canvas via a distance mask."""
        half = thickness / 2.0
        size = self.config.page_size
        x0 = max(0, math.floor(min(x1, x2) - half - 1))
        x1b = min(size, math.ceil(max(x1, x2) + half + 1))
        y0 = max(0, math.floor(min(y1, y2) - half - 1))
        y1b = min(size, math.ceil(max(y1, y2) + half + 1))
        if x1b <= x0 or y1b <= y0:
            return
        yy, xx = np.mgrid[y0:y1b, x0:x1b]
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 <= 0.0:
            return
        t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / l2, 0.0, 1.0)
        px = x1 + t * dx
        py = y1 + t * dy
        mask = (xx - px) ** 2 + (yy - py) ** 2 <= half * half
        region = self._canvas_np[y0:y1b, x0:x1b]
        region[mask] = ink

    def _draw_disk_np(
        self, cx: float, cy: float, radius: float, ink: tuple[int, int, int]
    ) -> None:
        size = self.config.page_size
        x0 = max(0, math.floor(cx - radius - 1))
        x1b = min(size, math.ceil(cx + radius + 1))
        y0 = max(0, math.floor(cy - radius - 1))
        y1b = min(size, math.ceil(cy + radius + 1))
        if x1b <= x0 or y1b <= y0:
            return
        yy, xx = np.mgrid[y0:y1b, x0:x1b]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
        region = self._canvas_np[y0:y1b, x0:x1b]
        region[mask] = ink

    def _draw_dashes_np(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        ss: float,
        width: int,
        ink: tuple[int, int, int],
    ) -> None:
        total = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if total <= 0:
            return
        ux = (p2[0] - p1[0]) / total
        uy = (p2[1] - p1[1]) / total
        dash = ss * 0.35
        gap = ss * 0.2
        t = 0.0
        while t < total:
            t_end = min(t + dash, total)
            self._draw_segment_np(
                p1[0] + ux * t, p1[1] + uy * t,
                p1[0] + ux * t_end, p1[1] + uy * t_end,
                width, ink,
            )
            t += dash + gap


def _quad_bezier_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    samples: int = 48,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for i in range(samples + 1):
        t = i / samples
        u = 1.0 - t
        x = u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0]
        y = u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1]
        points.append((x, y))
    return points


@dataclass
class SyntheticDataset:
    generator: SyntheticPageGenerator
    num_symbol_classes: int
    num_line_classes: int
    symbol_cat_id_to_idx: dict[int, int]
    line_cat_id_to_idx: dict[int, int]
    symbol_categories: list[dict[str, object]]
    line_categories: list[dict[str, object]]
    symbol_weights: list[float]
    line_weights: list[float]

    @property
    def num_symbols(self) -> int:
        return 0


def create_synthetic_dataset(
    config: SyntheticConfig = SyntheticConfig(),
    verovio_root: Path = DEFAULT_VEROVIO_ROOT,
) -> SyntheticDataset:
    """Builds the atlas + dataset. ``num_symbols`` is 0 (no static image set)."""
    atlas = GlyphAtlas(verovio_root=verovio_root, fonts=config.fonts)
    generator = SyntheticPageGenerator(atlas, config)

    symbol_cat_id_to_idx = {cid: i for i, (cid, _) in enumerate(SYMBOL_CATEGORIES)}
    line_cat_id_to_idx = {cid: i for i, (cid, _) in enumerate(LINE_CATEGORIES)}
    symbol_categories = [
        {"id": cid, "name": name, "supercategory": "symbol", "keypoints": ["origin"]}
        for cid, name in SYMBOL_CATEGORIES
    ]
    line_categories = [
        {"id": cid, "name": name, "supercategory": "line", "keypoints": ["start", "end"]}
        for cid, name in LINE_CATEGORIES
    ]

    return SyntheticDataset(
        generator=generator,
        num_symbol_classes=len(SYMBOL_CATEGORIES),
        num_line_classes=len(LINE_CATEGORIES),
        symbol_cat_id_to_idx=symbol_cat_id_to_idx,
        line_cat_id_to_idx=line_cat_id_to_idx,
        symbol_categories=symbol_categories,
        line_categories=line_categories,
        symbol_weights=[1.0] * len(SYMBOL_CATEGORIES),
        line_weights=[1.0] * len(LINE_CATEGORIES),
    )


type SyntheticSample = Data[
    CocoMetadata,
    DetectionSample[
        TensorImage[CHW, RGB, Int255],
        BoundingBoxes[tuple[NumBoxes, BoxDim], XYXY, Absolute, TopLeft],
        ClassLabels[tuple[NumBoxes], NumSymbolClasses],
        Keypoints[tuple[NumKeypoints, KeypointDim], X1Y1X2Y2, Absolute, TopLeft],
        ClassLabels[tuple[NumKeypoints], NumLineClasses],
    ],
]


def load_synthetic_sample(
    dataset: SyntheticDataset,
    index: int,
    device: torch.device,
) -> SyntheticSample:
    """Loads a freshly generated synthetic sample (same shape as load_coco_sample)."""
    return page_to_sample(dataset.generator.generate(), index, device)


def page_to_sample(
    page: SyntheticPage, index: int, device: torch.device
) -> SyntheticSample:
    """Converts a generated page to the ``load_coco_sample`` layout."""
    arr = np.ascontiguousarray(page.image.transpose(2, 0, 1))
    img = torch.from_numpy(arr).to(device)
    boxes = torch.tensor(page.boxes, dtype=torch.float32, device=device).reshape(-1, 4)
    box_labels = torch.tensor(page.box_labels, dtype=torch.int64, device=device)
    keypoints = torch.tensor(page.keypoints, dtype=torch.float32, device=device).reshape(-1, 4)
    keypoint_labels = torch.tensor(page.keypoint_labels, dtype=torch.int64, device=device)

    meta = CocoMetadata(
        img_id=index,
        file_name=f"synthetic_{index:06d}.png",
        width=page.width,
        height=page.height,
    )
    return Data(
        metadata=meta,
        sample=DetectionSample(
            image=TensorImage(img),
            boxes=BoundingBoxes(boxes),
            box_labels=ClassLabels(box_labels),
            keypoints=Keypoints(keypoints),
            keypoint_labels=ClassLabels(keypoint_labels),
        ),
    )
