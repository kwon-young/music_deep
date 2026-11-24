# SYNTHETIC.md — On-the-fly synthetic detection dataset

Handoff / status doc for the synthetic music-score page generator. A future
session should read this (and `EXPERIMENTS.md`) before continuing.

## Goal

A drop-in synthetic source for the OMR detector (same 91 classes as Trompa-COCO:
83 symbol boxes + 8 line keypoints) whose pages are generated **on the fly during
training**. The point is to remove the shape/position/size **bias** of the real
data (symbols always on staves, constrained sizes, fixed positions) by placing
objects anywhere, at any size, with exact ground truth by construction — a strong
supervised pretraining stage before fine-tuning on real Trompa.

Design rationale is discussed in the repo (see `DETECTION.md`, `FINAL_DETECTION.md`,
`LINE.md`, `AUGMENT.md`, `PRETRAINING.md`). No training-code changes yet; the
generator exposes the **same loader interface as `dataset.coco`** so it can be
dropped into `train_detection.py` at the `transform_image` seam.

## Status (current)

- [x] Self-contained module `src/dataset/synthetic.py` (Pillow + numpy only, no
      nvimagecodec/pyvips needed at runtime).
- [x] Same interface as coco: `load_synthetic_sample(dataset, index, device)`
      returns `Data[CocoMetadata, DetectionSample[TensorImage[CHW,RGB,Int255], ...]]`
      with absolute-pixel boxes (xyxy) and keypoints (x1y1x2y2), mirroring
      `load_coco_sample`. Category ids/names match Trompa-COCO exactly.
- [x] Preview script `scripts/visualize_synthetic.py` (raw + GT-overlay PNGs).
- [x] Benchmark script `scripts/benchmark_synthetic_vs_trompa.py`.
- [x] Tests `tests/dataset/test_synthetic.py` (8 cases, unittest, pass).
- [ ] NOT wired into `train_detection.py` yet (see Next steps).

## Implementation summary

### Glyph atlas (`GlyphAtlas`)
- Source: verovio repo (`DEFAULT_VEROVIO_ROOT = /home/kwon-young/prog/verovio`).
- name → codepoint from `fonts/<Font>/<font>_metadata.json` (`optionalGlyphs`) +
  `data/<Font>.xml` (standard glyphs).
- Each glyph pre-rendered once per font at nominal size, ink-cropped to an RGBA
  tile (80 of the 83 symbol classes are direct SMuFL glyphs; `dots` = circles,
  `slur`/`tie` = quadratic Bézier (3 points) drawn on a transparent tile).
- Fonts: `Bravura` (default), `Leland`, `Leipzig`, `Gootville` all supported.

### Page generation (`SyntheticPageGenerator`)
- Canvas = numpy `uint8 (H,W,3)`, light background + optional Gaussian noise.
- **Sizes decoupled from any page zoom**: each object samples its own staff
  space `ss` log-uniform in `[staff_space_min, staff_space_max]`; every page mixes
  tiny and huge objects. Symbol height = `symbol_height_ss × ss` (log-uniform);
  line length = per-class multiplier × `ss`.
- **No rotation** (dropped for speed), **no placement retries** (single random
  position; a rejected object is simply dropped).
- **Occlusion control** (`OcclusionTracker`): every placed ink rect is recorded;
  a placement is accepted only if no single existing object is covered (or
  covers) more than `max_pair` of the smaller rect, and total coverage of the
  candidate stays below `max_total`. Symbols use tight thresholds; lines use a
  loose pair-only check (`line_max_total_overlap=None`) because long thin lines
  naturally cross symbols.
- **Numpy compositing**: symbols alpha-blended with vectorized uint16 math;
  lines/dots rasterized with distance/disc masks (`_draw_segment_np`,
  `_draw_disk_np`, `_draw_dashes_np`). Pillow only for the atlas and the
  per-object LANCZOS resize.
- **Lines are single segments** (incl. `staff` — one line per annotation,
  matching real Trompa annotations), any orientation, thickness from
  `engravingDefaults` × `ss`.

### Current defaults (`SyntheticConfig`)
```
page_size=2048  staff_space_min=24  staff_space_max=96
objects_per_page=240  lines_per_page=200
symbol_height_ss_min=0.5  symbol_height_ss_max=3.0
fonts=('Bravura',)  background=247  noise=2.0  ink_min=5  ink_max=45  margin=4
symbol_max_pair_overlap=0.25  symbol_max_total_overlap=0.40
line_max_pair_overlap=0.50   line_max_total_overlap=None
seed=0
```

### Verified behavior
- All 83 symbol classes reachable (uniform class sampling); all 8 line classes
  produced. Line lengths now realistic: staff ~600px, system ~730px, stem
  ~160px, ledger ~150px (on 2048² pages).
- Occlusion bound holds on real pages (worst symbol pair-overlap = 0.25).
- Seeded reproducibility (`test_reproducible_with_seed`).
- mypy clean on the module, preview script, benchmark, and tests (Pillow ships
  its own stubs, `py.typed`).

## Performance (Quadro M1200; nvimgcodec TIFF decode FAILS on this Maxwell GPU)

From `scripts/benchmark_synthetic_vs_trompa.py`:
- Real Trompa full page (6300×8910, 56 MP): pyvips decode + to_float1 = **445 ms**,
  shared patch tail = **1180 ms** → ~1.6 s/sample.
- Synthetic: 2048² generate = **422 ms**, tail = **117 ms** → ~0.54 s/sample.
  (Before dropping rotation/retries + numpy compositing it was 791 ms.)

So per-sample synthetic is ~3× cheaper than real at 2048²; matching real
resolution (8192²) is ~3.0 s generate + ~1.2 s tail (slower). 2048² is the
recommended sample size; both paths get variance-dropped to a similar effective
patch count.

## How to run

```bash
# preview (writes synthetic_preview/page_XXX.png + _gt.png)
PYTHONPATH=src mamba run -n pytorch python scripts/visualize_synthetic.py --num_pages 3

# tests
PYTHONPATH=src mamba run -n pytorch python -m unittest tests.dataset.test_synthetic

# benchmark
PYTHONPATH=src mamba run -n pytorch python scripts/benchmark_synthetic_vs_trompa.py --iters 3
```

## Next steps

1. **Wire into training** (the plan agreed earlier): in `train_detection.py`,
   add a `--synthetic` path that (a) refactors the `transform_image` tail
   (`to_float1` → morphological → affine) into a shared helper, (b) adds
   `create_synthetic_detection_iterator` mirroring the existing iterator but
   calling `load_synthetic_sample` instead of `load_coco_sample` + decode, and
   (c) keeps `parse_coco(real_anno)` for the 91-class structure + weights, with
   the LR-budget `dataset_symbols` overridden by a `--synth_epoch_symbols`.
   The synthetic path bypasses nvimagecodec/pyvips entirely (pure numpy/torch,
   CPU), which is also a decode-speed win.
2. **Optimization if needed**: cache per-glyph scaled sprites (the LANCZOS
   resize is now the dominant cost), or a spatial-hash grid for the occlusion
   scan.
3. **Experiment**: pretrain on synthetic (supervised detection) → fine-tune on
   real Trompa → compare to Exp 029 baseline (symbols 0.919 / lines 0.827);
   watch the known bottlenecks (`octave`, `tuplet*`, rare accidentals). Log the
   run in `EXPERIMENTS.md`.
4. Optional: synthetic pages as an extra corpus for the Dense-LeJEPA SSL path
   (Exp 030).

## Open questions / knobs

- `lines_per_page=200` was bumped to compensate for single-attempt rejections;
  if long-line density is still low, raise `line_max_pair_overlap` (0.5 → 0.7)
  or add per-class retries.
- Aggressive decorrelation was chosen (any orientation lines, uniform class
  sampling, sizes decoupled). A "realistic" variant (axis-aligned staff/system,
  symbols near lines) could be a later knob.
