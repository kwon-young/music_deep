# Patch-Set Assignment: One-to-One per Patch, One-to-Many across Patches

## Goal

Use the detector's many predictions (~32k queries per modality per page for at most ~3k GT
objects) as **evidence** instead of discarding all but one per object, while making the
crowded case (distinct objects closer than the spread of duplicates) and the missed case
structurally safe, not threshold-dependent.

## Why the current setup is the wrong fit

* The head has no decoder: every patch emits K = 5 slots from one MLP on one token
  (`src/model/detector.py`, `SymbolHead` / `LineHead`). Nothing coordinates slots of
  **different** patches.
* The global one-to-one matcher (`src/model/matcher.py`, greedy on cls + L1 (+ GIoU) cost
  over all P×K queries) gives each GT to one query and sets every equally good neighbour to
  background (`DFINECriterion.loss_labels`). Which patch wins is close to arbitrary.
* Evidence:
  * Exp 037: 90% of the staff segments it found scored < 0.3. Refine → NMS → noisy-OR takes
    staff from 0.525 to 0.963 AP without changing detection (Exp 035 on 037).
  * Exp 032: a quality (OKS) target on the single matched query made ranking worse,
    because its neighbours stayed at 0. Its conclusion asked for soft targets on every
    query near the segment.
  * Analysis 003: 77% of staff segments are assigned to a patch in the left 40% of the
    segment, so the far endpoint is predicted from far away (median 4–5 px, p90 17–24 px).
* NMS is now part of the pipeline anyway, so training for one-to-one pays its cost without
  its benefit.

## The design in one paragraph

Each patch is responsible for the GT objects in a small **responsibility region** around
its cell. Within a patch, its K slots are matched one-to-one to those objects (a set
prediction, which a single token can learn because the slots are computed jointly).
Across patches nothing is suppressed: every patch that covers an object predicts it. Each
slot answers two separate questions with two outputs:
* a binary **presence** logit: does this slot predict an object at all? (target 1 for every
  voter, 0 for every other slot);
* the per-class sigmoid **class score**, trained with quality targets: which class, and how
  well placed? (target = IoU / OKS of its own prediction).

At inference, predictions are **associated** into clusters rather than suppressed. A
cluster never holds two predictions of the same patch (**cannot-link**), because the patch
was trained to say that its slots are distinct objects. Its geometry fuses its members, per
endpoint or edge, weighted by certainty. Its existence evidence is the sum of its members'
presence divided by the number of patches that should have voted (**support ratio**). It is
ranked by support × class score.

How each failure mode is handled:

| failure | how it is handled |
|---|---|
| two crowded objects merged | they share at least one responsibility region; that patch emits both, and cannot-link keeps them in two clusters |
| one object split in two | only possible if one patch duplicates it in two slots (trained against in-patch); the split-off cluster has support ≈ 1/N and ranks low |
| object missed | only possible if no covering patch has a free slot; this ceiling is computed from GT alone in Phase 0, and overflow priority goes to the objects with the fewest voters |
| long-line far endpoint | each endpoint is fused from the voters closest to it |
| length bias of vote counting | support is presence votes / expected voters, not noisy-OR |
| weak evidence confused with poor placement | presence says "an object is here", the class score says "and this well placed": a far voter of a staff line has presence ≈ 1 and a low class score, an empty slot has both low |
| merge radius tuning | cannot-link makes a generous radius safe (Phase 0 measures how safe) |

Terminology: a **voter** of GT g is a (patch, slot) matched to g; **expected voters**
N(g) are the kept patches whose responsibility region contains g; an **occupied** slot is
one whose presence is above the candidate floor.

## Detailed design

### 1. Responsibility regions (training-time, from GT)

All in patch units (invariant to the 54–74 px extraction resized to 64).

* Patch cell C_p: the unit square centred on the patch centre.
* Symbol g → patch p iff dist(C_p, box_g) ≤ m. Line g → p iff dist(C_p, segment_g) ≤ m.
* **Ignore band**: m < dist ≤ m + δ. Neither positive nor negative (see 3).
* Only kept patches count (blank patches are dropped by `variance_patch_drop`).
* **Overflow** (more than K responsible GT of a modality in one patch): keep the K with the
  fewest expected voters (ties: nearest to the patch centre); the rest become *ignored* in
  that patch. A staff line with 40 voters can lose one, a staccato dot with 2 cannot.
* **Chosen (Phase 0, Analysis 009): m = 0.5, δ = 0.5, K = 8 for both heads.**
* **Coincident GT are kept, not merged**: the 505 same-class symbol pairs closer than 1 px
  are legitimate. A notehead shared by two voices (one stem up, one stem down) is two
  overlapping notes typographically, and the detector must output both; their accidentals,
  dots, rests and dynamics are duplicated the same way (324 `noteheadBlack`, 85 accidentals,
  36 dynamics, 16 `dots`, 16 `slur`, 13 `noteheadHalf`, 10 rests, 5 clef changes). The
  design handles them without special cases: a patch responsible for both assigns them to two
  slots (both positive, in-patch one-to-one is over slots, not positions), and at inference
  the two predictions of that patch cannot be linked, so they seed two clusters at the same
  place. A geometric NMS could never keep both.

### 2. Within-patch assignment

* Per patch: cost[K slots × R_p responsible GT] = focal cls cost + L1 (+ GIoU for symbols),
  as in `_match_modality`, plus the quality term that Exp 032 asked for (−IoU / −OKS).
* Solve it exactly on the GPU without loops: with K = 5 there are at most P(5, r) ≤ 120
  injective maps, so enumerate them as a fixed index tensor, gather, sum, argmin, all batched
  over patches. With K = 8, use a vectorized greedy over sorted costs instead (P(8,8) = 40,320 is too many).
* Output keeps the `MatchIndices` shape (pred_indices, target_indices), now with repeated
  targets, plus a per-query `quality` and an `ignore` mask, in a new typed dataclass.

### 3. Losses (`DFINECriterion`)

Every slot is positive, negative or ignored, and the presence and class losses share that
labelling:

* **Positive**: matched to a responsible GT in 2.
* **Ignored**: unmatched, and its prediction has q ≥ q_ignore against a GT that is in its
  ignore band or overflowed in its patch, **and** that GT is not already matched by another
  slot of the same patch.
* **Negative**: every other slot. In-patch duplicates are therefore always negatives, which
  is what makes cannot-link valid.

Losses:

* **Presence** (new, one logit per slot): binary sigmoid focal loss, target 1 on positives,
  0 on negatives, no loss on ignored slots. Not class-weighted (it is class-agnostic), but
  weighted per GT like the regression terms below, so long lines do not dominate it.
* **Class score**: varifocal / quality focal loss on the per-class sigmoids, as now but with
  soft targets. Positive slot: target q = IoU (symbol) or OKS (line) of its detached
  prediction against its GT, on its GT's class; 0 on the other classes. Negative slot: 0 on
  every class with focal down-weighting. Ignored slot: no loss. The Exp 029 class weights
  apply here as before.
* Why both: with the class score alone, a far voter of a long line (real object, q ≈ 0.1)
  and an empty slot look the same, so the support ratio cannot tell weak evidence from poor
  placement. Presence separates them, and its targets are 0/1, so its floor is
  interpretable (≈ 0.5) instead of the current heuristic 0.05 on the class score.
* **Every factor of the ranking score is trained on negatives** (both presence and the
  class score have target 0 there). This is the GFL lesson (FCOS centerness, trained only on
  positives, can be high on background at inference) and Exp 035 A+'s ("quality only"
  collapsed staff). Consequence: never multiply in an extra quality predictor trained only
  on positives or on kept queries (e.g. the Phase A+ refiner quality head) unless it is
  gated by presence, and never rank by it alone. The class score already carries quality,
  so multiplying by a separate quality would count it twice.
* **Regression** (L1, GIoU, FGL): on every positive, weighted by 1 / N_voters(g), so every GT
  weighs the same as before and the Exp 029 class balance (size-difficulty weights 0.5 / 8.0)
  keeps its meaning. Otherwise a staff line (~30–80 voters) outweighs a dot (~2) by 20×.
* **Normalizer**: number of GT (as now), not number of positives.
* `line_oks` already exists in `src/model/refiner.py` (σ = 0.1, evaluator formula); the
  line OKS area is not in the parsed cache (Exp 035 A+ read it from the JSON), so use
  length × class thickness as Exp 032 did, or add `area` to the cache.

### 4. Refiner in the training step

* `refiner_batch` takes all voters as real queries, so there are many more queries per step.
  Subsample up to n_r voters per GT (e.g. 4, spread along the segment), plus one denoising
  query per GT as now. Cost stays close to Exp 038's.
* The 1.5-patch coarse-error mask stays. Far voters of long lines will be masked more often,
  so log the masked fraction.

### 5. Inference: association instead of suppression

Replaces `deduplicate_lines` (kept for comparison) with `associate_detections`, used for
lines **and** symbols:

```
candidates = occupied slots (presence ≥ floor, ≈ 0.5)   (lines: refined first)
label      = argmax of the class sigmoids
per label:
    for pred in candidates by decreasing class score:
        open = clusters within distance d of pred
               that contain no prediction from pred.patch     # cannot-link
        if open: join the nearest, update its fused geometry
        else:    start a new cluster
    per cluster:
        geometry = per-coordinate weighted mean, weight = class score × certainty
                   (certainty = inverse variance of that coordinate's FGL distribution;
                    Softer-NMS variance voting; the class score is the quality estimate)
        support  = Σ member presence / N_expected(geometry)   # existence evidence, in [0, 1]
        quality  = max member class score                      # best placement estimate
        score    = support × quality                           # ranking
```

* Each signal has one job:

  | used for | signal |
  |---|---|
  | candidate floor, which slots are occupied (cannot-link) | presence |
  | support ratio (existence evidence) | Σ presence / N_expected |
  | fusion weights (geometry) | class score × FGL certainty |
  | cluster ranking | support × class score |

* The floor on presence also makes the candidate count, and so the refiner's cost,
  predictable: presence should come out close to 0 or 1 (checked in Phase 2).
* `quality = max member class score` is the first choice; alternatives to compare in
  Phase 4: certainty-weighted mean of the members' class scores, mean of the top 3.
* Lines: distance = max of the two endpoint distances (as `deduplicate_lines`); symbols:
  1 − IoU or centre distance relative to size.
* N_expected is computed from the fused geometry with the same region rule as training,
  over the page's kept patches.
* **Certainty-aware line distance** (Analysis 010): a voter in the middle of a long staff
  segment cannot see its ends (median segment ≈ 16 patches), so its far endpoint is a guess.
  With a plain max-endpoint distance it would never join the cluster of the voters that do
  see the end, and would start a fragment cluster instead (today: thousands of staff fragments
  per dense page survive NMS). Use a per-endpoint distance normalised by the FGL spread of that
  endpoint (Mahalanobis-like), so an uncertain far endpoint can join, and let the fusion
  weights (certainty) take the endpoint from the voters near it.
* Upgrade path if the greedy order causes wrong links: per-patch joint linking (match a
  patch's predictions to nearby clusters with the same enumerated assignment as training).
* Variant to compare: cluster on coarse geometry → fuse → refine the fused line once (far
  cheaper than refining every candidate, which one-to-many multiplies).
* Final: top `max_lines` / `max_symbols` by cluster score, as now. COCO AP is per class, so
  scores only need to be calibrated within a class.

## Phases

### Phase 0 — GT statistics (Analysis 009, offline, no model)

Script `scripts/assignment_stats.py` over the 351 trainval pages, at extraction sizes 54,
64 and 74 px, for each (m, K):

1. **Overflow**: distribution of responsible GT per kept patch, per modality; fraction of GT
   ignored in at least one patch; fraction of GT with **zero** non-overflowed voters
   (= recall ceiling).
2. **Vote budget**: expected voters per GT, per class and per size bin (dots, ledger lines,
   staff).
3. **Cannot-link coverage**: for same-class GT pairs within distance d (d sweep), the
   fraction that share at least one responsible patch where neither is overflowed. This is
   the fraction of crowded pairs the structure protects; list the classes and layouts of
   the unprotected ones.

**Gate**: choose m, δ, K such that the zero-voter rate is < 0.05% of GT, ≥ 99% of GT have ≥ 2
voters, and cannot-link coverage is ≥ 99% at the chosen d. If lines overflow at K = 5 (a
patch on a staff sees 3–5 staff lines + stem + beam + ledger lines), raise the line K
before anything else.

**Outcome (done, Analysis 009 in `EXPERIMENTS.md`)**: gate passed with m = 0.5, K = 8.
* Zero-voter rate 0% in every configuration; ≥ 2 voters for 99.98% of symbols and 100% of
  lines at m = 0.5.
* Scale: staff space 54 px, so a 64 px patch is ~1.2 staff spaces. Same-class crowding
  happens at one staff space (adjacent staff and ledger lines, chord notes), not below:
  apart from 505 coincident GT (shared noteheads of two voices, kept), no symbol pair is
  within 32 px and no line pair within 16 px.
* Kept apart at one staff space (64 px): m = 0 / 0.25 / 0.5 → symbols 96.6 / 99.99 / 100%,
  lines 20 / 69 / 100%. Two parallel lines g patch units apart are kept apart with
  probability 1 + 2m − g, and g reaches 1.0 at 54 px extraction, so only m = 0.5 protects
  staff and ledger lines at every scale.
* At m = 0.5, K = 5 overflows 20–25% of lines (61% of staff segments); K = 8: 1.8–3.8%.
* For refined lines (duplicates within 2–8 px) crowding is not a risk at all; cannot-link
  matters for coarse-geometry association and for symbols.

### Phase 1 — association on the current detector (Analysis 011, offline)

The global one-to-one model never matches two slots of one patch to the same GT, so the plan
assumed its outputs satisfy the cannot-link assumption and tested the inference half before
training anything. **Outcome (Analysis 011): the assumption is false at inference**; see the
outcome block at the end of this phase. Plan as run:

* Exp 037 / 038 outputs, all queries above floor 0.05: NMS + noisy-OR (current) vs
  association + fusion + support ratio, for lines and symbols.
* New diagnostics (`scripts/diagnose_crowding.py`, reusing the replayed COCO matching of
  `scripts/diagnose_line_errors.py`):
  * **crowded-pair recall**: both GT of a same-class pair within d_crowd found
    (noteheads in seconds, adjacent ledger lines, staccato pairs, parallel beams);
  * **coincident-pair recall**: both GT of the 505 shared-notehead pairs found (one cluster
    per voice);
  * **staff recall by end type** (Analysis 010): barline, system edge (courtesy clef / time
    signature), mid-measure clef / key / time change; and the number of staff fragment
    clusters per page;
  * **merge rate** (two GT → one cluster) and **split rate** (one GT → two confident
    clusters);
  * per-endpoint error of fused vs best single member, by segment length (Analysis 003's
    far endpoint);
  * **coarse duplicate spread of symbols** (distance between same-GT predictions of
    different patches): symbols have no refiner, and Analysis 009 found chord notes one
    staff space (~54–70 px) apart, so this spread decides the symbol association radius.
* The current model has no presence logit: use its class score as a stand-in for presence
  (floor 0.05, support = Σ class score / N_expected). This tests the association, fusion
  and cannot-link code, not the presence/class split, which only Phase 2 can check.
* Expected: small gains only (the current scores are one-to-one, so there are few
  confident voters); the value is validating the inference code and the metrics.

**Phase 1 outcome (Analysis 011, all 351 pages)**:

* Association **without** cannot-link equals NMS for lines (common AP 0.9638 vs 0.9636) and
  slightly beats top-k for symbols (0.8851 vs 0.8730, confident duplicates halved): the
  grouping and fusion code is sound.
* **Cannot-link fails on the current model**: the unmatched slots of a patch copy its matched
  slot at low score (14% of line slots ≥ 0.05 have an in-patch twin). Cannot-link keeps each
  copy in its own cluster, and noisy-OR lifts it: 6,272 confident line duplicates vs 1,100.
* **Coincident noteheads are merged** (pair recall 0.46 → 0.12, with or without cannot-link):
  the two voices come from different patches, one slot each. Only Phase 2's target (both GT
  assigned to every responsible patch) turns the pair into a cannot-link case.
* **Spreads are 3–4× too wide** (97% of refined-line errors within 1 std, symbols 95%), so a
  "3 std" radius is ~10 true std and merges neighbours. A global calibration would not help by
  itself: scaling all stds by c equals dividing the radius by c, so this is a radius choice
  (~1 current std ≈ 3.5 true σ). Calibration matters only if the relative spreads are wrong.
* **Cluster score**: noisy-OR inflates clusters of weak votes; best-member / support scoring
  removes duplicates but loses AP because today's class score is not a presence probability.
  Needs the presence logit.
* Symbol association radius should scale with size (same-GT spread median 0.36 ×
  sqrt(area)), not a fixed 8 px. The refiner stays mandatory (coarse endpoints: −0.14 AP).
* Phase 2 acceptance on this script: cannot-link variant at least as good as no-cannot-link,
  coincident-pair recall ≥ the top-k baseline (0.46), presence close to 0 / 1.

### Phase 2 — training implementation (laptop)

* `src/model/assignment.py`: regions, overflow priority, enumerated in-patch assignment,
  typed outputs (no `Any`, typed tensor wrappers as in `src/music_types.py`).
* `src/model/detector.py`: one presence logit per slot in `SymbolHead` and `LineHead`
  (`preds_per_shape` + 1), bias initialised to the focal prior (−log 99) like the class
  logits; `DetectionOutput` gains a typed `pred_presence` field.
* Checkpoint loading: the heads' last `Linear` changes shape, so loading Exp 037 needs a
  remap that copies the existing rows per slot and initialises the presence rows (zero
  weights, prior bias). The same remap handles a larger line K if Phase 0 asks for one.
* `src/model/criterion.py`: presence focal loss and VFL on the class sigmoids, sharing the
  positive / negative / ignored labelling; 1 / N_voters weights.
* `src/model/refiner.py`: `refiner_batch` voter subsampling; `associate_detections`.
* `src/train_detection.py`: `--assignment {global_one_to_one,patch_set}` (the old matcher
  stays only to reproduce earlier experiments), `--responsibility_margin`,
  `--ignore_band`, `--line_num_shapes` if Phase 0 asks for a larger line K (a new K needs a
  resized head: initialise the extra slots from the existing ones).
* `src/inference_coco.py`: association for both modalities.
* Tests (`unittest`): region membership against brute force; enumerated assignment
  equals scipy `linear_sum_assignment` per patch; overflow priority; ignore rules (an
  in-patch duplicate is never ignored, and gets presence target 0); presence and class
  losses skip exactly the same ignored slots; VFL equals focal at q = 1; the checkpoint
  remap reproduces the Exp 037 class and geometry outputs exactly; association never puts
  two predictions of one patch in a cluster; fusion and support ratio on hand-built cases.
* Smoke runs: overfit a few 1536 px crops from the Exp 037 checkpoint and check that
  (a) the in-patch duplicate rate stays low, (b) voters' presence rises together (spread
  within a GT shrinks), (c) **presence comes out close to 0 or 1**, including on hard
  negatives (text, stray marks, slots just outside the ignore band), not only in the overall
  histogram, (d) the loss balance is close to Exp 038's (measure as in the Exp 038
  loss-balance table).

**Phase 2 outcome (Exp 042 smoke runs, 4 pages)** — implemented as above, with these changes
from the design, found by the smoke runs:

* **Presence init and optimisation**: presence bias at logit(base rate) (lines 0.35, symbols
  0.08), not 0.01; head output layers in a 10× LR group (`--head_lr_scale 10`); presence focal
  α = 0.5; every voter is a full presence positive, normalised by the number of voters
  (`--presence_weighting none`), not 1 / N_voters. Presence in the matching cost did not help
  (`--cost_presence 0`).
* **Lines are not strictly one-to-one inside a patch**: coarse line endpoints are off by
  patches along the line, so which slot wins the match is arbitrary and presence could not
  learn it. Each responsible line keeps its matched slot, and every other slot lying on a
  responsible line (perpendicular quality ≥ 0.5, σ 0.1 patch, extent ignored) is a positive too
  (`--line_on_line`). Lines associate without cannot-link (`--line_cannot_link` off); symbols
  keep strict in-patch one-to-one and cannot-link.
* **Line class quality targets are off** (`--line_quality_targets` to enable): the coarse OKS of
  a far voter is ~0, which would push real positives' class score to 0.
* Result: presence AUC symbols 0.945, lines 0.870 (linear-probe ceilings 0.95 / 0.87).
* **Symbol class head** (Exp 043): per-slot class rows depend on the slot specialisation of the
  one-to-one init (each class on one slot); the patch-set matcher gives objects to whichever slot
  fits the box, so the rows fall silent. Fix: one classifier shared by a patch's slots, reading
  the hidden vector and the slot's box, initialised from the specialist slots, trained on
  positives and in-patch duplicates only (open decision 3, factored score: presence rejects
  background). Smoke: 9 of 11 classes back to the one-to-one control (Exp 044 tests it at scale).
* **Line association** (section 5 revised): refined line voters agree across the line but not on
  its far end, so endpoint-distance association splits staff lines (37 clusters per GT) and an
  endpoint mean mixes good votes with guesses. Long collinear classes (staff, voltaBracket,
  system, octave) are associated by chains of voters lying on each other's line, cut at the
  modes of their certain endpoint votes (refined spread ≤ 16 px), ranked by support × endpoint
  agreement (`src/model/line_association.py`); the other line classes keep endpoint
  association with spread-weighted fusion. Smoke cache: line AP 0.389 → 0.523.

### Phase 3 — Exp 043 on Kaggle (Exp 040 became an unrelated experiment)

* **Same recipe and init as Exp 038**: Exp 037 checkpoint + Phase A refiner, 100 epochs,
  LR 1e-4, 1 warmup epoch, cosine, 54–74 px, size-difficulty weights 0.5 / 8.0, refiner in
  the step, `--compile --use_sdpa`, 2×T4. Only the assignment, losses and post-processing
  differ, so **Exp 038 is the control**. This avoids the restart confound of Exps 032/034
  and the 038 test run.
* Requires the Exp 038 full run to finish first (as the control and to know the Kaggle
  time budget; the in-patch assignment should cost less than the global greedy matcher,
  but a larger line K and more refiner queries add cost: measure it in Phase 2).

### Phase 4 — evaluation

* `evaluate_coco.py` on all pages, full pipeline for both models: 038 with NMS + noisy-OR,
  040 with association. Symbols per frequency group and on the three real small classes
  (Analysis 008).
* The Phase 1 crowding diagnostics on both. They are the direct test of the hypothesis,
  more than global AP.
* Staff miss rate per length bin (Analysis 003), far-endpoint error, recall ceiling
  reached vs Phase 0's theoretical one.
* Log in `EXPERIMENTS.md` (Analyses 009/010, Exp 043).

### Later (only if Phase 4 is positive)

* Learned cluster scoring: a small set network over each cluster's members (scores,
  geometries, patch offsets) that outputs score and geometry, trained on the
  cluster-to-GT matching.
* A held-out split by piece (`--holdout_fraction`) to check that the scores generalize
  beyond memorized pages.

## Risks and what would reveal them

| risk | signal | fallback |
|---|---|---|
| line slots overflow on staves | Phase 0 overflow and zero-voter rate | larger line K; smaller m |
| quality targets collapse scores early (Exp 032) | training cls loss, score spread per GT | warm up with q = 1 for a few epochs, then quality |
| far voters of long lines fall outside the refiner window | masked fraction in refiner stats | certainty-weighted fusion already down-weights them; restrict line voters to within L patches of an endpoint |
| in-patch duplicates break cannot-link | split rate | stronger in-patch negative weight |
| presence not close to 0/1, so the floor is arbitrary again | presence histogram on positives, negatives, hard negatives | calibrate the floor on training pages; raise the presence weight |
| inference cost (many more candidates above the floor) | s/page | cluster on coarse geometry, refine fused lines once |
| loss balance shift harms symbols | symbol AP per frequency group vs 038 | revisit 1 / N_voters (e.g. 1 / sqrt) |

## Open decisions

1. **K**: settled by Phase 0: K = 8 for both heads (was 5). The heads' last layer grows,
   so the checkpoint remap must initialise 3 new slots per patch (e.g. copies of existing
   slots with their presence bias at the prior).
2. **Init**: Exp 037 (same as 038, clean control) or Exp 038's final checkpoint (better
   start but no control). Recommended: Exp 037.
3. **Factored score (later, gated)**: replace the class sigmoids by presence × class given
   an object (a softmax trained only on positives) × quality. Cleaner in principle (class
   imbalance and background imbalance handled separately), but two of its three factors
   would be trained only on positives, so presence alone would have to reject every
   background slot (the GFL caveat in section 3), and the class logits would change
   meaning (warm-start risk, Exps 032/034). Only worth trying if Exp 043 shows that
   presence is reliably close to 0/1 on hard negatives.
4. **Symbol fusion without a symbol refiner**: symbol duplicates will not coincide the way
   refined lines do, so the association radius and certainty weights matter more for them.
   Phase 1 shows how much.
