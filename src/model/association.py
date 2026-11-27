"""Association of one-to-many detections (PATCH_SET_ASSIGNMENT.md, section 5).

Instead of suppressing duplicates (NMS), every candidate joins a cluster, and a cluster is one
object. Two rules make this safe in crowds:

* **cannot-link**: a patch never predicts one object twice (one-to-one inside a patch), so two
  candidates from the same patch are never put in the same cluster. A patch that predicts two
  nearby (or coincident) objects therefore keeps them apart.
* **certainty-aware distance**: each coordinate carries the standard deviation of its D-FINE
  edge distribution. A candidate that cannot see one end of a long line (large std there) can
  still join the cluster of the candidates that do, and the fused coordinate is an
  inverse-variance mean, so it comes from the candidates that are sure of it.

Geometry is two points per object, (x1, y1, x2, y2): box corners or line endpoints, in patch
units. The greedy loop runs on numpy arrays: it does many tiny operations, where torch's per-call
overhead would dominate.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Detections:
    """Candidates of one modality of one image.

    ``score`` is the class score (quality estimate: fusion weight and cluster quality),
    ``presence`` the existence evidence summed into the support (the class score until the
    detector has a presence logit), ``patch`` the source patch (cannot-link key).
    """

    geometry: torch.Tensor
    std: torch.Tensor
    label: torch.Tensor
    score: torch.Tensor
    presence: torch.Tensor
    patch: torch.Tensor


@dataclass(frozen=True)
class AssociationConfig:
    """``radius``: largest distance at which a candidate joins a cluster, in patch units when
    ``normalized`` is False, otherwise in combined standard deviations (per point, the larger of
    the two points counts). ``min_std`` (patch units) floors every std."""

    radius: float
    normalized: bool
    min_std: float = 1 / 64
    cannot_link: bool = True


@dataclass(frozen=True)
class Clusters:
    """``member[i]``: cluster of candidate i. Per cluster: fused geometry and std, label, best
    member score (``quality``), summed presence (``evidence``), sum of log(1 - score) of the
    members (noisy-OR: 1 - exp(log_miss)), member count and highest-scored member."""

    member: torch.Tensor
    geometry: torch.Tensor
    std: torch.Tensor
    label: torch.Tensor
    quality: torch.Tensor
    evidence: torch.Tensor
    log_miss: torch.Tensor
    size: torch.Tensor
    representative: torch.Tensor

    @property
    def noisy_or(self) -> torch.Tensor:
        return 1 - self.log_miss.exp()


@dataclass
class _LabelState:
    """Running sums of the clusters of one label (preallocated for ``capacity`` clusters)."""

    weight: FloatArray
    weighted: FloatArray
    inv_var: FloatArray
    count: int

    @staticmethod
    def empty(capacity: int) -> "_LabelState":
        return _LabelState(
            weight=np.zeros((capacity, 4)),
            weighted=np.zeros((capacity, 4)),
            inv_var=np.zeros((capacity, 4)),
            count=0,
        )

    def geometry(self) -> FloatArray:
        n = self.count
        return self.weighted[:n] / self.weight[:n]

    def variance(self) -> FloatArray:
        return 1.0 / self.inv_var[: self.count]

    def add(self, k: int, x: FloatArray, var: FloatArray, w: FloatArray) -> None:
        self.weight[k] += w
        self.weighted[k] += w * x
        self.inv_var[k] += 1.0 / var


def _distance(x: FloatArray, var: FloatArray, centers: FloatArray, center_var: FloatArray,
              cfg: AssociationConfig) -> FloatArray:
    """(M,) distance of one candidate to M clusters: the larger of its two points."""
    d2 = (centers - x).reshape(-1, 2, 2) ** 2
    if cfg.normalized:
        d2 = d2 / (center_var + var).reshape(-1, 2, 2)
    return np.sqrt(d2.sum(axis=-1).max(axis=-1))


def associate(dets: Detections, cfg: AssociationConfig) -> Clusters:
    """Greedy association, per label, in decreasing class score: each candidate joins the
    nearest cluster within ``cfg.radius`` that holds no candidate of its patch, otherwise it
    starts a new cluster. Clusters are returned grouped by label."""
    device = dets.geometry.device
    geometry = dets.geometry.detach().double().cpu().numpy()
    std = np.maximum(dets.std.detach().double().cpu().numpy(), cfg.min_std)
    var = std**2
    label = dets.label.detach().cpu().numpy().astype(np.int64)
    score = dets.score.detach().double().cpu().numpy()
    presence = dets.presence.detach().double().cpu().numpy()
    patch = dets.patch.detach().cpu().numpy().astype(np.int64)
    weight = score[:, None] / var if cfg.normalized else np.repeat(score[:, None], 4, axis=1)
    weight = np.maximum(weight, 1e-12)

    member = np.full(label.shape[0], -1, dtype=np.int64)
    out_geometry: list[FloatArray] = []
    out_std: list[FloatArray] = []
    out_label: list[IntArray] = []
    out_quality: list[FloatArray] = []
    out_evidence: list[FloatArray] = []
    out_log_miss: list[FloatArray] = []
    out_size: list[IntArray] = []
    out_rep: list[IntArray] = []
    offset = 0
    log_keep = np.log1p(-np.clip(score, 0.0, 1.0 - 1e-12))
    for c in np.unique(label):
        idx = np.nonzero(label == c)[0]
        idx = idx[np.argsort(-score[idx], kind="stable")]
        state = _LabelState.empty(idx.shape[0])
        blocked: dict[int, list[int]] = {}
        local = np.empty(idx.shape[0], dtype=np.int64)
        for j, i in enumerate(idx):
            k = -1
            if state.count:
                dist = _distance(geometry[i], var[i], state.geometry(), state.variance(), cfg)
                if cfg.cannot_link and (same := blocked.get(int(patch[i]))):
                    dist[same] = np.inf
                best = int(dist.argmin())
                if dist[best] <= cfg.radius:
                    k = best
            if k < 0:
                k = state.count
                state.count += 1
            state.add(k, geometry[i], var[i], weight[i])
            blocked.setdefault(int(patch[i]), []).append(k)
            local[j] = k
        n = state.count
        member[idx] = local + offset
        out_geometry.append(state.geometry())
        out_std.append(np.sqrt(state.variance()))
        out_label.append(np.full(n, c, dtype=np.int64))
        out_quality.append(np.zeros(n))
        np.maximum.at(out_quality[-1], local, score[idx])
        out_evidence.append(np.bincount(local, weights=presence[idx], minlength=n).astype(np.float64))
        out_log_miss.append(np.bincount(local, weights=log_keep[idx], minlength=n).astype(np.float64))
        out_size.append(np.bincount(local, minlength=n).astype(np.int64))
        first = np.full(n, -1, dtype=np.int64)
        # idx is in decreasing score, so the first member seen is the best one.
        for j in range(idx.shape[0] - 1, -1, -1):
            first[local[j]] = idx[j]
        out_rep.append(first)
        offset += n

    def cat_float(parts: list[FloatArray], width: int | None = None) -> torch.Tensor:
        if not parts:
            return torch.zeros((0, width) if width else (0,), dtype=torch.float32, device=device)
        return torch.from_numpy(np.concatenate(parts)).float().to(device)

    def cat_long(parts: list[IntArray]) -> torch.Tensor:
        if not parts:
            return torch.zeros(0, dtype=torch.long, device=device)
        return torch.from_numpy(np.concatenate(parts)).to(device)

    return Clusters(
        member=torch.from_numpy(member).to(device),
        geometry=cat_float(out_geometry, 4),
        std=cat_float(out_std, 4),
        label=cat_long(out_label),
        quality=cat_float(out_quality),
        evidence=cat_float(out_evidence),
        log_miss=cat_float(out_log_miss),
        size=cat_long(out_size),
        representative=cat_long(out_rep),
    )


def support_ratio(clusters: Clusters, expected: torch.Tensor) -> torch.Tensor:
    """Summed presence over the number of patches expected to vote for the cluster, in [0, 1]
    when presence is a probability and every voter is an expected one."""
    return (clusters.evidence / expected.clamp(min=1).to(clusters.evidence)).clamp(max=1.0)
