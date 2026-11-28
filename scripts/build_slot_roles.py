"""Exp 037 slot roles: for each symbol GT, the slot (query % 5) of its best-IoU candidate;
per class, the distribution over the 5 slots. Saved as JSON {"num_slots": 5, "roles": {class_index:
[p_0..p_4]}} (scripts/exp037_symbol_slot_roles.json): the specialist slot of each class initialises
the shared symbol classifier (--symbol_shared_class --slot_roles).

Usage: PYTHONPATH=src:scripts mamba run -n pytorch python scripts/build_slot_roles.py
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from diagnose_association import load_page  # noqa: E402
from dataset.coco import load_coco_sample, parse_coco  # noqa: E402
from model.assignment import pair_quality  # noqa: E402

cache = Path("experiments/037_029_recipe_from_023/slot_cache")
dataset = parse_coco(Path("data/trompa-coco/annotations/instances_trainval2017.json"), cache_dir=Path("data/cache"))
by_id = {m.img_id: i for i, m in enumerate(dataset.images)}
counts = torch.zeros(dataset.num_symbol_classes, 5)
for path in sorted(cache.glob("*.pt"), key=lambda p: int(p.stem))[:: max(1, 351 // 60)]:
    page = load_page(path)
    s = load_coco_sample(dataset, Path("data/trompa-coco/trainval2017"), by_id[page.image_id], torch.device("cpu")).sample
    gt, gl = s.boxes.data.float() / 64, s.box_labels.data.long()
    geo = page.symbols["geometry"].float()
    slot = page.symbols["query"].long() % 5
    for g in range(gt.shape[0]):
        q = pair_quality(geo, gt[g].expand_as(geo), False)
        best = int(q.argmax())
        if q[best] >= 0.5:
            counts[int(gl[g]), int(slot[best])] += 1
roles = {int(c): (counts[c] / counts[c].sum()).tolist() for c in range(counts.shape[0]) if counts[c].sum() > 0}
Path("scripts/exp037_symbol_slot_roles.json").write_text(json.dumps({"num_slots": 5, "roles": roles}, indent=1))
ent = [-(torch.tensor(p).clamp(min=1e-9).log() * torch.tensor(p)).sum().item() for p in roles.values()]
print(f"{len(roles)} classes with roles; mean entropy {sum(ent) / len(ent):.3f} nats; total GT {int(counts.sum())}")
print("dominant slot per class (first 12 by count):", sorted(((int(counts[c].sum()), c, int(counts[c].argmax())) for c in roles), reverse=True)[:12])
