# music deep

The goal of this repo is to train a vision transformer for optical music recognition.

Steps to do:

1. dataset constitution:
    * imslp dataset for SSL
2. dataset loading
3. Vision transformer model
  * with patch dropping
4. LeJEPA self-supervised learning

## Datasets

```
$ tree data                                      
data
└── imslp
    ├── images
    │   ├── IMSLP00022-001.tiff
    ...
    │   ├── IMSLP96504-007.tiff
    │   └── IMSLP96504-008.tiff
    └── imslp.jsonl
data/trompa-coco
├── annotations
│   └── instances_trainval2017.json
└── trainval2017
    ├── Beethoven_Op119_Nr01-Breitkopf_001.png
    ...
    └── Schumann-Clara_Romanze-ohne-Opuszahl_a-Moll_003.png
```

### IMSLP

```
$ jq . data/imslp/imslp.jsonl
{
  "name": "IMSLP00022-001.tiff",
  "score": "IMSLP00022",
  "page": 1
}
{
  "name": "IMSLP00022-002.tiff",
  "score": "IMSLP00022",
  "page": 2
}
...
```

### Trompa-COCO

The Trompa-COCO dataset is a synthetic dataset of full-page music scores. Key characteristics and design decisions for this project include:
- **Synthetic & Sparse**: The dataset is synthetic, and the visual information is fundamentally very sparse.
- **Interline Height**: The interline height is ~64 pixels, which perfectly aligns with our chosen patch size of 64.
- **Full-Page Training**: We train strictly on full pages (resolutions around 6400x8000). This avoids the headache of handling partial/cut symbols at the edges of crops and preserves the true underlying symbol distribution, which is critical for class rebalancing.
- **Patch Size 64 is Non-Negotiable**: Given the massive full-page resolutions, using a patch size of 64 is the absolute minimum feasible size to fit in memory. Do not attempt to reduce the patch size to solve localization issues; the 64x64 patch size is a strict constraint of the domain.
