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

3 directories, 2592 files
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
