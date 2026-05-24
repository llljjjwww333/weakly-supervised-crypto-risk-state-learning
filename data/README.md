# Data Notes

Raw and processed data are intentionally not versioned in this repository.

Expected local layout after running the download and preprocessing pipeline:

```text
data/
  raw/spot/1h/
  interim/spot/1h/
  processed/features/1h/
  processed/windows/1h/
  labels/
  metadata/
```

Use the commands in the top-level [README](../README.md) to regenerate these directories locally.
