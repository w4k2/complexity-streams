# Complexity streams

> Little info.

## Scripts and directories
- `config.py` - common stream configuration used in all experimental scripts.
- `stream-generator.py` - base stream generator; stores files in `streams/`, which due to the size, **are ignored by repository**.
- `complexity-calculator.py` - calculator of all complexities for all generated streams; stores files in `complexities/`, which **are stored in repository**.
- `example-stream.py` - exemplary stream loader.
- `example-complexity.py` - exemplary complexity signal loader.

## Common configuration

Filename convention:

```
(?drift_type)_f(?dimensionality)_c(?clusters)_r(?replication)
```

- Dimensionality (`dimensionality`):
    - 8 features (`8`),
    - 16 features (`16`),
    - 32 features (`32`).
- Clusters (`clusters`):
    - 2 clusters (`2`),
    - 3 clusters (`3`),
    - 4 clusters (`4`).
- Concept drifts (`drift_type`):
    - balanced streams:
        - sudden (`bal_sudden`),
        - gradual (`bal_gradual`),
        - incremental (`bal_incremental`).
    - streams imbalanced dynamically synchronously.
        - sudden (`imb_sudden`),
        - gradual (`imb_gradual`),
        - incremental (`imb_incremental`).
- Other configuration (not in filename):
    - 10 replications,
    - fully informative,
    - 2000 chunks,
    - 500 samples in chunk,
    - 7 drifts.