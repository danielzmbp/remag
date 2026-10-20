# REMAG methods

[Back to the README](https://github.com/danielzmbp/remag/blob/main/README.md)

## Model architecture

REMAG combines 136 features describing 4-mer composition with `p` coverage features. The embedding dimension `D` is set by `--embedding-dim` and defaults to 256 in the CLI.

| Component | Layer widths with coverage |
| --- | --- |
| Composition encoder | 136 → 128 → 64 |
| Coverage encoder | p → 64 → 32 |
| Training projection head | D → 128 → 512 |

Both encoders use BatchNorm, LeakyReLU, and dropout of 0.05. Coverage encoder widths stay fixed for every positive `p`. Fusion uses bidirectional four-head cross-attention, fixed 0.5 direct-versus-attended gates, a learned residual weight, three fusion scales, nonlinear compression, adaptive dropout, and a final MLP.

Barlow Twins training uses the projection head. Graph construction and rescue use the pre-projection `D`-dimensional contig embeddings.

When no coverage is supplied, the composition encoder uses 136 → 256 → 128, followed by projection to `D` and a D → 512 → 512 training head.

Implementation: [models.py](https://github.com/danielzmbp/remag/blob/main/remag/models.py).

The README diagram is rendered from [architecture.dot](https://github.com/danielzmbp/remag/blob/main/docs/images/architecture.dot). To regenerate it with Graphviz from the repository root:

```bash
dot -Tpng docs/images/architecture.dot -o docs/images/architecture.png
```

## Clustering

REMAG constructs a k-nearest-neighbor graph using cosine similarity, then applies greedy Leiden clustering across multiple resolutions with cosine-similarity edge weights. The default is 15 neighbors, reduced when there are too few contigs.

Only contigs meeting `--graph-min-contig-length` enter the initial graph. Admitted shorter contigs share the same embedding space and remain available for rescue. The original graph is retained for cache validation and whole-bin merge checks.

Implementation: [clustering.py](https://github.com/danielzmbp/remag/blob/main/remag/clustering.py).

## Rescue

Rescue first merges fragmented bins, then recruits unassigned contigs. Both stages use length-weighted centroids and a cosine similarity threshold of 0.70. Bins below `--min-bin-size` remain eligible seeds until the final export filter. `--skip-rescue` disables both stages.

### Whole-bin merging

Bins are considered in a stable smallest-first pass. Each source tries its nearest eligible target, which must be at least as large. Centroids remain fixed throughout merging.

A merge requires at least one original clustering-graph edge connecting the bins' current members, including contigs absorbed by earlier merges. If the nearest eligible target fails this check or the marker limits below, no alternative target is tried.

### Contig recruitment

Centroids are recomputed once after merging and remain fixed during recruitment. Unassigned graph contigs and admitted shorter contigs are considered together in embedding row order. Each contig tries only its nearest bin; recruitment does not require an original graph edge.

### Marker safeguards

Duplication is the percentage of detected marker families found on more than one contig in a bin. Each family is counted once per contig for clustering and rescue; multiple non-overlapping copies within one contig are reported separately in the final quality statistics.

By default, an accepted marker-bearing candidate must leave at most 5% duplicated marker families and increase duplication by strictly less than 5 percentage points. Marker counts update after each accepted merge or recruitment. Marker-free contigs skip the marker checks during recruitment.

- `--rescue-max-total-duplication` sets the final duplication ceiling. Whole-bin merges cap this setting at 10%.
- `--rescue-max-duplication-increase` sets the strict increase limit: exactly +5 percentage points is rejected at the default setting of 5.

Whole-bin merges have one additional allowance: at cosine similarity ≥0.95, a target already above the duplication ceiling may accept a merge that does not worsen duplication. The strict increase limit and graph-edge requirement still apply. This allowance does not apply to individual-contig recruitment.

Implementation: [rescue.py](https://github.com/danielzmbp/remag/blob/main/remag/rescue.py), [output.py](https://github.com/danielzmbp/remag/blob/main/remag/output.py), and [miniprot_utils.py](https://github.com/danielzmbp/remag/blob/main/remag/miniprot_utils.py).
