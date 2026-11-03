# REMAG methods

[Back to the README](https://github.com/danielzmbp/remag/blob/main/README.md)

## Eukaryotic filtering

The HyenaDNA classifier scores contigs of at least `--min-contig-length`. Shorter contigs are not classified and do not appear in the rejected-contig file.

A contig up to 4,096 bp is scored as one window. Longer contigs are scored with up to 16 evenly spaced windows. Each window holds up to 4,094 bases after tokenization. The whole contig is kept when its mean eukaryote probability is at least 0.30. The `confidence` column in the classification table is diagnostic only. Kept contigs retain their original headers, sequences, and order.

Inference uses fixed batches of 64 contigs and up to 256 windows. Non-default `--hyenadna-batch-size` values are ignored with a warning.

If no contigs pass, REMAG writes an empty filtered FASTA and stops before training. Checkpoint or prediction validation errors stop the run. Other failures are logged:

- If the classifier fails to initialise, the original FASTA is used unfiltered.
- If a batch fails, its contigs are kept without prediction rows.

Output names use the FASTA name without `.gz` and its extension: `contigs.fa.gz` gives `contigs_eukaryotic_filtered.fasta`. `--filter-only` with `--skip-bacterial-filter` produces no filtered FASTA.

Implementation: [filter_policy.py](https://github.com/danielzmbp/remag/blob/main/remag/filter_policy.py) (checkpoint identity and settings) and [predictor.py](https://github.com/danielzmbp/remag/blob/main/remag/hyenadna_classifier/predictor.py).

## Features

Each admitted contig contributes its full sequence plus up to `--num-augmentations` (default 8) training fragments. A candidate is discarded if it overlaps an earlier fragment by 25% or more of the longer one, and sampling stops after 10 attempts per requested fragment, so short contigs can receive fewer. A fragment is the contig with one or both edges trimmed, or one side of a removed central region, and is at least `--min-contig-length` long. Contigs longer than 50 kb are split in half, and each half receives its own set of fragments. Fragment positions are seeded from the contig name, so they are identical across runs.

Composition is the canonical 4-mer frequency profile: reverse complements are merged into 136 features.

Coverage adds a mean and a standard deviation per sample:

- **BAM/CRAM:** per-base depth over each fragment, divided by the sample's mapped reads in millions. For CRAM, mapped reads are alignments without flag 4.
- **Interval COV/bedGraph:** per-base values over each fragment, with uncovered positions counted as zero.
- **Contig-level TSV/TXT:** the contig's value copied to all its fragments, with no standard deviation column. The first column is the contig ID and the last is the value. Use two columns: a file is read as interval coverage when its first data line has four or more columns, with integers in columns 2 and 3 and a number in column 4.

Coverage columns are log1p-transformed, then min-max scaled to [0, 1] column by column.

Implementation: [features.py](https://github.com/danielzmbp/remag/blob/main/remag/features.py).

## Training

Positive pairs are fragments of the same contig, capped at `--max-positive-pairs`. Pair selection uses a fixed seed of 42, independent of `--random-seed`. The batch size is halved until it fits the number of pairs, with a minimum of 2.

The learning rate is `batch_size / 256 × base_learning_rate × 0.2`, using the requested `--batch-size`. It warms up linearly from 10% over 10 epochs, then follows a cosine decay to 1% over the remaining epochs. Early stopping uses a patience of 20 epochs and restores the best weights. Only the full-contig embeddings are used for clustering and rescue.

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

REMAG constructs a k-nearest-neighbor graph using cosine similarity, then applies greedy Leiden clustering across multiple resolutions with cosine-similarity edge weights. The default is 15 neighbors, reduced when there are too few contigs. Edges below `--leiden-similarity-threshold` (default 0.1) are dropped. Each contig adds edges to its own neighbors, so a mutual-neighbor pair is connected twice and carries double weight.

Greedy clustering repeats one round until no candidate remains:

1. Run Leiden on the remaining graph at every `--greedy-resolutions` value (default 0.1, 0.5, 1, 2, and 5).
2. Score every cluster of two or more contigs as an F1 of completeness and purity:
   - completeness = distinct marker families / 133
   - purity = 1 − (marker hits − distinct families) / marker hits
3. Reject clusters in which more than 10% of the 133 families are duplicated.
4. Keep the best-scoring cluster across all resolutions as a bin and remove its contigs from the graph.

The loop also stops when fewer than two contigs or no edges remain. A cluster without markers scores 0, but can still be selected once nothing better remains.

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

## Output and reuse

`pre_rescue.csv` lists every admitted contig. Contigs outside the clustering graph start as noise. Edge indices in `knn_graph_edges.csv` refer to rows of `knn_graph_contigs.csv`, not to embedding rows.

Marker annotation failures stop binning. A successful search with no accepted matches reports zero detected genes.

Saved training and clustering results are checked against recorded settings and input fingerprints before reuse. Compatible embeddings can be reused without saved model weights, and rescue restarts from `pre_rescue.csv`. Matching reruns keep the original `params.json`. Incompatible or incomplete provenance requires `--force` or a new output directory.

A cached filtered FASTA is reused without reclassification, even when the recorded classifier policy differs or is unknown; a warning is logged.

`--force` removes recognised REMAG outputs: bins, caches, filtering results, logs, temporary miniprot directories, and UMAP plots. Other files are left in place. Cleanup stops before deleting anything in three cases:

- an input file would be removed;
- `bins/` is a symlink;
- a temporary directory contains symlinks.

`--force --filter-only` also clears earlier binning results.
