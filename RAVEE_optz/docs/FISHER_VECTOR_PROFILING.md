# Fisher Vector CPU preparation and profiling

The mathematical extraction path remains global PCA, GMM fitting, per-frame
PCA/Fisher Vectors, then final PCA. PCA fitting scopes, default solvers, GMM
parameters, random sampling, dimensions and equations are unchanged.

## Optional profiling

In the shell used to launch the application:

```powershell
$env:RAVEE_PROFILE_FV = '1'
python toolbox/main.py
```

The normal selector and processing authorization still apply. This flag does not
start extraction by itself. Unset the shell variable to disable profiling; no
production `.env` change is needed. Profiling is disabled by default.

Each video's FV cache operation prints one `FV profile: {...}` JSON record to
the console, including on failure. It is an in-memory report: no profile files
or cache metadata are written. A verified cache hit reports loading/lookup times
and feature dimensions; it does not rerun extraction to obtain missing timings.

Programmatic callers can use `prepare_features(..., fv_profile=True)`, or supply
`profile=FVProfile()` directly to `extract_framewise_features`. The collector is
defined in `toolbox/fv_profiling.py`. It retains counts/totals and scalar metadata,
never references to the matrices. Legacy two-argument injected extractors remain
supported; they only supply detailed internal timings if they support the hooks.

| Timing key | Work measured |
|---|---|
| `trajectory_loading` | Loading the raw trajectory `.npy` |
| `descriptor_assembly` | Named descriptor fields and concatenation |
| `global_pca_fit`, `global_pca_transform` | Global PCA fit and transformation |
| `gmm_fit` | GMM fit on the unchanged sampled/full descriptor matrix |
| `trajectory_frame_mapping` | Original inclusive 15-frame memberships |
| `frame_descriptor_preparation` | Indexed raw descriptor rows for each frame |
| `frame_pca_fit`, `frame_pca_transform` | Independent local PCA fits/transforms |
| `gmm_posterior` | GMM `predict_proba` on each frame's reduced descriptors |
| `fisher_statistics` | Sufficient statistics, derivatives and output assembly |
| `frame_vector_assembly` | Assembling the framewise FV matrix |
| `final_pca_fit`, `final_pca_transform` | Final PCA fit and transformation |
| `feature_text_serialization` | Existing compatibility text output |
| `cache_serialization` | Existing uncompressed `np.savez` archive write |
| `cache_array_loading` | Cache archive reads, including validation rereads |
| `cache_lookup` | Initial manifest/checksum lookup |
| `extraction_total`, `cache_stage_total` | Enclosing extraction/cache wall time |

Repeated frame timings are aggregated as `seconds` and `calls`. Stages not
executed (for example local PCA for a small frame) are absent. Enclosing totals
overlap the individual timings and must not be added to them. Cache-stage totals
also include validation/checksum/publication overhead; the initial lookup timing
does not include every later internal verification. These are wall times, not
CPU-time or peak-memory measurements.

Metadata includes trajectory/frame counts, summed/minimum/maximum per-frame
descriptor counts, local PCA/zero-fallback counts, matrix shapes/dtypes/byte sizes,
GMM sample dimensions, K, PCA dimensions and relevant package versions.

GUI substages and completed/total frame counts use the existing background
worker/event queue. Frame updates are limited to once per 0.5 seconds plus start
and completion; stage transitions are immediate. Detailed timing is independent
of these progress messages.

## Lossless CPU preparation

`read_descriptors` uses named structured-array fields in the existing order:
trajectory (30), HOG (96), HOF (108), MBHx (96), MBHy (96). Homogeneous field
dtypes use one concatenate. Unusual mixed field dtypes retain the historical
intermediate concatenation/promotion order to preserve rounding.

Per-frame preparation uses `all_desc[indices]` from the already constructed raw
descriptor matrix. It preserves insertion order of frames, trajectory order,
duplicates and inclusive memberships. This copies one frame's selected rows,
without another full-size descriptor matrix. It does **not** index the global
PCA output: every frame still fits its own PCA, with the original <=63-row zero
fallback. The compatibility text file is still written.

Synthetic fixtures compare descriptor matrices and frame inputs with a frozen
reference using `np.array_equal()`. Fixed model parameters also check FV equations
and assembled outputs exactly. Independent full training runs are not promised
to be bitwise identical: existing randomized PCA/GMM/sampling and native-library
behavior remain unchanged. No seeds have been added or changed in production.

## Cache implications

FV inputs already fingerprint the complete `feature_extraction.py` source and
NumPy/scikit-learn versions. This source edit therefore creates a new FV identity
and a new downstream FINCH identity, even for mathematically equivalent values.
Preprocessing and raw IDT identities are unchanged. The profiling flag/results
are not cache inputs; toggling profiling alone does not change cache identities.

No cache is deleted, migrated or silently adopted, and checksum/shape validation
is unchanged. Earlier artifacts/manifests remain in place and can still be
verified under their recorded source/environment. Current code will not claim
that an artifact with a different recorded extractor hash is compatible.

## Offline validation and microbenchmark

```powershell
python -B -m unittest tests.test_feature_extraction tests.test_pipeline_cache -v
python -B -m tests.benchmark_fv_descriptors
python -B -m unittest discover -s tests -t . -q
```

The benchmark uses deterministic float32 structured arrays, a warmup and medians
of nine measurements. Local frame selection uses 1,024 indices including repeated
references. It performs no PCA/GMM training, video work or cache writes.

Observed on the working environment, 2026-11-17 (milliseconds):

| Trajectories | Assembly before | Assembly after | Frame preparation before | Frame preparation after |
|---:|---:|---:|---:|---:|
| 2,000 | 11.324 | 1.118 | 5.460 | 0.524 |
| 20,000 | 129.550 | 15.768 | 5.670 | 0.577 |

Both comparisons passed exact array/dtype equality. These timings measure
descriptor preparation only. They do not establish an overall FV speedup. Global
PCA, GMM fitting, repeated local PCA/posteriors, memory pressure and disk I/O
still require profiling on an authorized real workload.
