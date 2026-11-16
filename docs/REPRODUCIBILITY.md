# Reproducibility record — 2026-11-16

## Scope and evidence

This records the existing Windows AMD64 `.venv`, using Python **3.14.0**.
Versions below were read with `importlib.metadata` using that interpreter;
no package was installed/upgraded and no `pip freeze` was used. No root
`requirements.txt`, `pyproject.toml`, `setup.py` or lock file was present.
The tables are a curated environment record, **not a portable dependency lock**
or proof that all versions/wheels are available from a public package index.

### Direct application dependencies observed

| Import / purpose | Distribution | Observed version |
| --- | --- | --- |
| NumPy arrays | numpy | 2.5.4 |
| Dataframes and event logs | pandas | 3.1.0 |
| Scientific functions | scipy | 1.18.1 |
| PCA/GMM/nearest-neighbor utilities | scikit-learn | 1.9.1 |
| Video metadata, preview and media | opencv-python | 5.0.0.93 |
| Video helpers | moviepy | 2.2.1 |
| FINCH plotting | matplotlib | 3.11.2 |
| Process mining and CSV/XES/BPMN | pm4py | 2.7.23.11 |
| Images and Tk previews | Pillow | 11.3.0 |
| Gemini SDK | google-genai | 3.3.0 |
| Local configuration | python-dotenv | 1.2.4 |

### Media support and optional research tools

| Distribution | Observed version | Role |
| --- | --- | --- |
| imageio | 2.38.1 | Media support in the existing environment |
| imageio-ffmpeg | 0.8.1 | FFmpeg package support; not a system FFmpeg version |
| torch | 2.15.0+cu134 | Optional CLIP/TimeSformer/VideoMAE tools |
| transformers | 5.22.2 | Optional research model loading/inference |
| pynndescent | Not installed | Optional small-input path; needed above FINCH's approximate-neighbor threshold |

Tkinter/Tcl/Tk are supplied by the Python installation, not by a pip entry in
this table. A usable desktop/Tcl runtime is required for the application GUI.
Graphviz `dot` and Docker Desktop are external dependencies; their installed
versions and working runtime availability were not established by this cleanup.
Codec availability and model weights/revisions also need separate recording for
an experimental replication. No GUI/video processing or model inference was run.

## Existing Docker specification — inspected, unchanged, not built

The local `IDT/Dockerfile` specifies `ubuntu:20.04`, builds OpenCV and OpenCV
contrib **3.4.12**, installs pandas **1.5.3** and NumPy **1.21**, and leaves
scikit-learn unpinned. Ubuntu packages are installed/upgraded without a complete
snapshot/digest lock. The container's Python/native build environment is separate
from the Windows Python 3.14 environment above. These are source-file observations,
not installed-container versions or verification that a fresh build succeeds.

The pinned upstream submodule revision alone does not contain all local IDT
patches. Preserve those patches and data as described in the existing cleanup
inventory. This cleanup does not change/build/run the submodule.

## Offline test contract

```powershell
python -B -m unittest discover -s tests -t . -v
```

Loading `tests` installs test-process-only isolation before application imports:

- Remove inherited Gemini profiles/model selectors and Google API credentials;
  prevent dotenv auto-discovery; use a fake default credential.
- Point configuration tests at an absent test `.env` by default. Individual tests
  may explicitly patch this to a synthetic temporary configuration.
- Reject unmocked `genai.Client` construction. Test clients and media are mocks.
- Reject reading the real project `.env` or writing/moving/removing files in
  production data/cache/output locations. Generated fixtures use OS temp paths.

Production configuration loading and caches are unchanged. The CSV/XES regression
constructs fixed data for legacy epoch-encoded numbers (including lost fractional
durations) and current numeric fields. It checks schema/order, IDs, labels,
numeric types, round trips and source nonmutation for 11- and 20-segment cases.
It never opens `EventLog_Output/all_videos_detailed_eventlog.csv`.

Two local artifact checks in `test_process_model_and_labels_v3.py` are **opt-in**
with `RAVEE_RUN_LOCAL_INTEGRATION=1`; deterministic XES/metadata checks also run
in the default suite. The optional checks read a local XES or video header only;
Gemini, process discovery/rendering and external processing remain mocked.
Do not use repository-root test discovery for the standalone research scripts.

## Existing batch input limit

`batch_label_suggester.py` currently sets `_MAX_VIDEO_BYTES = 64 * 1024 * 1024`
(**67,108,864 bytes**) and `_MAX_TEXT_BYTES = 1024 * 1024`. It sends the complete
video inline at a requested sampling rate of 3 FPS; it does not re-encode/upload
oversize media or split the operation into per-segment generation calls.
This is an application limit, not a claim about every provider/model limit.
Unsupported/oversize inputs fail while individual labeling/manual review remain
available. This limit was verified from source and was not changed.

## Remaining portability limits

The inventory omits a complete transitive dependency/OS/codec/model-artifact lock.
Fresh installation instructions in `Install.md` are a starting point, not an
identical-environment guarantee. Keep the working environment; do not install
every optional research package as a core requirement or upgrade it merely to
match this record. Cache keys include relevant package versions and source or
explicit semantic signatures, so environment/semantic changes can affect reuse.
Live model accuracy, quota behavior and native GUI rendering require separate
evaluation; offline regression results do not certify them.
