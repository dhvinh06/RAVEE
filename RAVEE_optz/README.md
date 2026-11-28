# RAVEE

RAVEE turns videos into reviewed activity segments, semantic tasks and event logs
for process mining. This working fork adds cached non-demo processing, Gemini
Label Suggester V3, optional TwelveLabs Pegasus labeling/aggregation,
human-reviewed Semantic Task Aggregation and process-model export.

## Architecture

| Stage | Implementation | Result |
| --- | --- | --- |
| Preprocess input videos | `toolbox/preprocessing.py` | Reduced videos |
| Extract raw trajectories | `IDT/` Docker subsystem | Frame-indexed trajectory arrays |
| Encode features | `toolbox/feature_extraction.py` | PCA/GMM/Fisher Vector features |
| Temporal clustering | `finch/finch.py` | Original FINCH segments |
| Label and review segments | `gui/gui_elements.py`, `label_suggester.py` | Confirmed activity labels |
| Discover and review tasks | `toolbox/semantic_aggregation.py`, `gui/semantic_review.py` | Confirmed groups of original segments |
| Export logs and model | `toolbox/event_log_creation.py` | Detailed/task CSV and XES; task BPMN as PNG/SVG |

`toolbox/pipeline_cache.py` stores validated stages under `temp/pipeline_cache/`.
Semantic boundary analyses use `temp/semantic_boundary_cache/`. Valid entries are
reused. Raw trajectories retain frame indices; video-level descriptor CSVs cannot
replace them. Task grouping preserves original segment IDs, frame boundaries,
labels and provenance. Gemini suggestions require human review and confirmation.
The process model uses the confirmed task-level XES, without opening a viewer.

## Project layout

| Path | Purpose |
| --- | --- |
| `toolbox/` | Application entry point, processing, caches, aggregation and exports |
| `gui/` | Startup video selection, shared theme, segment labeling and semantic task review |
| `finch/` | FINCH implementation; local feature/cluster output folders |
| `IDT/` | Separate Git repository tracked as a submodule |
| `video_input/` | Original non-demo videos |
| `DemoFiles/`, `demo_RAVEE/` | Precomputed demo data and illustrations |
| `temp/` | Expensive completed-stage caches and semantic analyses |
| `EventLog_Output/` | Detailed/task event logs, review manifests and process models |
| `tests/` | Offline/mock regression suite and application import check |
| `docs/CLEANUP_INVENTORY.md` | Cleanup decisions, preserved candidates and IDT changes |

The root `test_gemini_labels_retry.py` and `test_label_suggester_v2.py` are
opt-in experiments outside the offline regression suite. The latter calls
the current V3 module. `make_contact_sheet.py` creates the demo illustration
when run explicitly.

## Setup

Use PowerShell from the project root. See [Install.md](Install.md) for fresh setup.
The current working environment uses Python 3.14 and `.venv`. Keep that environment
and its installed package versions when resuming existing caches. Source hashes
and library versions are part of cache fingerprints; even cosmetic edits to
fingerprinted processing/encoder files can change cache keys.

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
```

For optional AI suggestions, put `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your local
`.env`, or set it in the environment. `GEMINI_MODEL` selects the model; the code
currently defaults to `gemini-3.5-flash-lite`. Do not commit keys. Media settings
are `RAVEE_SUGGEST_MODE=video|frames` and `RAVEE_SUGGEST_CONTEXT_SECONDS` (0-10,
default 3). Manual labeling/review remains available without Gemini.

Multiple numeric/named credentials are supported through `GEMINI_ACTIVE_KEY`;
see `.env.example`. Optional `GEMINI_LABEL_MODEL` and `GEMINI_AGGREGATION_MODEL`
override the model independently. [Configuration, quota handling, intermediate
task objectives and comparison snapshots](docs/GEMINI_CONFIGURATION_AND_TASK_OBJECTIVES.md)
describe the explicit request workflow and preserved baselines.

Confirmed daily Gemini quota exhaustion opens a nonblocking recovery dialog in
both review GUIs: continue manually, explicitly select another configured
credential, or cancel the AI request while keeping review progress. Credentials
are never rotated automatically; completed cached work is retained.

## Run the application

Non-demo mode is selected by `DEMO = False` in `toolbox/main.py`.
Run the same entry point from IntelliJ with no program arguments, or from PowerShell:

```powershell
python -B .\toolbox\main.py
```

This opens **Select Video to Process** before running any pipeline stages. The
window lists `video_input/` videos, supports **Browse** for an external video,
**Refresh List**, and single selection. Cache inspection runs in the background:

| Status | Meaning |
| --- | --- |
| Cached | Preprocessing, raw IDT, Fisher Vectors and FINCH all verify |
| Partial Cache | A valid dependency chain exists, but a later stage is missing/invalid |
| New | No usable verified preprocessing cache exists for the exact source/settings |

Verification uses source SHA-256, source path, processing settings, code/package
fingerprints, completion manifests, artifact checksums and stage validators.
Merely selecting or browsing a video never starts processing. **Process Video**
authorizes missing preprocessing and Docker IDT extraction **for that one video**;
no `--allow-idt` is needed in GUI mode. Valid stages are reused, and missing stages
run in a worker while the window shows stage progress and errors. Failures leave
completed stages available for retry. After FINCH, the selector closes and the
existing granularity, labeling and semantic review GUIs open on the main thread.
Docker must be installed/running if extraction is actually needed. Closing an idle
selector exits; a running job must finish or fail before the window can close.

New CPU and CUDA IDT extractions use six workers with equal frame ranges and
a 30-frame overlap on each interior boundary. CPU is the default backend.
To select CUDA in the GUI or CLI, set `$env:RAVEE_IDT_BACKEND = 'cuda'` in
PowerShell before starting the app. Set `RAVEE_IDT_WORKERS` to a value from
1 to 6 to override the worker count. The backend and worker count are part
of the raw IDT cache identity. Explicit CPU legacy trajectory adoption uses
one worker by default.

### CLI without the startup selector

The original review/export workflow remains available explicitly:

```powershell
python -B .\toolbox\main.py --no-gui --video ".\video_input\example.mp4" --allow-idt
```

`--no-gui` (alias `--cli`) bypasses **only the startup selector**; labeling/task
review still uses its existing windows. Omit `--video` to process the input
directory as before. For an entirely noninteractive run through FINCH, with no
labeling, Gemini, review, event-log export or process-model rendering:

```powershell
python -B .\toolbox\main.py --no-gui --prepare-only --video ".\video_input\example.mp4" --allow-idt
```

Omit `--allow-idt` for a cache-only CLI resume (missing preprocessing/raw IDT raises
`CacheMiss`; missing FV/FINCH can still be computed). All previous flags remain:
`--adopt-existing-preprocessing`, `--adopt-existing-idt`, `--rebuild-idt`, and
`--allow-idt`. For backward compatibility, supplying any legacy processing flag
without `--gui` selects the original directory CLI mode. Use `--gui` to combine
legacy adoption/rebuild settings with the selector. `--video` and `--prepare-only`
select CLI mode and cannot be combined with `--gui`.

Legacy adoption still requires known source/configuration provenance and full
validation. `--rebuild-idt` rebuilds the image only when extraction is needed.
Existing v1 cache namespaces are retained; compatible older manifests are reused
only when their exact inputs, checksums and stage validators match. No cache
migration or deletion is required. To use the precomputed demo, set `DEMO = True`;
the demo retains its original review flow.

The labeling GUI proposes atomic activity labels. Semantic Task Aggregation then
groups adjacent phases toward observable objectives. Review merge/split/rename
operations before confirming; matching action labels or cluster IDs do not decide
tasks. Aggregation makes API requests only after an explicit GUI request. Refresh
rechecks semantic boundaries for the current video/case, preserving old valid
fallbacks when the service is unavailable. It does not rebuild pipeline stages.

In Segment Activity Labeling, **Suggest All** beside **Suggest label** sends the
current video and every FINCH segment in one generation request. Results appear
in the **Batch suggestion** column; **Use batch suggestion** fills the selected
editor without committing it. Reviewed labels and drafts remain unchanged until
you explicitly use/commit a suggestion. Batch results use a separate content-aware
cache and the shared credential/quota-recovery controls. See
[batch labeling behavior and limits](docs/BATCH_LABEL_SUGGESTIONS.md).

To review an existing detailed log without preprocessing, IDT or FINCH:

```powershell
python -B .\toolbox\semantic_aggregation.py --video "PATH_TO_EXISTING_REDUCED.avi" --detailed-csv .\EventLog_Output\all_videos_detailed_eventlog.csv --case-id 1 --output-dir .\EventLog_Output\semantic_review --base-name reviewed --offline
```

Replace the placeholder with the matching reduced video. `--offline` disables
Gemini; omit it only when you want to request AI suggestions. A saved confirmed
review loads as a draft and still requires confirmation. Detailed logs are
checkpointed before task review; canceling does not publish a new confirmation.
Normal review/export processing retains the filenames `all_videos_detailed_eventlog.csv/.xes`,
`all_videos_task_eventlog.csv/.xes` and task process-model PNG/SVG files.
In selector mode these aggregate filenames contain only the selected video.
Graphviz `dot` must be on PATH for image export; logs remain usable if rendering fails.

## Test without external work

Gemini remains the default. To select Pegasus independently for labeling or task
aggregation, see [optional TwelveLabs setup, uploads and cache behavior](docs/TWELVELABS_PROVIDER.md).
Its SDK is an optional dependency in `requirements-twelvelabs.txt`; existing
Gemini configuration and the `python toolbox/main.py` entry point are unchanged.

```powershell
python -B -m unittest discover -s tests -t . -v
```

The suite mocks Gemini, TwelveLabs, Docker, IDT, feature extraction, FINCH and Graphviz execution.
Regression serialization uses deterministic synthetic logs. Two checks reading
existing local logs or cached-video metadata require the explicit
`RAVEE_RUN_LOCAL_INTEGRATION=1` opt-in and are skipped by default. The `tests`
package isolates real dotenv/credentials, blocks unmocked Gemini clients and
protects production caches from writes. The import
check uses a fresh process and blocks application network, GUI, video and external
process work. Do not use repository-root test discovery to run the opt-in experiments.

See [the environment record and test isolation contract](docs/REPRODUCIBILITY.md)
for verified package versions, remaining portability limits and local check details.

Run an individual suite with its package name, for example:

```powershell
python -B -m unittest tests.test_semantic_aggregation tests.test_semantic_reliability -v
```

## IDT and preserved data

IDT has local source changes and data changes. A parent-repository commit only
records the submodule commit ID; it does not include those dirty files. Preserve
them separately before moving or cloning the project. See the exact inventory and
backup/commit options in [docs/CLEANUP_INVENTORY.md](docs/CLEANUP_INVENTORY.md).
`.gitmodules` declares the existing upstream URL; it does not publish local patches.
Do not run a forced submodule update over the current working tree.

Ignore rules keep new secrets, environments, caches, input media and generated
outputs out of normal Git additions. Previously tracked demo assets, outputs,
IDE settings and semantic-cache entries remain tracked and intact; changing that
tracking policy is a separate review decision. Parent ignore rules do not govern
the separate IDT repository.

## Further documentation and attribution

- [Non-demo pipeline and cache details](toolbox/NON_DEMO_PIPELINE.md)
- [Label Suggester V3 and process-model export](toolbox/PROCESS_MODEL_AND_LABELS_V3.md)
- [Semantic Task Aggregation and review](toolbox/SEMANTIC_AGGREGATION.md)
- [Original component mapping](Supplementary_Material.md)

`README.txt` contains historical V2 migration notes. Earlier implementation notes
may describe data available at that time; use actual cache manifests to determine
what is available now.

RAVEE builds on TW-FINCH, Improved Dense Trajectories, PM4Py and the IKEA Assembly
dataset. Original supplementary studies remain in the root PDF files. See
[LICENSE](LICENSE) for GNU GPL v3 licensing.
