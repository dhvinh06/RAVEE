# RAVEE

RAVEE turns videos into reviewed activity segments, semantic tasks and event logs
for process mining. This working fork adds cached non-demo processing, Gemini
Label Suggester V3, human-reviewed Semantic Task Aggregation and process-model export.

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
| `gui/` | Segment labeling and semantic task review |
| `finch/` | FINCH implementation; local feature/cluster output folders |
| `IDT/` | Separate Git repository tracked as a submodule |
| `video_input/` | Original non-demo videos |
| `DemoFiles/`, `demo_RAVEE/` | Precomputed demo data and illustrations |
| `temp/` | Expensive completed-stage caches and semantic analyses |
| `EventLog_Output/` | Detailed/task event logs, review manifests and process models |
| `tests/` | Offline/mock regression suite and application import check |
| `docs/CLEANUP_INVENTORY.md` | Cleanup decisions, preserved candidates and IDT changes |

The root `test_clip_roi.py`, `test_timesformer.py`, `test_videomae.py`,
`test_gemini_labels_retry.py` and `test_label_suggester_v2.py` are opt-in experiment
scripts, not offline regression tests. Some load models, inspect videos or call
Gemini. The script named `test_label_suggester_v2.py` calls the current V3 module;
its filename remains unchanged for CLI compatibility. `make_contact_sheet.py`
creates the demo illustration only when run explicitly.

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

## Run the application

Non-demo mode is currently selected by `DEMO = False` in `toolbox/main.py`.
Place source videos in `video_input/`, then resume verified caches with:

```powershell
python -B .\toolbox\main.py
```

This opens granularity selection, segment labeling and task review as needed.
Missing preprocessing or raw IDT cache stops with guidance; Docker extraction
requires explicit opt-in. For a new input, only when you intend to do expensive work:

```powershell
python -B .\toolbox\main.py --allow-idt
```

Docker must be running for that command if extraction is needed. Existing raw
outputs can be validated and adopted with `--adopt-existing-preprocessing` and
`--adopt-existing-idt` when their origin/configuration is known. `--rebuild-idt`
forces an image rebuild only when extraction is necessary. Do not delete caches
as a troubleshooting step. To use the precomputed demo, set `DEMO = True`.

The labeling GUI proposes atomic activity labels. Semantic Task Aggregation then
groups adjacent phases toward observable objectives. Review merge/split/rename
operations before confirming; matching action labels or cluster IDs do not decide
tasks. Aggregation makes API requests only after an explicit GUI request. Refresh
rechecks semantic boundaries for the current video/case, preserving old valid
fallbacks when the service is unavailable. It does not rebuild pipeline stages.

To review an existing detailed log without preprocessing, IDT or FINCH:

```powershell
python -B .\toolbox\semantic_aggregation.py --video "PATH_TO_EXISTING_REDUCED.avi" --detailed-csv .\EventLog_Output\all_videos_detailed_eventlog.csv --case-id 1 --output-dir .\EventLog_Output\semantic_review --base-name reviewed --offline
```

Replace the placeholder with the matching reduced video. `--offline` disables
Gemini; omit it only when you want to request AI suggestions. A saved confirmed
review loads as a draft and still requires confirmation. Detailed logs are
checkpointed before task review; canceling does not publish a new confirmation.
Normal processing exports `all_videos_detailed_eventlog.csv/.xes`,
`all_videos_task_eventlog.csv/.xes` and task process-model PNG/SVG files.
Graphviz `dot` must be on PATH for image export; logs remain usable if rendering fails.

## Test without external work

```powershell
python -B -m unittest discover -s tests -t . -v
```

The suite mocks Gemini, Docker, IDT, feature extraction, FINCH and Graphviz execution.
Some integration tests read existing local logs or cached-video metadata; an
optional local fixture can be skipped when unavailable or unreadable. The import
check uses a fresh process and blocks application network, GUI, video and external
process work. Do not use repository-root test discovery to run the opt-in experiments.

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
