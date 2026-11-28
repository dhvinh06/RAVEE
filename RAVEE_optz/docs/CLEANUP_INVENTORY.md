# Safe cleanup inventory — 2026-11-15

This inventory distinguishes generated clutter from project data. Cleanup resumes
an interrupted working-tree change; existing user edits are retained. No commit,
push, history reset, submodule update, Gemini request or Docker/IDT run is part of
this work.

## Safe changes

| Files | Action and reason |
| --- | --- |
| `test_event_log_serialization.py`, `test_pipeline_cache.py`, `test_process_model_and_labels_v3.py`, `test_semantic_aggregation.py`, `test_semantic_reliability.py` | Move the five offline regression suites into `tests/`; adjust repository-root paths and the shared test import. Keep assertions and mocks. |
| `tests/__init__.py` | Make the suites importable as a package. |
| `tests/test_imports.py` | Check 22 application/utility import paths in a fresh process; block application network, SDK client creation, GUI, video opening, image writing and external processes. |
| `tests/test_process_model_and_labels_v3.py` | Skip the optional personal cached-video fixture when absent or unreadable; retain mocked context-window tests. |
| `README.md`, `Install.md` | Document current architecture, existing environment, startup, offline tests, exports, cache preservation and dirty submodule handling. |
| `README.txt`, `integrate_label_suggester.py` | Mark V2 migration instructions as historical; preserve their contents and filenames. |
| `toolbox/NON_DEMO_PIPELINE.md`, `toolbox/PROCESS_MODEL_AND_LABELS_V3.md`, `toolbox/SEMANTIC_AGGREGATION.md` | Update test commands to package paths under `tests/`. |
| `gui/gui_elements.py`, `gui/semantic_review.py` | Use “Segment Activity Labeling”, “Semantic Task Aggregation” and “segment ID” consistently; clarify the review module docstring. No callback or decision changes. |
| `make_contact_sheet.py` | Put execution under `main()` and a script guard so importing the utility cannot open a video or overwrite the illustration. |
| `.gitignore` | Ignore new secrets, environments, bytecode, IDE state, backups, media, model artifacts, cache folders and generated exports. Do not untrack existing data. |
| `.gitmodules` | Declare the existing `IDT` path and its observed upstream remote; retain the pinned gitlink revision. |
| 19 tracked `*.pyc` in root, `gui/`, `toolbox/`, `finch/` bytecode folders | Removed generated Python bytecode only. Python can regenerate it; Git retains the committed version. Source files and environment files remain intact. |

Processing modules retain their existing names and public interfaces. In
particular, do not edit preprocessing, feature extraction, cache logic, FINCH,
label prompts or semantic grouping logic for cosmetic cleanup: cache keys include
source hashes. The IDT build signature includes source/configuration files, so
even an extra README or ignore rule inside IDT can invalidate reuse.

## Preserve; approval required before deletion or relocation

| Candidate | Why it remains |
| --- | --- |
| `label_suggester_v1.py.bak`, `gui/gui_elements.py.bak` | Historical source backups with uncertain rollback value. |
| `integrate_label_suggester.py` | Historical migration utility; may still support old installations. |
| `test_gemini_labels_retry.py`, `test_label_suggester_v2.py` | Opt-in label-suggestion experiments. The V2-named script imports the current suggester. |
| `README.txt`, `Supplementary_Material.md`, supplementary PDFs, `demo_RAVEE/` | Historical documentation, attribution and research/demo material. |
| Tracked `.idea/` files | Existing personal IDE configuration; ignored for new additions, not removed. |
| `temp/`, `IDT/features/`, `IDT/data/`, `video_input/`, `finch/input/`, `finch/output/`, `DemoFiles/`, `EventLog_Output/` | Expensive caches, raw trajectories, input/reduced media, trained/encoded artifacts, original segment data, illustrations and reviewed exports. |

No duplicate utility has been proved interchangeable enough to remove. No
blanket “temporary folder” cleanup is safe: `temp/` contains expensive results.
Ignore rules do not remove files already tracked by Git. Any later decision to
untrack user data must preserve the working files and receive separate review.

The working tree already contains edits to
`EventLog_Output/semantic_review/gift_refined_semantic_groups.json` and the
matching task CSV/XES. They are user results and are not cleanup changes. The JSON
cannot be read by the execution account; no permission changes are attempted.

## IDT status and preservation

The parent gitlink points to `499ea2027ad416e5a13cd64fad90d541e6be2f00`.
The observed remote is
`https://github.com/AraMambreyan/Improved-Dense-Trajectories.git`.

| IDT path | Observed local change | Preservation |
| --- | --- | --- |
| `Dockerfile` | Pins pandas to `1.5.3`. | Keep as source/configuration. |
| `run_idt.py` | Supports `RAVEE_RAW_ONLY` and atomic CSV/trajectory publication. | Keep as source; required by current cached-pipeline integration. |
| `.dockerignore` | Untracked Docker configuration. | Copy explicitly; an ordinary Git diff does not include it. |
| `features/mbh_features.csv` | Modified descriptor output. | Back up as data; do not discard or overwrite. |
| `data/1_3197334-hd_1920_1080_25fps_reduced.avi` | Untracked reduced input video. | Back up as media; an ordinary Git diff does not include it. |

All five paths remain untouched. A parent-repository commit records only the IDT
commit ID; it does not save these dirty or untracked files. Before transferring
the project, copy the complete IDT working directory and the expensive data/cache
directories to a separate backup. A source-only alternative is to export a binary
Git patch for tracked source changes and separately copy `.dockerignore`; also
copy the modified CSV, videos, raw trajectories and any other useful outputs.
Verify the backup before changing the original.

For a later versioned solution, review and commit only the intended source/config
patches inside IDT, make that revision available in a remote you control, and then
update the parent gitlink and submodule URL to that reachable revision. Keep large
generated data in the separate backup. These are instructions for later review;
no such Git mutation is performed during cleanup. Avoid forced submodule updates,
`git clean`, resets or replacing the IDT directory over the current dirty tree.

## Verification

Run from the root with the existing environment:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -t . -v
```

On 2026-11-15: **102 tests discovered, 101 passed, 1 skipped** because the existing
personal preprocessing video is unreadable by the execution account. This
includes the guarded import check for 22 application/utility paths. Existing
mocked cache, FINCH orchestration, Gemini, aggregation, human review and export
tests pass. No real model inference, feature extraction or Graphviz rendering is
required. This validates offline behavior and imports, not live GUI appearance or
Gemini/video quality.

The five relocated suites were compared with their committed originals: only
repository-root resolution, the shared test import and the optional unreadable
video-fixture skip changed. Nine protected processing/prompt/export source files
match HEAD after normalizing line endings. All root, application and test Python
sources parse successfully. The IDT gitlink and working-tree changes are retained.

Git status is inspected with a command-local ownership override because the
sandbox account differs from the repository owner. Global Git configuration is
unchanged. Whole-tree `git diff --stat` cannot hash the unreadable user review
JSON; source-specific diffs remain available.
