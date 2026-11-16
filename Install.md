# Installation

## Resume the existing working environment

The current project runs with Python 3.14 in `.venv` on Windows. Keep this
working environment and its package versions unchanged when reusing expensive
caches. No dependency installation or upgrade is part of the cleanup.

From the RAVEE root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
python -B -m unittest discover -s tests -t . -v
python -B .\toolbox\main.py
```

The last command opens the interactive **Select Video to Process** window.
Select or browse to one video, then click **Process Video**. Valid completed
caches are reused; that click authorizes missing preprocessing/IDT stages for
the selected video. Selecting/browsing alone does not start extraction. Docker
must be running if missing IDT extraction is needed. After processing, the
granularity, segment-labeling and task-review windows open.

### CLI execution and authorization

To bypass only the startup selector and retain labeling/task-review windows:

```powershell
python -B .\toolbox\main.py --no-gui --video ".\video_input\example.mp4" --allow-idt
```

`--cli` is an alias for `--no-gui`. `--allow-idt` authorizes missing extraction
in CLI mode. Without it, missing preprocessing/raw IDT raises `CacheMiss`;
valid caches are reused, while missing FV/FINCH may still be computed. Omitting
`--video` preserves the original directory-processing CLI behavior.

For a noninteractive preparation run that stops after FINCH, without labeling,
Gemini, task review or export:

```powershell
python -B .\toolbox\main.py --no-gui --prepare-only --video ".\video_input\example.mp4" --allow-idt
```

Keep the existing `--adopt-existing-preprocessing`, `--adopt-existing-idt` and
`--rebuild-idt` options. Legacy processing flags select CLI mode unless `--gui`
is explicit. `--video`/`--prepare-only` cannot be combined with `--gui`.

## Fresh environment on another machine

Install a Python version supported by the dependencies and including Tkinter.
The existing environment is the reference; this project does not yet have a
portable, fully locked dependency specification. To create a separate environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pandas scipy scikit-learn opencv-python moviepy matplotlib pm4py Pillow google-genai python-dotenv
$env:PYTHONPATH = (Get-Location).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
```

These commands are for a new installation, not an instruction to replace the
current `.venv`. Different library versions can produce different cache keys.
PyNNDescent is optional for smaller FINCH inputs and required for approximate
nearest neighbors above FINCH's configured threshold. Experimental model scripts
separately require packages such as PyTorch and Transformers; they are not part
of the normal offline regression suite. The active GUIs use Tkinter, not PyQt5.

Install Graphviz and put `dot` on PATH for PNG/SVG process-model exports. MoviePy
uses an FFmpeg backend. OpenCV needs a working MP4 encoder for video-mode Gemini
requests; the current implementation also has a still-frame fallback.

IDT is a Git submodule pointing to the Improved Dense Trajectories Docker wrapper.
A fresh recursive clone retrieves the upstream pinned revision, but that revision
alone does not include this working copy's uncommitted RAVEE patches. Preserve
and transfer the local IDT source changes described in
[the cleanup inventory](docs/CLEANUP_INVENTORY.md) before trying new extraction.
Docker Desktop must be installed/running when missing IDT work is authorized by
**Process Video** in GUI mode or `--allow-idt` in CLI mode. Do not replace the current IDT directory to initialize
it; it already contains important local modifications and data.

Optional Gemini suggestions use `GEMINI_API_KEY` or `GOOGLE_API_KEY`, loaded from
the environment or a local `.env`. `GEMINI_MODEL` overrides the configured default.
`GEMINI_LABEL_MODEL` and `GEMINI_AGGREGATION_MODEL` override the model per stage;
otherwise each uses `GEMINI_MODEL` and then the existing code default.
`GEMINI_ACTIVE_KEY` selects a numeric/named credential from `.env.example`.
Confirmed daily quota exhaustion opens the existing recovery dialog; credentials
are switched only by explicit user selection, never automatically.
Manual labels and task review do not require an API key. Never commit `.env`.

**Suggest All** sends the full current video inline in one generation request.
Its verified application limit is **64 MiB (67,108,864 bytes)** of raw video plus
a 1 MiB prompt/manifest limit. Larger videos fail without an upload, re-encoding
or per-segment fallback. Individual suggestions/manual review remain available.
See [batch behavior and limits](docs/BATCH_LABEL_SUGGESTIONS.md).

## Offline regression and optional local checks

The `tests` package isolates dotenv discovery and Gemini environment variables,
uses a fake default credential, blocks unmocked Gemini clients and forbids writes
to production research/cache directories. All generated test data stays in OS
temporary directories. The serialization regression uses deterministic synthetic
current/legacy logs, independent of the latest experiment output.

Two checks that read a local XES or preprocessing-video header are skipped by
default. To opt into those artifact-dependent checks explicitly:

```powershell
$env:RAVEE_RUN_LOCAL_INTEGRATION = '1'
python -B -m unittest tests.test_process_model_and_labels_v3 -v
Remove-Item Env:RAVEE_RUN_LOCAL_INTEGRATION
```

They still mock Gemini, discovery/rendering and never reprocess video or run IDT.
Missing/unreadable local fixtures are reported as skips where supported. The
normal offline command above does not require these research artifacts.

The verified package inventory, Docker observations and remaining portability
limits are recorded in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). That
inventory is not a dependency lock and is not an instruction to upgrade packages.

See [README.md](README.md) for architecture, launch commands, cache adoption,
standalone task review, exports and test commands.
