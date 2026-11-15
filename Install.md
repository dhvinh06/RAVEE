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

The last command resumes cached processing and opens the labeling/review GUIs.
It does not allow missing Docker/IDT extraction automatically.

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
Docker Desktop must be installed/running only when you explicitly allow missing
IDT work with `--allow-idt`. Do not replace the current IDT directory to initialize
it; it already contains important local modifications and data.

Optional Gemini suggestions use `GEMINI_API_KEY` or `GOOGLE_API_KEY`, loaded from
the environment or a local `.env`. `GEMINI_MODEL` overrides the configured default.
Manual labels and task review do not require an API key. Never commit `.env`.

See [README.md](README.md) for architecture, launch commands, cache adoption,
standalone task review, exports and test commands.
