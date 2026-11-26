# Optional TwelveLabs Pegasus provider

Gemini remains the default for both AI stages. Pegasus uses the same FINCH
segments, label controls, task-boundary validation and human-review workflow.
It never runs TwelveLabs automatic segmentation or changes the video pipeline.

## Setup and selection

The optional SDK is not installed by this integration. In the working virtual
environment, install it explicitly when you want to use Pegasus:

```powershell
python -m pip install -r requirements-twelvelabs.txt
```

The optional specification pins `twelvelabs==1.3.8`, verified in the official
[SDK pyproject](https://github.com/twelvelabs-io/twelvelabs-python/blob/main/pyproject.toml).
This is not a complete transitive environment lock or a certification on Python
3.14. No packages were installed/upgraded during implementation.

Add these values to your local `.env`, preserving existing Gemini values:

```dotenv
TWELVELABS_API_KEY=your_private_key
TWELVELABS_MODEL=pegasus1.6
RAVEE_LABEL_PROVIDER=gemini
RAVEE_AGGREGATION_PROVIDER=gemini
```

Set either provider to `twelvelabs` independently. For example,
`RAVEE_LABEL_PROVIDER=gemini` and `RAVEE_AGGREGATION_PROVIDER=twelvelabs` compares
Pegasus grouping with Gemini-generated, human-reviewed activity labels.
Both unset provider selectors default to Gemini. Invalid selectors fail clearly;
an API error never switches providers. `python toolbox/main.py` remains the entry
point. The AI status area identifies the selected provider/model in both review
windows. Label suggestions held in memory are scoped by provider/model; switching
back retains the corresponding proposals without changing labels or drafts.

Gemini's existing per-stage models, multi-key selection, quota recovery and cache
identities remain intact. TwelveLabs has one configured credential here, not a
credential-rotation system. Its confirmed plan/credit exhaustion dialog offers
manual continuation/cancel; Gemini credential selection is disabled there.

## Asset upload and reuse

The adapter uses the official `TwelveLabs` SDK and synchronous `client.analyze`
with `model_name`, `VideoContext_AssetId` and `SyncResponseFormat(json_schema)`.
It uses `assets.create(method="direct", file=...)` for local files and polls
`assets.retrieve` until ready. Relevant official references:

- [Quickstart](https://docs.twelvelabs.io/docs/get-started/quickstart/analyze-videos-and-images)
- [Analyze API](https://docs.twelvelabs.io/api-reference/analyze-videos/analyze)
- [Python Analyze signature](https://docs.twelvelabs.io/v1.3/sdk-reference/python/analyze-videos/sync-analysis)
- [Direct upload requirements](https://docs.twelvelabs.io/v1.3/sdk-reference/python/upload-files/direct-uploads)
- [Structured response constraints](https://docs.twelvelabs.io/docs/guides/analyze-videos-and-images/structured-responses)

Assets are cached under `temp/twelvelabs/assets/`, shared by default across
individual labels, Suggest All and aggregation. The manifest records content
SHA-256, byte size, asset ID, an opaque credential-scope digest and a checksum;
it stores no API key. The video hash is also sent as `ravee_sha256` user metadata.
Reuse requires the matching manifest, authenticated retrieval with the current
key and matching remote video-hash metadata. A failed asset or permission error
is reported, not silently adopted or reuploaded. A confirmed 404 permits a new
upload during the explicit AI request. Different credentials use different asset
namespaces conservatively, even if the keys belong to the same account.

The reference is saved immediately after upload, before readiness polling, so a
later explicit retry can resume a still-processing asset. Asset writes can fail
on a read-only cache; warnings identify the loss of persistent reuse. Server
assets are never deleted by RAVEE. The asset lock prevents simultaneous duplicate
uploads inside this process; it is not a cross-process upload lock.

New uploads above **64 MiB** require explicit GUI approval. Programmatic callers
must supply an `on_upload(path, size)` callback returning true; otherwise the
operation fails before upload. Previously uploaded, verified assets can be reused
without uploading again. Local direct uploads are conservatively capped at
**200,000,000 bytes**, matching the documented 200 MB limit. Multipart/public-URL
uploads are not implemented. No automatic re-encoding or splitting occurs.

Header inspection checks FPS, frame count, duration, resolution and aspect ratio
before a new analysis. The synchronous path accepts 1-second to 1-hour videos
within the documented resolution/aspect constraints. Unsupported inputs fail
without changing preprocessing or FINCH results.

## Labeling and task contracts

Individual Suggest supplies the complete asset with a target manifest identifying
the exact FINCH frame range. It uses `start_time`/`end_time` with surrounding
context when the container time origin can be verified. If `ffprobe` is available,
RAVEE reads only `format.start_time` metadata (no frame extraction). Non-zero
origins offset the API window, not FINCH frames. Without a verified origin, the
request analyzes the complete asset and explicitly limits its label to the target
manifest. Subsecond targets retain their frames; the analysis context window is
padded to satisfy the API's minimum one-second window.

Suggest All sends the complete video asset and ordered FINCH IDs/boundaries in
**one Analyze request**, including all 20 IDs for a 20-segment video. Asset upload
and readiness reads are separate API operations. A warm result cache makes zero
Analyze calls. Transient retries can repeat an Analyze request; there is no loop
of one generation per segment and no provider-generated segmentation.

Pegasus's remote schema omits unsupported exact-array-size constraints, while
RAVEE validates the original complete schema locally, including exact batch
cardinality, unique/known IDs, field types and evidence. Duplicate JSON fields,
nonfinite values, truncated/unfinished responses and partial results are rejected
before result caching. Existing V3 label validation/canonicalization is shared.

Task aggregation supplies reviewed labels, timestamps, exact frame/time boundaries,
neighboring segment context and the full video asset. Existing batching, boundary
validation, intermediate-objective vetoes, uncertainty handling and review/Undo
remain in place. Invalid contracts are reported as unresolved boundaries for
manual review; process-level rationales cannot override intermediate task
boundaries. Output proposals never become confirmed groups automatically.
The recorded label-provider/model fields describe the current configuration;
they are not proof that an earlier human-reviewed label originated from that
model. The reviewed label text itself is included in the input fingerprint.

## Result caches and failures

Pegasus results use `temp/twelvelabs/individual/`, `batch/` and
`semantic_boundaries/`. An explicit custom cache directory gets a `twelvelabs/`
subdirectory. Gemini's historical cache directories/identities are unchanged.
TwelveLabs identities include provider, model, prompt, generation settings,
schema, input video content and the relevant boundaries/reviewed labels/window.
Aggregation retains per-boundary checksums and comparison-run snapshots.
Verified result-cache hits require no SDK client, upload or remote analysis.
Existing caches, review manifests, event logs and process models are not deleted.

The SDK request timeout is 120 seconds, and automatic SDK retries are disabled.
The adapter permits up to three attempts for known temporary rate limits and
500/502/503/504 failures, with bounded waits (1/2 seconds, or Retry-After up to
30 seconds). Ambiguous upload outcomes are never automatically retried.
Readiness polling has a ten-minute deadline, plus any in-flight bounded request.
Timeouts and unknown 429 responses require an explicit later retry. Authentication,
confirmed plan allowance/credit exhaustion and invalid responses are distinct.
No generic 429 is called daily quota exhaustion. Errors redact current and
previously used credentials. No automatic key/provider switching occurs.

## Offline verification and remaining evaluation

```powershell
python -B -m unittest tests.test_twelvelabs_provider -v
python -B -m unittest discover -s tests -t . -q
```

Tests mock SDK construction, responses, uploads/readiness, metadata and Tkinter.
They cover both AI stages, complete batch labels, cache isolation/integrity,
boundary failures, provider configuration, retries, quota recovery and preservation
of human edits. The offline fixture removes real TwelveLabs/Gemini configuration
and rejects unmocked clients; production `.env` and caches are protected.

SDK installation/import compatibility, account permissions, actual upload latency,
server schema acceptance, provider charges and label/group quality still require
an explicitly authorized live evaluation. No API calls, preprocessing, extraction,
FINCH or Docker runs were made for this integration. Pegasus has not been shown
to produce better labels; compare reviewed outputs under recorded configurations.
