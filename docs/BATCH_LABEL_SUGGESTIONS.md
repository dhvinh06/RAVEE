# Suggest All segment labels

`toolbox/main.py` calls `run_snippets()` in both demo and non-demo mode. That
function constructs the chronological FINCH segments and opens
`gui.gui_elements.label_video_clusters()`. This is the same Segment Activity
Labeling window used when IntelliJ launches `toolbox/main.py`.

The window has **Suggest label** and **Suggest All** beside each other. Individual
V3 suggestions keep their existing request, context media and validation behavior.
Suggest All runs `batch_label_suggester.suggest_all_activities()` on a worker and
sends the complete current video once, together with every 1-based segment ID,
inclusive frame boundary, FPS and corresponding time interval. Repeated FINCH
cluster IDs do not combine distinct segments. A normal cold request for 20
segments makes **one** Gemini `generate_content` call, never 20 calls. A valid
batch cache hit makes **zero** calls. Existing bounded transient retries and an
explicit quota-recovery credential selection can replay the same pending batch.

## Review and state preservation

Batch results appear in the **Batch suggestion** column for every segment. Select
a segment to inspect its suggestion; **Use batch suggestion** copies that one
suggestion into the editor. **Commit & Next / Commit & Finish** still saves the
human-reviewed label. Receiving results never writes to `LABEL`, changes the
selected segment or replaces the editor's current draft. Individual and batch
suggestions remain separate. Existing V3 suggestions can still be used.

Navigation, playback, draft editing and committing remain available during the
worker request. The two request controls share an in-flight guard, including
direct callback invocation. A batch cannot start while an individual request is
pending, and vice versa.

The shared label model, selected credential and `RequestSession` are reused.
Confirmed daily quota exhaustion uses the existing nonmodal recovery dialog.
Continue manually disables new AI requests while retaining review; Cancel keeps
the current review; explicit credential selection recreates the client and
resumes the same complete batch. There are no Files API uploads or remote file
IDs in this feature.

## Structured response and cache

The JSON response contains a `segments` array with exactly one item for each
expected `segment_id`. Every item includes the existing V3 action label, visual
evidence, uncertainty, visibility and bow-action fields. Validation rejects missing,
extra, duplicate or mistyped IDs, invalid labels/fields, malformed JSON, duplicate
JSON keys, blocked responses and truncated responses. The entire batch is accepted
or rejected together. Uncertain segments remain present as **Unclear activity**.

Only fully validated results are atomically cached under
`temp/batch_label_cache/`. Cache identity includes the video's content hash/size,
ordered IDs and frame boundaries, model, batch prompt, structured generation
configuration and media settings. Checksums and response validation also apply on
cache reads. Credentials and human label edits are excluded. Invalid/incomplete
entries cause one fresh whole-batch request. No individual, semantic, FINCH or
pipeline caches are read, migrated or deleted by this feature.

## Limits and offline verification

The application sends the original video inline, with a 64 MiB raw video cap
and 1 MiB prompt/manifest cap. It does not encode a shortened video, upload a large
file or silently fall back to per-segment generation. Unsupported containers,
invalid metadata, out-of-video boundaries and oversize media fail visibly while
manual review and individual suggestions remain available. Supported containers
are MP4/M4V, AVI, MOV, MPEG/MPG, WebM, WMV, 3GP and FLV. Video sampling is requested
at 3 FPS; very short actions can still be uncertain. Large batches can exceed a
model's context/output limits; incomplete responses are rejected, never committed.

```powershell
python -B -m unittest tests.test_batch_label_suggestions tests.test_quota_recovery tests.test_imports -v
```

Focused tests mock provider clients, video metadata and Tk widgets. They exercise
the actual GUI through both `toolbox.main.main()` branches using synthetic FINCH
assignments, and verify one generation for 20 segments, response rejection, cache
invalidation/reuse, preserving commits/drafts/selection, individual V3 coexistence,
duplicate prevention and the existing quota decisions. No live Gemini calls,
Docker/IDT, feature extraction or video processing occur. Native desktop rendering
and real model accuracy require a separate interactive check.
