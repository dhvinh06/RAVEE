# Gemini credentials and intermediate task objectives

The real `.env` is private and is not modified by this implementation. Start with
`.env.example`, supply your own credentials locally, and retain your existing
`GEMINI_MODEL` setting. Example variable names (values intentionally empty):

```dotenv
GEMINI_API_KEY=
GEMINI_API_KEY_2=
GEMINI_API_KEY_3=
GEMINI_API_KEY_RESEARCH=
GEMINI_ACTIVE_KEY=2
GEMINI_LABEL_MODEL=
GEMINI_AGGREGATION_MODEL=
```

`1` selects `GEMINI_API_KEY`; `2` selects `GEMINI_API_KEY_2`; `RESEARCH` selects
`GEMINI_API_KEY_RESEARCH`. Missing explicit selections fail instead of falling
back to a different project. Without `GEMINI_ACTIVE_KEY`, the previous precedence
`GOOGLE_API_KEY` then `GEMINI_API_KEY` is retained. Both stages use the same shared
client factory in `gemini_runtime.py` and the selected credential.

Configuration is reread at the next explicit AI request. Process environment
overrides `.env`; values initially loaded from `.env` are tracked so they do not
mask subsequent file edits. An in-flight aggregation run keeps its selected
client/model, rather than switching projects between batches. No key rotation
occurs on an error. Switching keys does not call or invalidate preprocessing,
IDT, FV or FINCH, and credential identity is excluded from local AI result keys.
Media settings retain their existing startup environment behavior.

Each stage uses its own optional model override, then `GEMINI_MODEL`, then the
unchanged existing default. Hydrogen-specific generation settings come from one
shared helper; other models retain their existing limits/settings. No model is
automatically selected based on quota or service errors.

## API errors and credentials

| Error | Behavior |
| --- | --- |
| HTTP 503 / transient 500, 502, 504 | At most four attempts, exponential full jitter |
| HTTP 429 with RPM/TPM evidence | Same bounded retry policy |
| HTTP 429 with RPD/daily quota evidence | Stop; report exhaustion; no automatic credential rotation |
| Ambiguous 429 | Stop; ask the user to inspect quota details instead of assuming it is transient |
| HTTP 401/403 | Stop; report authentication/permission failure |
| Invalid JSON/schema/evidence | Stop; preserve manual review and valid completed results |

`Retry-After` is respected, with at most 60 seconds total backoff. A daily quota
violation remains nonretryable even if RetryInfo is also present. SDK retries are
disabled to avoid multiplying application attempts. Selecting another authorized
project is a manual configuration decision subject to provider limits. Google
applies quota per project, so different keys in one project share its limits.

Redaction covers every configured profile, inactive profiles and credentials
already used during this session. Credentials are hidden in object reprs and
are excluded from caches, result metadata and GUI messages. Existing evidence
validation and redacted diagnostics remain active.

## Project resources and uploads

Production V3 and aggregation send small inline clips/stills; they do not upload
or cache Gemini Files API identifiers. The opt-in `test_gemini_labels_retry.py`
prototype does use remote files. Its client now uses the same selected profile.
Every `--reuse-file` reference is checked with `files.get` under that selected
client before generation. Google's API checks the requesting project's ownership
and resource availability. A foreign/expired reference fails without automatically
uploading a replacement. Compatible resources, including access via another key
in the same project, can be reused after verification.

A new upload requires explicit `--allow-upload`. Local videos/caches remain
unchanged. Prototype result files receive unique names when a prior output exists.

## Intermediate work objectives

Segment means a FINCH action interval. Task means adjacent actions serving a
distinct intermediate work objective or observable intermediate result. Process
means the complete workflow, potentially containing several tasks on one object.

The prompt now makes this hierarchy explicit. Each boundary additionally reports
`objective_before`, `objective_after`, `objective_transition`, `objective_scope`
and `transition_evidence`. These must explain the immediate results on either
side, and whether the next action continues/retries/corrects/checks that result or
starts another. Object identity or a shared process goal alone cannot authorize
a task merge. Different verbs can still be phases of the same intermediate task.
There is no reference activity vocabulary, segment-count split or required number
of tasks.

The batched request still evaluates all consecutive boundaries; eleven segments
can be evaluated in one request returning ten decisions. Existing guards reject
missing/reordered decisions and preserve uncertainty. New explicit process-only
or changed-intermediate-objective evidence cannot support an automatic join.
Legacy boundary fields remain readable; new fields are additive and typed.

Diagnostics are advisory: a proposal spanning contradictory objectives or only
process-level reasoning is flagged. A whole-sequence proposal lacking immediate
objective evidence is flagged for inspection. A supported single task is allowed.
Warnings do not split by count, issue another request or prevent confirmation.
Boundary evidence is inspectable in the source-details pane. Original raw group
proposals and the protected result are both retained in the AI audit snapshot.

## Controlled comparisons and preserved outputs

Hold the confirmed detailed CSV fixed when comparing aggregation models. Change
only `GEMINI_AGGREGATION_MODEL`, open that existing log with the review CLI, and
explicitly request suggestions. Compare records/context/prompt hashes, not task
counts. Labeling does not need to be repeated.

Every explicit proposal invocation saves a uniquely named JSON snapshot under
`temp/semantic_boundary_cache/runs/`. It records model, actual generation config,
prompt hash, reviewed input hash, video identity, reviewed records, proposed and
protected groups, boundary decisions/sources, warnings and request statistics.
No credential identity is included. `semantic_runs/<unique-run>/` beside review
outputs preserves the reviewed CSV, configuration, confirmed groups and task CSV.
Confirmed manifests record request provenance and whether suggestions were
requested; configuration alone does not prove a model produced a manual result.

Before existing event logs, process-model images or confirmed manifests are
replaced at compatibility paths, their old bytes are copied into a unique
`semantic_history/` directory. Failures to preserve an existing baseline stop
replacement rather than silently discard it. Old review/cache formats remain
readable. The changed task prompt/generation contract naturally creates new AI
cache identities; old files are retained. Changing credentials alone does not.

## Evidence and verification limits

Code inspection verified that every boundary must be assessed and that a chain
of model judgments reporting one objective can form one group; no postprocessor
forces an entire video into one group. The former contract did not demand
immediate before/after results, leaving room for a broad process interpretation.
The reported run's review and 60 boundary files were unreadable in the runner;
its actual model decisions and model identity could not be reconstructed.
Overly broad model reasoning is a hypothesis for that run, not a proven cause.

All development verification is offline with synthetic fixtures and mocked
clients. No provider requests, uploads, video processing or IDT changes occur.
Live quota payloads, project access and revised task accuracy still need testing
by an authorized user. Native GUI visual checks remain limited by the runner's
Tcl/Tk initialization problem.

Official references:

- [API keys and project association](https://ai.google.dev/gemini-api/docs/api-key)
- [Project rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [API errors](https://ai.google.dev/gemini-api/docs/api-errors)
- [Files API ownership](https://ai.google.dev/api/files)
