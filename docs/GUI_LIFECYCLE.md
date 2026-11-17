# GUI transition ownership

The normal `toolbox/main.py` entry point now owns one lazy hidden Tk interpreter.
The selector, granularity, segment labeling and task review use successive
Toplevel screens on that interpreter. Each screen's mainloop returns before the
next screen starts; no Python mainloops are nested. `--prepare-only` never creates
a Tk root. Standalone GUI helper calls still work with their own managed root.

## Verified failure paths

Before this fix, the labeling GUI scheduled `poll_suggestions` every 150 ms without
retaining/canceling its `after` ID. Commit & Finish called `root.destroy()` inside
the callback. Cleanup subsequently ran against an already-destroyed window, while
the poll could remain in Tcl's event queue. Entering the next review mainloop
could execute the old script after its registered Python/Tcl command was deleted.

AI workers did not directly call widgets, but their nested functions captured
`QuotaRecovery` and `UploadConsent`. Those UI controllers held roots and closures
containing widgets/Variables. Releasing a worker's captures, or cyclic garbage
collection on a worker, could therefore finalize old Tcl resources off the GUI
thread. Tkinter Variable and ImageTk destructors themselves call Tcl, so checking
only explicit `.set()`/`.configure()` calls was insufficient.

Several StringVars/PhotoImages also relied on the implicit default interpreter
while separate roots were created/destroyed across stages. Those resources now
have explicit owners.

## Ownership rules

`gui/lifecycle.py` supplies:

- `AfterCallbacks`: retain each after/idle ID, remove it when dispatched, cancel
  remaining IDs on close and reject rescheduling after shutdown starts. A wrapper
  already dispatched before close becomes a no-op. Callback errors propagate.
- `WindowLifetime`: explicit variable/image ownership, quit first, destroy only
  in the owning function's finally block, release native resources on the main
  thread, then detach destroyed widgets' interpreter references.
- `WorkerInbox`: a thread-safe pure-Python endpoint that drops/drains events after
  close, so late results cannot target a destroyed screen or a subsequent case.
- `AIJob`/`run_ai_job`: a module-level worker target with frozen data/options.
  Workers receive neither a root, widgets, Tk Variables nor UI controller methods.
- `DecisionChannel`: pure-Python Event/lock requests for quota/upload decisions.
  The UI controller polls it on the main thread. Closing the owner cancels its
  poll, resolves pending waits and releases UI references.

The GUI recovery/upload properties return bound **channel** methods, not methods
bound to a UI controller. Both Gemini and TwelveLabs use the same ownership rule.
No widget operation, Variable mutation or `root.after()` call is made by these
workers. The video preparation controller also drops events after closure.

Tkinter/Pillow expose no public dispose API for their Python variable/photo
wrappers. Cleanup invokes the existing variable teardown on the owner thread,
then clears its private interpreter/root fields; photo handles and both photo
interpreter references are likewise cleared. Owning only the current preview and
disposing its predecessor on replacement avoids retaining every playback frame.
This prevents a later Python GC on any thread from making
native Tcl calls. These operations are confined to shutdown/fully destroyed
widgets; closed callbacks never reuse them. Tests exercise this destructor
contract rather than masking Tcl errors with broad exception handlers.

## Shutdown and data safety

Finish/close marks the screen closed and cancels owned timers before returning
from its event loop. Finally closes the inbox and decision channels, stops
playback, releases capture/images/Variables, then destroys only that screen. The
hidden interpreter is destroyed on the main thread when the application exits.
Dialog and external destroy notifications are handled idempotently.

An SDK call already in flight is not forcibly killed or joined on the GUI thread.
It may finish/cache its already-authorized response; its later UI events are
dropped. Waiting quota/upload requests are released without approval or credential
switching, and jobs not started before their inbox closed do not initiate work.
This avoids blocking the next review while preserving completed provider caches.

Label commitment, pending-draft checks, task review/Undo and CSV/XES export order
are unchanged. No FINCH, FV, prompts or cache algorithms were modified.

## Offline checks

```powershell
python -B -m unittest tests.test_gui_lifecycle -v
python -B -m unittest discover -s tests -t . -q
```

Strict desktop mocks reject wrong-thread and post-destroy widget access. Tests
close while individual/Suggest All workers are active, open Task Review before a
late response finishes, invoke stale callback wrappers, repeat completed batch
transitions for both provider selectors, exercise selector shutdown and verify the
actual entry point shares one lazy owner. Existing quota/review tests retain
their state-preservation assertions.

Isolated native Tcl/Tk subprocess tests additionally check destructor/GC behavior
and repeated hidden screens. They explicitly skip only unavailable native runtime
initialization/display support. The current execution environment can read the
installed Tcl scripts but cannot load a usable `init.tcl`; these two native checks
therefore skip here. Deterministic Tk-variable/photo disposal and thread/transition
tests still run. This environment limitation is not suppressed in application
code and is not a new requirement to alter the user's Python installation.

No live providers, video preprocessing/extraction, FINCH or Docker are invoked.
