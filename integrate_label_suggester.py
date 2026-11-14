"""One-time, conservative integration of Label Suggester into gui/gui_elements.py.

Run from the RAVEE root with:
    python integrate_label_suggester.py

It preserves run_snippets()/FINCH and creates gui_elements.py.bak.
It refuses to edit if the expected source anchors have changed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "gui" / "gui_elements.py"
BACKUP = TARGET.with_suffix(".py.bak")


def replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Cannot integrate {name}: expected exactly one anchor, found {count}.")
    return source.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(f"Missing {TARGET}. Copy this script to the RAVEE project root.")
    if not (ROOT / "label_suggester.py").is_file():
        raise FileNotFoundError("Copy label_suggester.py to the RAVEE project root first.")

    original = TARGET.read_text(encoding="utf-8")
    if "# RAVEE_LABEL_SUGGESTER_INTEGRATED" in original:
        print("Label Suggester is already integrated. No changes made.")
        return

    source = original
    source = replace_once(
        source,
        "from tkinter import ttk, messagebox\n",
        "from tkinter import ttk, messagebox\n"
        "from queue import Queue, Empty\n"
        "from threading import Thread\n"
        "from label_suggester import suggest_activity  # RAVEE_LABEL_SUGGESTER_INTEGRATED\n",
        "imports",
    )
    source = replace_once(
        source,
        '    play_button = ttk.Button(navigation, text="Play")\n'
        '    play_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))\n',
        '    play_button = ttk.Button(navigation, text="Play")\n'
        '    play_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))\n'
        '    navigation.columnconfigure(2, weight=1)\n'
        '    suggest_button = ttk.Button(navigation, text="Suggest label")\n'
        '    suggest_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))\n',
        "suggest button",
    )
    source = replace_once(
        source,
        '    tree_scroll.grid(row=9, column=1, sticky="ns")\n',
        '    tree_scroll.grid(row=9, column=1, sticky="ns")\n'
        '\n'
        '    suggestion_text = tk.StringVar(value="Click Suggest label for an AI suggestion.")\n'
        '    ttk.Label(side, textvariable=suggestion_text, wraplength=260).grid(\n'
        '        row=10, column=0, sticky="w", pady=(6, 0)\n'
        '    )\n',
        "suggestion status",
    )
    source = replace_once(
        source,
        '    playing = False\n    after_id = None\n',
        '    playing = False\n    after_id = None\n'
        '    suggestions = {}  # segment index -> proposed text (not committed)\n'
        '    suggestion_queue = Queue()\n'
        '    pending_suggestion = None\n',
        "suggestion state",
    )
    functions = '''    def update_suggestion_ui():
        if pending_suggestion is not None:
            suggest_button.configure(text="Suggesting...", state="disabled")
            suggestion_text.set(
                "Analyzing this segment..." if pending_suggestion == current_index
                else "AI is analyzing another segment..."
            )
        elif current_index in suggestions:
            suggest_button.configure(text="Use suggestion", state="normal")
            suggestion_text.set(f"AI suggestion: {suggestions[current_index]}")
        else:
            suggest_button.configure(text="Suggest label", state="normal")
            suggestion_text.set("Click Suggest label to generate a suggestion.")

    def request_suggestion():
        nonlocal pending_suggestion
        index = current_index
        if index in suggestions:
            # The user explicitly chooses to put the suggestion in the entry;
            # Commit & Next remains the only action that stores a LABEL.
            label_entry.delete(0, "end")
            label_entry.insert(0, suggestions[index])
            save_current_draft()
            update_progress_and_tree()
            label_entry.focus_set()
            suggestion_text.set("Suggestion inserted. Review and click Commit & Next.")
            return
        if pending_suggestion is not None:
            return

        pending_suggestion = index
        update_suggestion_ui()
        segment = segment_at(index)

        def worker():
            try:
                label = suggest_activity(
                    video_path, segment["start_frame"], segment["end_frame"]
                )
                suggestion_queue.put((index, label, None))
            except Exception as exc:
                suggestion_queue.put((index, None, str(exc)))

        Thread(target=worker, daemon=True).start()

    def poll_suggestions():
        nonlocal pending_suggestion
        try:
            while True:
                index, label, error = suggestion_queue.get_nowait()
                pending_suggestion = None
                if label:
                    suggestions[index] = label
                update_suggestion_ui()
                if error and index == current_index:
                    suggestion_text.set(
                        "AI unavailable; manual labeling still works. " + error[:160]
                    )
        except Empty:
            pass
        root.after(150, poll_suggestions)

'''
    source = replace_once(
        source,
        '    def go_to_segment(index):\n',
        functions + '    def go_to_segment(index):\n',
        "worker, queue and suggestion handlers",
    )
    source = replace_once(
        source,
        '        show_segment_first_frame()\n'
        '        update_progress_and_tree()\n'
        '        label_entry.focus_set()\n',
        '        show_segment_first_frame()\n'
        '        update_progress_and_tree()\n'
        '        update_suggestion_ui()\n'
        '        label_entry.focus_set()\n',
        "segment navigation",
    )
    source = replace_once(
        source,
        '    play_button.configure(command=toggle_playback)\n',
        '    play_button.configure(command=toggle_playback)\n'
        '    suggest_button.configure(command=request_suggestion)\n',
        "button binding",
    )
    source = replace_once(
        source,
        '    go_to_segment(0)\n    root.mainloop()\n',
        '    go_to_segment(0)\n    root.after(150, poll_suggestions)\n    root.mainloop()\n',
        "suggestion queue polling",
    )

    # Validate the generated Python source BEFORE touching the original.
    compile(source, str(TARGET), "exec")
    if BACKUP.exists():
        raise FileExistsError(
            f"Backup already exists: {BACKUP}. Move it elsewhere before integration."
        )
    shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(source, encoding="utf-8")
    print("Integrated Label Suggester into:", TARGET)
    print("Backup saved as:", BACKUP)
    print("FINCH and existing LABEL handling are unchanged.")


if __name__ == "__main__":
    main()
