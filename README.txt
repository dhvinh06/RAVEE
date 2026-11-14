RAVEE LABEL SUGGESTER v2 - VIDEO CLIPS
======================================

This is a DROP-IN replacement for the earlier 'label_suggester.py'.
It preserves the same suggest_activity(video_path, start_frame, end_frame)
function used by the GUI. You do NOT need to run integrate_label_suggester.py
again or modify your toolbox/display_video_snippets.py / main.py.

INSTALL (PowerShell, from RAVEE root, inside .venv):

1. Stop main.py / close the Tkinter GUI.
2. Make a backup of your previously working module:
   Copy-Item .\label_suggester.py .\label_suggester_v1.py.bak
3. Extract this ZIP and copy both .py files to the RAVEE root,
   replacing label_suggester.py. They must be next to 'toolbox' and 'gui'.
4. Keep your existing Gemini environment variables. For example:
   $env:GEMINI_MODEL = 'gemini-3.5-flash-lite'
   [bool]($env:GEMINI_API_KEY -or $env:GOOGLE_API_KEY)

TEST BEFORE GUI:
   python .\test_label_suggester_v2.py --start 88 --end 170

Expected console message: 'Label Suggester: video clip, ... FPS, ... MiB'
A label should appear after Gemini responds. If it prints 'still-frame
mode', MP4 encoding may have fallen back; inspect the warning above.

Then run the GUI normally:
   $env:PYTHONPATH = (Get-Location).Path
   python .\toolbox\main.py

Compare the original 6-frame method with the v2 clip method:
   python .\test_label_suggester_v2.py --start 88 --end 170 --mode frames
   python .\test_label_suggester_v2.py --start 88 --end 170 --mode video

WHAT CHANGED:
- An MP4 of each segment is encoded temporarily via OpenCV 'mp4v'.
- Frames are kept in chronological order with natural time progression.
- Clip is limited to a contiguous middle excerpt of up to 20 seconds.
- Encoded at ~6 FPS, max width/height 480 pixels; prompt asks Gemini to
  inspect the clip at 3 FPS instead of six isolated still frames.
- Prompt requests a precise visible VERB+OBJECT, not the overall task.
- AI output is an UNVERIFIED suggestion; the user must still confirm it.
- If MP4 encoding fails, old six-frame method is used with a warning.
- If Gemini API fails (including 503), manual labeling still works.
- Clip is cleaned up after reading; no new permanent video files.

LIMITATIONS:
- Gemini may still miss rapid/subtle motions and cannot infer actions
  that are invisible or too small in the video.
- Source audio is not preserved in the temporary clip.
- No recognition metrics yet. Validate against the actual video yourself.
- Each request sends video data to Google Gemini: pay attention to privacy,
  usage limits, and charges. ChatGPT subscription is unrelated.
- Requires OpenCV, NumPy, google-genai (already used by your project).
- Live API behavior cannot be tested here without your Gemini API key.

ROLLBACK:
   Copy-Item .\label_suggester_v1.py.bak .\label_suggester.py -Force
