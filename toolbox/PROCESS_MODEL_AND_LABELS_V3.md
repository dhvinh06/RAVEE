# Process Model và Label Suggester V3

## Xuất sơ đồ quy trình

`showProcessModel()` cũ gọi `pm4py.view_bpmn()`. Trong PM4Py đang cài, nhánh
viewer có thể gọi `matplotlib.pyplot.show()` (chờ đóng cửa sổ) hoặc mở trình xem
ảnh của hệ điều hành. Đây là phụ thuộc GUI không cần thiết ở cuối pipeline và
có thể gây cảm giác treo/chờ trên Windows. Không tái hiện cửa sổ treo trực tiếp
trong lần sửa này vì chỉ kiểm thử offline/mock.

Hàm mới giữ cách đọc XES, discover process tree và chuyển BPMN, nhưng chỉ xuất:

- `<tên XES>_process_model.png`
- `<tên XES>_process_model.svg`

Mặc định ảnh nằm cạnh XES. Không mở cửa sổ, trình duyệt hay Matplotlib.
Mỗi lệnh Graphviz `dot` có timeout 30 giây, chạy ẩn cửa sổ console trên Windows.
Việc tạo file vẫn chạy đồng bộ nhưng không chờ người dùng đóng viewer. Có thể
truyền `timeout=` hoặc `output_dir=` khi gọi riêng hàm.

Ghi vào thư mục tạm trước rồi thay thế file đích sau khi render thành công.
Nếu PNG lỗi, vẫn thử SVG. Lỗi đọc log, discover, thiếu `dot` trên PATH, timeout,
quyền ghi hoặc Graphviz trả lỗi đều được báo rõ trên stderr; hàm trả các đường
dẫn thực sự lưu được, không làm hỏng CSV/XES và không dừng phần gọi hàm.

Có thể tạo lại sơ đồ chỉ từ XES đã xuất, không chạy video/pipeline. Chạy từ
PowerShell tại thư mục RAVEE, sau khi chủ động muốn render thật:

```powershell
& ".\.venv\Scripts\python.exe" -B -c "import sys; sys.path.insert(0, 'toolbox'); from event_log_creation import showProcessModel; print(showProcessModel('EventLog_Output/all_videos_task_eventlog.xes'))"
```

Lệnh trên chưa được chạy trong lần sửa này; `dot` được mock trong kiểm thử.

## Label Suggester V3

Chữ ký `suggest_activity(video_path, start_frame, end_frame, sample_count=6)` và
kiểu trả về chuỗi giữ nguyên, không đổi GUI hoặc dictionary `LABEL`.

Mỗi yêu cầu có các phần riêng, kèm tên vai trò và frame/timestamp nguồn:

1. BEFORE CONTEXT: tối đa 3 giây trước segment.
2. TARGET SEGMENT: đúng interval cần gán nhãn, không nhập nhằng với context.
3. AFTER CONTEXT: tối đa 3 giây sau segment.

Context được cắt ở đầu/cuối video. Đổi bằng `RAVEE_SUGGEST_CONTEXT_SECONDS`
trong khoảng 0–10; mặc định là 3. Giữ video mode và fallback still frames.
Ở frames mode, target lấy tối đa `sample_count` ảnh, mỗi phần context tối đa 3
ảnh. Cả yêu cầu có giới hạn tổng dung lượng inline. Segment dài vẫn dùng trích
đoạn giữa tối đa 20 giây như V2; prompt nêu rõ phần bị lược và yêu cầu không suy
diễn từ những frame chưa xem.

Gemini chỉ gán nhãn cho target. Context dùng để nhận biết trạng thái trước/sau,
không được lấy hành động của segment khác làm nhãn. Prompt phân biệt:

- `Tie ribbon bow`: thấy tạo vòng/nút nơ mới trong target.
- `Tighten ribbon bow`: kéo siết một chiếc nơ đã hình thành.
- `Adjust ribbon bow`: chỉnh vị trí vòng/đuôi nơ đã có.

Đây là quy tắc phân biệt khi thấy hành động với nơ, không phải danh sách hoạt
động cố định. Nhãn cho các hành động khác vẫn dùng từ vựng mở, 2–6 từ tiếng Anh.
Cách viết được chuẩn hóa nhẹ và prompt yêu cầu dùng cùng wording cho hành động
lặp lại; không ép các segment liên tiếp phải có nhãn khác nhau.

Phản hồi nội bộ là JSON gồm nhãn, bằng chứng trong target, mức bất định,
`target_visible` và phân loại thao tác nơ. Ứng dụng trả `Unclear activity` khi
bằng chứng trống, target không đủ rõ, uncertainty cao hoặc không phân biệt được
thao tác nơ. Prompt cấm suy đoán thao tác bị tay che chỉ từ trạng thái trước/sau.
Đây vẫn là gợi ý của mô hình, không phải bảo đảm nhận diện đúng; người dùng vẫn
review bằng **Use suggestion**, rồi xác nhận bằng **Commit & Next**.

JSON sai, API lỗi hoặc timeout tiếp tục đi qua cơ chế lỗi hiện có của worker
GUI; người dùng vẫn gán nhãn thủ công. HTTP timeout là 60 giây mỗi request và
SDK client được đóng sau khi dùng. Không đổi model đang cấu hình hoặc khóa API.

## Kiểm thử offline trên dữ liệu cache

```powershell
& ".\.venv\Scripts\python.exe" -B -m unittest tests.test_process_model_and_labels_v3 -v
```

Suite kiểm tra PNG/SVG bằng Graphviz mock, lỗi/timeout, giữ nguyên log; đọc XES
cache hiện có để kiểm tra đường gọi mà không discover/render thật; chỉ đọc
metadata của video preprocessing cache để kiểm tra ranh giới context; mock toàn
bộ Gemini/encoding khi kiểm tra request. Không upload dữ liệu, không gọi API,
không chạy Docker, IDT, GMM/PCA, FINCH hoặc toàn bộ video. Các test dùng cache
cá nhân sẽ skip nếu máy khác không có những file đó.

Nếu sau này muốn đánh giá **một segment** với Gemini thật, script
`test_label_suggester_v2.py` hiện có vẫn gọi cùng API hàm và sẽ dùng V3. Truyền
`--video` là đường dẫn `.avi` trong `temp/pipeline_cache/preprocessing/<hash>/`,
`--start`/`--end` là frame của segment cần xem, và `--mode video` hoặc `frames`.
Đây là một yêu cầu API thật và có thể tính phí; chưa chạy trong lần sửa này.
Không cần chạy lại pipeline để thử một segment.


## Segment-action scope and final human edits

V3 names the specific visible operation in the target, rather than the broader
process, scene or finished result. The prompt asks for the narrowest supported
verb + object and target evidence supporting that exact action. Before/target/
after media remain separate and chronological; a final text reminder re-anchors
the answer to the original target and its actual supplied excerpt. No evaluation
reference groups or activity keyword list are embedded in the prompt.

For action labeling only, each context side is bounded by both the configured
maximum and the target duration. For example, a one-second target gets at most
one second of context per side. Task aggregation retains its wider context via
`_context_windows`; clip extraction, long-target excerpt disclosure and fallback
frames are otherwise unchanged. The five JSON fields, bow compatibility metadata,
model selection, token limit and `suggest_activity()` API remain unchanged.

A structural parser cannot prove that a confident label is semantically correct.
Model accuracy, including context dominance on a particular video, still needs
live evaluation. Human review remains required. V3 suggestions are session-local
in `gui_elements.py`, separate from committed `LABEL` values; no production V3
label-response disk cache was found.

Finish now checks both missing labels and saved-label/draft differences. Revisiting
an earlier segment, editing its text, then finishing another segment redirects to
the uncommitted edit instead of silently exporting its old committed text. Each
edit still needs explicit Commit. `main.py` snapshots the committed dictionary;
`extractLog()` writes these strings as `activity`; aggregation consumes that
reviewed field. Suggestions do not overwrite it asynchronously.

Aggregation boundary keys and confirmed-review identities already include reviewed
activities, segment bounds/times and video identity. Label changes cannot reuse
incompatible boundaries, even as refresh fallbacks. No cache schema change was
needed, and label edits themselves never request Gemini or delete cache files.
The existing boundary key also fingerprints all encoder-module source: this V3
code update changes that fingerprint, conservatively leaving older boundary
entries unmatched. They remain on disk. Only an explicit Suggest/Refresh request
can generate new entries; a warm cache with unchanged current inputs is reused.

Focused offline checks:

```powershell
python -B -m unittest tests.test_review_preview_and_label_accuracy -v
```
