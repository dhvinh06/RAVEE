# Semantic Event Aggregation

## Vị trí trong pipeline

FINCH → segment gốc → Label Suggester V3 + xác nhận nhãn thủ công → detailed log
→ **review nhóm task có ý nghĩa** → task-level log → Process Model từ task XES.

Không sửa thuật toán FINCH, segment/cluster gốc, Label Suggester V3, Gemini API
configuration hay cache preprocessing/IDT/FV/FINCH. Video đã preprocessing được
dùng trực tiếp. `main.py` gọi màn hình review sau khi tạo detailed dataframe, ở
cả demo và non-demo. Đường gọi Process Model tiếp tục dùng task-level XES.

`extractTaskLog(..., confirmed_groups=...)` xuất đúng nhóm đã xác nhận. Cách gọi
cũ không có `confirmed_groups` vẫn giữ task mapping, gộp nhãn sau mapping và lọc
posture như trước. Mapping cũ chỉ là gợi ý tên cho các singleton ở review mới;
không tự quyết định nhóm semantic. Nhánh review giữ mọi segment, kể cả posture,
để người dùng gộp vào mục tiêu phù hợp hoặc giữ riêng.

## Đề xuất Gemini

Màn hình ban đầu giữ từng segment riêng. **Chỉ khi bấm Suggest task groups
(Gemini) và đồng ý gửi dữ liệu** mới gọi API. Client chạy trong worker thread,
giữ cách lấy `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `.env`, `GEMINI_MODEL`, HTTP
timeout và video/frames mode hiện có của V3. Thiếu key/lỗi API/JSON sai không
làm mất các nhóm đang chỉnh; người dùng tiếp tục review thủ công.

Gemini nhận ID segment, nhãn hiện có đã được người dùng review, frame, timestamps,
duration và media tương ứng từng segment, cùng ngữ cảnh trước/sau. Nếu detailed
log có thêm `ai_label`, trường đó cũng được gửi; log cũ chỉ có nhãn `activity`
được xác nhận nên không thể khôi phục lại những AI suggestion chưa từng lưu.
Cluster IDs không được dùng làm tín hiệu gộp. Encoder clip/still của V3 được tái
sử dụng nguyên trạng. Clip dài có mô tả chính xác phần trích đoạn, không giả vờ
đã xem toàn bộ segment. Tổng media có giới hạn dung lượng inline.

Ngoài từng segment, Gemini được xem **TRANSITION**: khoảng một giây cuối segment
trước và một giây đầu segment sau, giới hạn trong đúng hai segment liền kề.
Đoạn này giúp theo dõi cùng vật thể xuyên qua ranh giới FINCH. Metadata ghi rõ
đây là phần phát lại chồng lấn với media segment, không phải thêm một lần thao
tác. Video/frames mode, fallback và giới hạn tổng media của V3 vẫn được dùng.

Đề xuất theo cửa sổ tối đa 12 segment. Hai nhóm nằm ở ranh giới các cửa sổ được
xem xét lại với nhau (tối đa 24 segment) để tránh cắt một task chỉ vì batching.
Chuỗi dài hơn giới hạn này giữ ranh giới để người dùng gộp thêm. Một lần đề xuất
có thể gồm nhiều API requests; không có request nào được chạy khi mở review.

Prompt yêu cầu chung **cùng vật thể vật lý + cùng mục tiêu cụ thể**, kể cả khi
có các lần thử lại, sửa sai, tinh chỉnh hoặc tạm dừng để kiểm tra rồi tiếp tục.
Việc một micro-action kết thúc không tự chứng minh task đã hoàn tất. Mô hình
phải xem diễn biến trước/sau inspection, nhận biết đổi vật thể, đổi mục tiêu hay
bắt đầu thao tác mới, và không suy đoán vật thể bị tay che khuất.

Phản hồi nội bộ Gemini nay có thêm `boundaries`: đúng một quyết định cho mỗi
cặp segment liên tiếp, gồm ID trái/phải, `object_continuity` và
`objective_continuity` (`same/different/unclear`), `operation_relation`
(`continuation/retry/correction/refinement/inspection_pause/new_operation/unclear`),
mức bất định, bằng chứng quan sát và tên task chung nếu đủ cơ sở.

Ứng dụng chỉ nối khi hai segment liền kề, cùng vật thể/mục tiêu được xác nhận,
quan hệ là tiếp tục/thử lại/sửa sai/tinh chỉnh/inspection, có bằng chứng và tên
task, đồng thời không bị đánh dấu bất định cao. Có thể sửa đề xuất ban đầu gồm
nhiều singleton khi các ranh giới có đủ bằng chứng liên tục. Đổi vật thể, đổi
mục tiêu, bắt đầu thao tác mới hoặc thiếu bằng chứng đều giữ tách. Nhóm bất định
cao và `Unclear activity` không được dùng làm cầu nối.

Không có rule theo tên hoạt động hoặc cluster ID để quyết định nối/tách, và
prompt không chứa danh sách hoạt động cố định. Các ví dụ ruy-băng/đồ trang trí
chỉ nằm trong regression tests. Một nhãn giống nhau không chứng minh cùng vật
thể hoặc mục tiêu; một nhãn khác nhau cũng không tự tạo ranh giới task.

Schema cuối cùng cho GUI, file review đã xác nhận và Event Log vẫn giữ nguyên.
Phản hồi API thiếu/sai quyết định ranh giới bị từ chối, không dùng làm đề xuất
hoàn tất. Không thêm API request; giới hạn output của request aggregation được
nâng lên 8.192 token để đủ chứa bằng chứng ranh giới. Model/key/timeout không đổi.

Kết quả mỗi nhóm gồm:

```json
{
  "start_frame": 0,
  "end_frame": 74,
  "source_segment_ids": [1, 2, 3, 4, 5],
  "suggested_task_label": "Attach decoration",
  "explanation": "The same decoration is repositioned and reinserted toward one attachment objective"
}
```

Validator bắt buộc phủ mọi segment đúng một lần, đúng thứ tự. Chỉ gộp frame và
timestamp liền kề, không đổi frame gốc, không bỏ sót/trùng/tạo ID mới, không gộp
qua videoID khác. Gemini là bộ đề xuất, không phải xác nhận cuối.

## Review Tkinter

Màn hình dùng Tkinter/ttk và preview OpenCV giống phong cách GUI hiện có; GUI
gán nhãn V3 không bị sửa. Các chức năng:

- Chọn nhóm để xem nhãn segment gốc, timestamps, lý do và phát đúng đoạn video.
- Chọn nhiều nhóm liền kề rồi **Merge selected adjacent groups**.
- **Split after source ID** tách tại ranh giới segment gốc; không cắt lại FINCH.
- Sửa task label/lý do, bấm **Apply label / explanation**.
- **Undo**, hoặc **Keep original segments** để quay về singleton.
- **Confirm task groups** xác nhận task log; không tự xác nhận đề xuất AI.

Detailed CSV/XES được checkpoint trước review. Hủy review không công bố nhóm
mới và không xuất process model mới. Các task log cũ, nếu có, vẫn là kết quả của
lần xác nhận trước, không phải một review vừa hủy.

Sau xác nhận, `<base>_semantic_groups.json` được ghi atomic, gồm nhóm, trạng
thái confirmed và fingerprint detailed records + video identity (path/size/mtime).
File thiếu/hỏng hoặc khác đầu vào không được nhận. Review cũ hợp lệ có thể nạp
lại làm draft; vẫn cần người dùng xác nhận. Đây là artifact review riêng, không
thay đổi manifest hoặc cơ chế cache pipeline.

## Export

- Detailed CSV/XES giữ nguyên từng segment, nhãn và FINCH cluster ban đầu.
- Task CSV/XES dùng tên task đã xác nhận, timestamp đầu/cuối từ segment nguồn,
  tổng duration, `source_segment_ids`, `source_activities`, `task_group_id` và
  `group_explanation`.
- Các nhóm đã được reviewer tách không tự gộp lại chỉ vì có cùng tên.
- Process Model tiếp tục dùng **task XES** và xuất PNG/SVG như phần sửa trước.

## Chạy riêng bằng dữ liệu đã cache

Chạy tại thư mục RAVEE. Chọn đúng AVI preprocessing tương ứng với detailed CSV
và videoID. Không dùng `main.py` nếu mục đích chỉ là review lại task từ log đã có.

Ví dụ tìm video gói quà hiện có (nếu có nhiều phiên bản, chọn đúng bản đã dùng
để tạo detailed log):

```powershell
$video = Get-ChildItem -Path ".\temp\pipeline_cache\preprocessing\*\1_3197334-hd_1920_1080_25fps_reduced.avi" | Select-Object -First 1
& ".\.venv\Scripts\python.exe" -B ".\toolbox\semantic_aggregation.py" `
  --video $video.FullName `
  --detailed-csv ".\EventLog_Output\all_videos_detailed_eventlog.csv" `
  --case-id 1 `
  --output-dir ".\EventLog_Output\semantic_review" `
  --base-name gift_review --offline
```

Lệnh này chỉ mở review dữ liệu sẵn có; Gemini bị vô hiệu hóa. Không gọi Docker,
IDT, FV, FINCH hay chạy lại GUI gán nhãn segment. Output ở thư mục riêng, không
ghi đè CSV đầu vào. CSV nhiều video cần chọn `--case-id` và xử lý từng case.

Sau khi xác nhận sẽ có `gift_review_detailed_eventlog.csv/.xes`,
`gift_review_task_eventlog.csv/.xes` và `gift_review_semantic_groups.json`.
Thêm `--render-model` nếu muốn render sơ đồ thật từ task XES đã xác nhận.

Để thử một bộ đề xuất offline, truyền `--proposals-json proposals.json` (JSON
list nhóm hoặc object có `groups` theo schema ở trên). Để chủ động dùng Gemini,
bỏ `--offline`, rồi bấm nút đề xuất trong review; không có API call tự động.
Những lệnh mở GUI, render thật và gọi API chưa được chạy trong lần triển khai.

Để đánh giá lại các segment 9–11 trên cache, có thể dùng `--base-name gift_refined`
để không nạp draft cũ của `gift_review`. Mở log/video tương ứng như trên, bỏ
`--offline` khi bạn chủ động muốn gọi Gemini, rồi bấm **Suggest task groups**.
Xem đề xuất cùng preview và kiểm tra lý do có dựa vào cùng vật thể/mục tiêu;
Merge/Split/Undo/rename/Confirm vẫn hoạt động như trước. Kiểm thử offline không
thể xác nhận chất lượng suy luận thị giác của Gemini thật.

## Kiểm thử

```powershell
& ".\.venv\Scripts\python.exe" -B -m unittest test_semantic_aggregation test_pipeline_cache -v
```

Suite offline/mock có các ca decoration retries, cutting versus tying, unclear
actions, nhãn trùng nhưng task khác, boundary batching, merge/split/rename/undo,
phủ đủ nguồn và timestamp, lỗi API/JSON, hủy review, checkpoint/atomic review,
export CSV/XES hai mức và đường gọi Process Model tới task XES. Bộ cache cũ
chạy bằng mock; không khởi chạy extractor hoặc FINCH thật. Chưa đánh giá chất
lượng đề xuất bằng Gemini thật hoặc thao tác cửa sổ GUI thật.

Regression mới dùng trường hợp 9–11: cắt, quan sát rồi cắt chỉnh **cùng** sợi
ruy-băng. Mock cố ý trả ba task riêng nhưng bằng chứng ranh giới cho thấy cùng
mục tiêu; kết quả là `[9, 10, 11]`, `trim ribbon`, với giải thích
`The person repeatedly trims the same ribbon to adjust its length.` Có thêm
ca đổi sợi ruy-băng/mục đích, đổi thao tác trên cùng vật, bị che khuất và một
quy trình xử lý bề mặt kim loại để kiểm tra thuật toán không phụ thuộc ví dụ.

Chạy riêng ca 9–11, không mở video hoặc gọi API:

```powershell
& ".\.venv\Scripts\python.exe" -B -m unittest test_semantic_aggregation.GeminiProposalTests.test_mock_gemini_split_trimming_proposals_are_corrected_by_boundary_evidence -v
```
