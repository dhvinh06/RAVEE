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

The review starts from singleton segments. Gemini is called only after an
explicit **Suggest task groups** action and confirmation. Requests run on a worker;
Tk widgets are updated through the UI queue. `GEMINI_LABEL_MODEL` selects the
individual/batch label model, and `GEMINI_AGGREGATION_MODEL` selects the task model.
Each falls back to `GEMINI_MODEL` and then the existing code default. Missing keys,
API failures or invalid JSON preserve the current human review.

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

Đề xuất theo cửa sổ tối đa 12 segment, cửa sổ kế tiếp chồng một segment. Mỗi
ranh giới chỉ cần đánh giá một lần, không còn request phụ để nối hai batch.
Khi toàn bộ boundary cache còn trống và dịch vụ không lỗi, số request là
`ceil((N - 1) / 11)` với N segment; N=1 không cần API. Cache hợp lệ được bỏ qua
trước khi encode media hoặc tạo client. Không có request khi chỉ mở review.

## Intermediate objectives and human review

An activity segment records an atomic action. A task groups adjacent phases
serving the same observable intermediate result, such as preparation, execution,
retry, correction, inspection and refinement. A shared object or broad process
goal alone is insufficient to merge distinct intermediate results. Boundary
responses report `objective_before`, `objective_after`, `objective_transition`,
`objective_scope` and `transition_evidence`. Uncertain/contradictory evidence stays
available for human review. Merge, split, rename, Undo and explicit confirmation
remain authoritative; neither model suggestions nor matching verbs auto-confirm
an event log. The reviewed segment labels and FINCH bounds remain unchanged.

## Độ tin cậy API và tiếp tục sau lỗi

Aggregation uses `gemini_runtime.py` for model selection, credentials, client
creation and retry classification. `GEMINI_ACTIVE_KEY` selects a configured
numeric/named credential; without it, the legacy GOOGLE_API_KEY/GEMINI_API_KEY
precedence applies. A GUI credential selection applies to subsequent AI stages
in the application session without rewriting `.env`. No model/key fallback is
automatic. Aggregation requests remain sequential under the shared lock, with
progress such as `Checking boundary 4/10` and reused/unresolved counts.

| Failure | Actual behavior |
| --- | --- |
| HTTP 500/502/503/504 | Temporary service failure; at most four attempts |
| HTTP 429 with RPM/TPM evidence | Temporary rate limit; same bounded retry policy |
| Confirmed requests-per-day exhaustion | No automatic retry/rotation; show the existing nonmodal recovery dialog |
| Ambiguous HTTP 429 | Stop; do not claim daily exhaustion or assume a temporary rate limit |
| HTTP 401/403 | Stop; report authentication/permission failure |
| Invalid JSON/schema/evidence | Stop; do not fabricate or automatically retry a response |

Transient retries use full jitter with 1/2/4-second windows. `Retry-After`
(seconds or HTTP date) is respected, with at most 60 seconds total backoff and
60 seconds timeout per attempt. SDK retries are disabled. Daily quota evidence
remains nonretryable even if the response also contains retry guidance.

For confirmed daily exhaustion, **Continue manually** disables further AI
requests for the review, **Select credential** explicitly resumes the pending
request using another configured authorized credential, and **Cancel** stops
that request without discarding review. Keys in one project share its limits.
Clients are recreated on an explicit switch; completed cached boundaries and
human edits are preserved. Recovered task proposals require an explicit load,
review and confirmation; they never silently replace edits made while paused.
See [shared configuration and recovery](../docs/GEMINI_CONFIGURATION_AND_TASK_OBJECTIVES.md).

Mỗi boundary thành công được ghi atomic riêng tại `temp/semantic_boundary_cache`,
có completion marker và checksum. Khóa cache gồm đường dẫn + SHA-256 nội dung
video, các frame/ID/nhãn/timestamps đầu vào, model, prompt, generation settings,
batch size, media mode/context và phiên bản mã aggregation/encoder. Cache hỏng,
đang ghi dở hoặc khác đầu vào không được sử dụng. Cả quyết định nối và quyết
định giữ tách có bằng chứng hợp lệ đều được cache; lỗi API không được cache.
Cache này tách biệt cache preprocessing/IDT/FV/FINCH và file xác nhận thủ công.

Sau lỗi dai dẳng, circuit breaker dừng các batch chưa chạy để tránh gọi dồn một
dịch vụ đang unavailable. Boundary đã hoàn tất (kể cả từ lần trước) vẫn được dùng;
boundary chưa giải quyết giữ riêng để review. GUI không tự Confirm. Nếu không
boundary nào giải quyết được, các nhóm hiện tại của reviewer được giữ nguyên.
Bấm Suggest lần sau chỉ gọi các boundary còn thiếu; Merge/Split/Undo/rename/
Confirm vẫn hoạt động như trước.

### Refresh AI suggestions

Nút **Refresh AI suggestions** bỏ qua cache boundary analysis của video/case
đang review và yêu cầu đánh giá lại mọi ranh giới liền kề bằng prompt/model đang
cấu hình. Nó không xóa cache cũ và không thay đổi preprocessing, IDT, FV, FINCH,
Label Suggester hay các nhóm đã xác nhận. Boundary không liền kề vẫn giữ tách
theo validator, không cần API. Các kết quả fresh hợp lệ được ghi atomic thay
thế đúng entry; request Suggest bình thường sau đó có thể tái sử dụng chúng.

CLI có `--refresh-suggestions`, ví dụ thêm vào lệnh review cached video ở dưới.
Flag này chỉ bật refresh cho **lần yêu cầu AI đầu tiên được người dùng xác nhận**,
không gọi Gemini khi mở GUI. Những lần Suggest tiếp theo dùng cache bình thường;
nút Refresh luôn buộc đánh giá mới. `--offline` vẫn khóa cả hai nút và không gọi
API, kể cả khi có flag refresh.

Nếu 503 hoặc lỗi khác khiến refresh dừng, giữ nguyên entry cũ chưa thay được.
Chỉ cache khớp video/case/đầu vào và prompt/model/configuration hiện tại mới được
dùng làm fallback; không dùng cache từ model/prompt khác như kết quả mới. Case ID
được đưa vào khóa cache để các case dùng cùng video không ảnh hưởng lẫn nhau.

GUI hiển thị số **fresh**, **reused**, **OLD fallbacks**, **unresolved**, cùng số
API calls và lỗi (ví dụ HTTP 503). Chọn một nhóm sẽ thấy nguồn đánh giá ở từng
ranh giới liên quan: `fresh`, `reused`, `OLD cached fallback — NOT fresh` hoặc
`unresolved`. Kết quả cũ không bao giờ được đếm là fresh. Boundary không có cache
hợp lệ và chưa đánh giá xong giữ riêng để review thủ công. Không tự Confirm.

Caller Python có thể dùng `propose_task_groups(..., refresh_suggestions=True,
stats=report, progress=callback)`; `report['boundary_sources']` chứa provenance
theo cặp ID. Caller không cung cấp cả stats lẫn progress sẽ nhận warning nếu có
fallback cũ. Schema task groups và Event Log vẫn giữ nguyên.

Số request được kiểm tra bằng mock, không có lỗi HTTP: 11 segment **1 → 1** lần
đầu, **0** khi cache đủ; 25 segment tối đa **5 → 3**, rồi **0** khi cache đủ.
Với batch size=3 và 5 segment: **3 → 2**, rồi **0**. Ca lỗi mock: một batch thành
công + một batch lỗi 503 đủ bốn attempts = **5 calls**; lần tiếp tục chỉ cần
**1 call** cho hai boundary còn thiếu, lần sau nữa **0**. Retry attempts được
tính vào số API calls báo ở GUI; đây không phải số đo từ Gemini thật.

## Task-objective discovery

Aggregation distinguishes atomic movements from tasks that produce one concrete
observable result. Preparation, execution, retries, corrections, inspection and
refinement can form one task even when action labels or techniques differ. Task
completion/departure, a changed objective or an independent target starts a new
task; visual similarity, shared tools and identical labels never suffice to merge.
There is no fixed activity vocabulary or predetermined number of tasks.

Each boundary retains the existing evidence fields and adds `task_boundary`
(`continuing/new_task/unclear`) and a consistent `task_objective` phrase. Relations
also include `preparation`, `execution` and `completion_check`. Explicit continuing
task evidence can join a changed atomic `new_operation`; legacy decisions without
these fields retain the old conservative rules. `object_continuity=related` allows
visibly linked components of one target task, only with explicit task evidence.
Different/unclear targets or objectives, insufficient evidence, high uncertainty,
nonadjacent segments and explicit new-task boundaries still prevent merging.

One explicit objective must remain consistent throughout a merged run, preventing
pairwise decisions from drifting into an unrelated objective. Initial singleton
proposals can be reconciled using boundary evidence. An `Unclear activity` label
can participate only when explicit low-uncertainty video evidence establishes the
ongoing task across it; the label alone cannot determine the grouping. Group-level
high uncertainty or an explicit rejection of a shared objective still veto merging.

Each request includes up to two contiguous neighboring segments on either side as
context-only metadata and media, including transitions. Context IDs cannot appear
in the target partition. Long clips retain V3's duration limit; two endpoint stills
add preparation/completion state evidence when the clip is a middle excerpt. The
existing video/frames settings, inline byte budget, batching and request counts are
preserved. These bounded observations do not guarantee that occluded or unseen
steps can be resolved; uncertain boundaries stay available for human review.

Review, source segments, exports and cache storage formats remain unchanged.
Prompt/implementation fingerprints naturally create new semantic analysis entries
on the next explicit AI request; previous entries are retained, and incompatible
prompt results are not presented as fresh or reused. Pipeline caches are untouched.

Offline regressions cover multi-phase objectives with different action labels,
related assembly components, completion followed by another task on the same
object, similar actions on different objectives/targets, objective drift,
context-only IDs, long-clip endpoints, cache-only resume and old cache preservation.
The gift-wrapping sequence is a test fixture only, not an algorithm rule. No real
Gemini request or video pipeline run is needed for these tests.

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
& ".\.venv\Scripts\python.exe" -B -m unittest tests.test_semantic_aggregation tests.test_pipeline_cache -v
```

Kiểm thử riêng retry, Retry-After, partial failure, circuit breaker và cache:

```powershell
& ".\.venv\Scripts\python.exe" -B -m unittest tests.test_semantic_reliability -v
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
& ".\.venv\Scripts\python.exe" -B -m unittest tests.test_semantic_aggregation.GeminiProposalTests.test_mock_gemini_split_trimming_proposals_are_corrected_by_boundary_evidence -v
```


## Resizable review preview

Semantic review now uses a horizontal `PanedWindow`. Its initial split gives
roughly 60% of the working width to the video; drag the sash to adjust it. The
larger initial window targets about 350-450 px of displayed video height where
the desktop permits it, while shrinking to fit smaller screens. Playback controls
sit immediately below the image, followed by the task label and a wrapped,
editable grouping explanation. Source segment details have moved below the list.
Merge, Split, Apply edits (including rename), Undo, Reset, Suggest/Refresh,
Confirm and Cancel retain their existing callbacks and human confirmation rules.

The renderer uses the largest aspect-preserving rectangle inside the actual
preview widget, including upscaling low-resolution frames. It no longer relies
on `PIL.Image.thumbnail()`, which cannot enlarge an image. The full frame is
resized without cropping; remaining letterboxing follows from the aspect ratios.
Window and sash changes repaint the current cached frame with a coalesced 40 ms
callback. Resizing never seeks or reads video and does not restart playback.
Resize and playback timers are canceled separately on close. The shared theme is
unchanged. Native window geometry and playback still need desktop visual QA;
synthetic tests cover sizing and the real callback workflow without a Tk display.
