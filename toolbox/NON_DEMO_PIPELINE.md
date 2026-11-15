# Tiếp tục non-demo pipeline

## Dữ liệu IDT đang có

`IDT/run_idt.py` có hai loại đầu ra khác nhau:

- `hog_features.csv`, `hof_features.csv`, `mbh_features.csv`: cột `name` xác định
  video; các cột số chứa **một Fisher Vector tổng hợp cho toàn bộ video**.
  Cột đầu không tên là index do pandas ghi. Tên CSV là loại descriptor, không
  phải case ID. Không thể khôi phục frame indices/trajectories từ các vector này.
- `<case>_<video>_reduced-trajectory.npy`: structured array chứa `frame_num`,
  `trajectory`, HOG, HOF và MBH. Đây là đầu vào của
  `extract_framewise_features` và temporal FINCH.

Khi kiểm tra dữ liệu hiện tại, HOG/HOF chỉ chứa `person01_boxing_d1_uncomp`;
MBH chứa cả mẫu boxing và `1_3197334-hd_1920_1080_25fps_reduced` (24.832 giá trị
cho mỗi video). Không có raw trajectory `.npy`. MBH của video mới được giữ
nguyên nhưng chưa đủ để phân đoạn video theo frame. Pipeline báo lỗi có hướng
dẫn thay vì bỏ qua descriptor hoặc coi một video-level vector là chuỗi frame.

Video mới có 2.153 frame đầu vào; bản preprocessing hiện tại có 1.539 frame,
640×480, 15 fps, phù hợp với thuật toán reducer hiện có. Các mốc thời gian và
logic reducer/Event Log hiện có được giữ nguyên.

Reader raw lấy descriptor theo tên trường trong structured array và chuyển
`frame_num` một phần tử thành scalar rõ ràng, tương thích NumPy hiện tại.
Thuật toán PCA/GMM/FV và FINCH được giữ nguyên.

## Cache và tính đầy đủ

Cache nằm trong `temp/pipeline_cache`, chia theo preprocessing, IDT,
Fisher Vectors và FINCH. Không xóa cache hoặc dữ liệu nguồn sau khi xuất log.

Mỗi cache có fingerprint của đầu vào/nội dung và cấu hình liên quan, hash mã xử
lý và phiên bản thư viện. Ví dụ, thay `required_cluster` chỉ làm mất cache FINCH;
thay mã Fisher Vector giữ lại preprocessing/IDT. Không dựa riêng vào tên file,
thời gian sửa hay việc một file đã tồn tại.

Chỉ ghi `complete.json` sau khi producer kết thúc, đầu ra được kiểm tra và file
đã được xuất bản bằng atomic replace. Mỗi lần đọc kiểm tra SHA-256 artifact.
Video phải decode đủ frame và đúng thông số reducer; raw trajectories phải có
đúng schema, dữ liệu hữu hạn và frame hợp lệ; số hàng FV phải khớp tập frame mà
trajectories bao phủ; FINCH phải khớp số hàng/số cụm. Kết quả lỗi hoặc công việc
đang dở không được coi là cache hoàn tất.

Các file cũ chưa có manifest **không tự động được coi là cache hợp lệ**.
Hai tùy chọn `--adopt-existing-preprocessing` và `--adopt-existing-idt` là lựa
chọn di chuyển dữ liệu cũ: chỉ dùng khi bạn biết file tương ứng được tạo từ đúng
video/cấu hình hiện tại. Pipeline kiểm tra tên khớp chính xác, thời gian không
cũ hơn nguồn, cấu trúc/tính đầy đủ, rồi sao chép vào cache và ghi manifest gắn
với hash nguồn hiện tại. Những kiểm tra này không thể chứng minh lịch sử nguồn
và cấu hình của một file cũ vốn không có manifest. Không nhận CSV toàn video
thay cho raw trajectories; không nhận lại các file FV/FINCH cũ thiếu provenance.

## Chạy từ PowerShell tại thư mục RAVEE

Thiết lập import path cho cách chạy script hiện có của project (đặc biệt khi
không dùng cấu hình PyCharm):

```powershell
$env:PYTHONPATH = (Get-Location).Path + [IO.Path]::PathSeparator + $env:PYTHONPATH
```

Nếu đã có cache hoàn tất do pipeline mới tạo, tiếp tục bằng:

```powershell
& ".\.venv\Scripts\python.exe" ".\toolbox\main.py"
```

Lệnh này tái sử dụng các stage hợp lệ rồi tiếp tục GUI gán nhãn và xuất CSV/XES.
Khi thiếu preprocessing/raw IDT, mặc định dừng, không tự khởi chạy Docker.

Nếu có raw `.npy` hoàn tất từ lần chạy cũ và biết rõ nguồn/cấu hình của nó:

```powershell
& ".\.venv\Scripts\python.exe" ".\toolbox\main.py" --adopt-existing-preprocessing --adopt-existing-idt
```

**Với dữ liệu hiện tại chỉ có global MBH CSV**, cần tạo raw trajectories một
lần cho video mới. Sau khi chủ động quyết định chạy bước nặng, dùng:

```powershell
& ".\.venv\Scripts\python.exe" ".\toolbox\main.py" --adopt-existing-preprocessing --allow-idt
```

Lệnh cuối chưa được chạy khi sửa lỗi. Nó tái sử dụng bản preprocessing cũ sau
khi kiểm tra, chạy IDT raw cho những video đầu vào còn thiếu, rồi tính FV/FINCH
và mở GUI. CSV MBH tổng hợp không thể thay thế lần trích xuất raw còn thiếu.
Nếu bị ngắt sau một stage, lần chạy tiếp theo tái sử dụng các stage đã hoàn tất.

## Docker và video mẫu

Docker dùng image có tag/label theo fingerprint nguồn build. Chỉ kiểm tra/build
image khi cần extraction; image có label đúng được tái sử dụng. Thay video hoặc
CSV không làm rebuild. `--rebuild-idt` buộc build lại ở lần extraction cần thiết;
nó không vô hiệu hóa raw IDT cache đang hợp lệ.

`IDT/.dockerignore` loại `data`, `features` và các file sinh ra khỏi image.
Mỗi container mount một thư mục riêng chỉ chứa video đầu vào đang xử lý,
read-only, và một thư mục đầu ra riêng. `RAVEE_RAW_ONLY=1` yêu cầu lưu raw
trajectories; cách chạy standalone không đặt biến này vẫn giữ các tùy chọn
HOG/HOF/MBH của `run_idt.py` (phải mount `data` và `features` khi chạy container).

Nguồn boxing nằm sẵn trong `IDT/data` và trước đây bị `COPY . .` đưa vào image;
extractor cũ duyệt mọi tệp trong thư mục đó. Pipeline mới không truyền video mẫu
này trừ khi người dùng chủ động đặt nó trong `video_input`. Không xóa file mẫu,
CSV hay video gốc.

## Kiểm thử nhẹ

```powershell
& ".\.venv\Scripts\python.exe" -m unittest tests.test_pipeline_cache -v
```

Kiểm thử dùng dữ liệu CSV hiện có và mock cho Docker, preprocessing, FV,
FINCH, GUI và Event Log. Không chạy extraction hay toàn bộ pipeline thật.
