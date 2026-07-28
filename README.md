# DataLabeling Python

Công cụ gán nhãn ảnh (box, box xoay, polygon) kèm train YOLO tích hợp.
Bản Python của app C# WPF ImageLableing.

## Cài đặt

    pip install -r requirements.txt
    python main.py

## Quy trình

1. **Mở thư mục ảnh** — nhãn cũ trong `labels/*.json` tự load lại.
2. **Gán nhãn**: chọn class, kéo chuột vẽ box; kéo nút tròn xanh để xoay box;
   chế độ Polygon: click từng đỉnh, click gần điểm đầu để đóng.
3. **Auto Label** (tùy chọn): load model `.pt`/`.onnx`, chạy batch hoặc preview
   duyệt từng ảnh; hỗ trợ tách frame video.
4. **Export**: chọn task detect / obb / segment, chia train/val, sinh `data.yaml`.
5. **Train**: chọn model nền + epochs, theo dõi log trực tiếp; kết quả trong `runs/`.

## Phím tắt

| Phím | Chức năng |
|---|---|
| `A` / `←` | Ảnh trước |
| `D` / `→` | Ảnh sau |
| `Ctrl+S` | Lưu nhãn ảnh hiện tại |
| `Ctrl+C` / `Ctrl+V` | Copy / paste ROI |
| `Delete` | Xóa shape đang chọn |
| `1`–`9` | Chọn nhanh class |
| `0` | Fit ảnh |
| `Esc` | Hủy polygon đang vẽ |
| Lăn chuột | Zoom tại con trỏ |
| Kéo chuột giữa | Pan |

## Định dạng nhãn

`labels/<tên ảnh>.json` — giữ nguyên cấu trúc của bản C#
(`BoundingBoxes[{Id, ClassName, X, Y, Width, Height, Angle}]`,
`Polygons[{Id, ClassName, Points}]`, tọa độ pixel, góc độ).

## Test

    pytest -v
