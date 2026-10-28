# UHPC AI — Ẩn giới hạn vật liệu khỏi giao diện

Bản cập nhật giao diện cho repository: `https://github.com/ledung1112k4-netizen/UHPC-AI-Mix-Design` (kiểm tra cấu trúc `main/app.py` ngày 28/10/2026).

## Mục đích

- Giữ các ô nhập **cấp phối vật liệu (kg/m³)** trong bài toán thuận, tuổi dự đoán, cường độ mục tiêu và chính sách **Có/Không/Tùy chọn** cho FA, GGBFS, silica fume, bột thạch anh.
- Bỏ tất cả ô chọn min/max vật liệu của bài toán thuận và ngược: xi măng, cát/cốt liệu, FA, GGBFS, SF, bột thạch anh, nước, phụ gia, sợi.
- Bỏ ô chọn giới hạn sợi thép và bảng min/max vùng huấn luyện khỏi giao diện.
- Giữ kiểm tra giới hạn **trong mã nguồn**. Trong bài toán thuận, khi cấp phối vượt ngưỡng, ứng dụng báo vật liệu vi phạm và không gọi mô hình. Trong bài toán ngược, các biên được truyền vào `DesignRules` và tối ưu.
- Không sửa mô hình `.pkl`, dữ liệu CSV hoặc `uhpc_core.py`.

## 3 bước cập nhật

1. Tải code GitHub mới nhất về máy, giải nén tại thư mục chứa `app.py`, `uhpc_core.py`.
2. Chép **`hide_material_limits.py`** và **`uhpc_limits.py`** vào **cùng thư mục** với `app.py`.
3. Mở terminal tại thư mục đó và chạy:

   ```bash
   python hide_material_limits.py
   python -m py_compile app.py uhpc_limits.py
   streamlit run app.py
   ```

Nếu thành công, script sao lưu trước khi sửa thành `app.py.backup_before_hidden_limits`. Tải **`app.py` đã sửa** và **`uhpc_limits.py`** lên GitHub để Streamlit triển khai lại.

> Trước khi chạy script, nếu đã áp dụng gói nâng cấp khác, hãy tải đầy đủ mã nguồn GitHub hiện tại; **đừng thay `app.py` bằng một bản cũ**. Script chỉ vá khi nhận diện đúng các đoạn mã đích và sẽ dừng mà không ghi khi không tương thích.

## Chỉnh mức giới hạn về sau (không hiện trên UI)

Trong tệp **`uhpc_limits.py`**:

- `INVERSE_MATERIAL_LIMITS`: biên tìm kiếm của bài toán ngược.
- `FORWARD_MATERIAL_LIMITS`: biên kiểm tra bài toán thuận.
- `MAX_STEEL_VOL_PERCENT`: giới hạn sợi thép tính theo phần trăm thể tích (mặc định 2%).
- `REQUIRED_COMPONENT_MIN`: mức tối thiểu cho FA/GGBFS/SF/bột thạch anh khi người dùng chọn **Bắt buộc**.

Mặc định xi măng **400–1000 kg/m³** trong cả hai bài toán. Silica fume và bột thạch anh **có thể bằng 0 kg/m³** khi chọn chế độ không sử dụng hoặc tùy chọn, nhưng khả năng dự đoán hợp lý còn tùy thuộc vào dữ liệu huấn luyện và cài đặt kiểm soát ngoại suy. Các giới hạn này **không phải tiêu chuẩn TCVN**.

## Kiểm tra

4 bài kiểm thử trên một bản mô phỏng các đoạn giao diện hiện có đã đạt. Script chưa được chạy trực tiếp trong môi trường Streamlit production với toàn bộ mô hình/dữ liệu trên GitHub. Chạy `python -m unittest discover -s tests -p 'test_*.py'` nếu bạn chép kèm thư mục tests. Nếu script báo thiếu mốc mã nguồn, giữ nguyên app và gửi lỗi để cập nhật theo phiên bản mới.
