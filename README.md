# Sửa triệt để giới hạn và lọc cấp phối UHPC AI — 28/10/2026

## Vì sao app vẫn còn bảng min/max?
Tệp `hide_material_limits.py` được phát hành trước đây là **bản vá chạy tại máy**, không tự cập nhật GitHub. Nếu không chạy vá rồi commit/push `app.py` + `uhpc_limits.py`, Streamlit vẫn hiển thị phiên bản cũ. Nếu Streamlit dùng nhánh khác nhánh vừa cập nhật, giao diện cũng vẫn không đổi.

## Bản này sửa gì?

1. Ẩn các ô min/max **của từng vật liệu** ở hai tab thuận/ngược; giữ giao diện nhập cấp phối, lựa chọn vật liệu và mục tiêu MPa.
2. Giữ giới hạn vật liệu trong mã `uhpc_engineering.py`, **không hiện widget**.
3. Chặn nghiệm xi măng thấp ngay trong bounds tối ưu DE, không chỉ cảnh báo sau dự đoán.
4. Tăng bộ điều kiện cho cấp phối theo mục tiêu: tổng chất kết dính, W/B vật lý (không tính bột thạch anh), tỷ lệ FA/GGBFS/SF. Nếu không có nghiệm, báo không đạt.
5. Kiểm tra cùng những điều kiện ấy khi chọn phương án hợp lệ và trước khi công bố kết quả; ẩn bảng cấp phối nếu chưa đủ điều kiện.
6. Bài toán thuận từ chối cấp phối ngoài bộ sàng lọc nghiên cứu (ví dụ 60 kg xi măng/m³).
7. **Không đụng mô hình `.pkl`, dữ liệu `.csv`, công thức đặc trưng 15 biến hoặc cách notebook tính W/B có tính bột thạch anh.**

## Ngưỡng nội bộ — thiết kế nghiên cứu thận trọng, KHÔNG phải TCVN

| Cường độ mục tiêu | Xi măng tối thiểu (kg/m³) | Tổng chất kết dính tối thiểu (kg/m³) | Nước/(XM+FA+GGBFS+SF) tối đa |
|---|---:|---:|---:|
| Dưới 140 MPa | 400 | 650 | 0.30 |
| Từ 140 đến dưới 170 MPa | 500 | 750 | 0.26 |
| Từ 170 MPa trở lên | 600 | 850 | 0.22 |

Ngoài ra áp dụng xi măng tối đa 1000 kg/m³, tổng FA+GGBFS+SF ≤65% chất kết dính, SF ≤30% chất kết dính; chi tiết 9 khoảng trong `uhpc_engineering.py`. Các con số chỉ là **quy tắc lọc nghiệm do dự án lựa chọn**; không suy ra quan hệ cường độ–xi măng như một định luật. Các công trình sử dụng vật liệu/dưỡng hộ khác có thể cần điều chỉnh. Để dùng kết quả thực tế cần tối ưu và xác nhận bằng mẫu 28 ngày có thiết kế thí nghiệm.

**Lưu ý:** W/B trong đặc trưng học máy `Ty_le_W_B` vẫn là nước / (XM+FA+GGBFS+SF+bột thạch anh); W/B kiểm soát vật lý ở đây là nước / (XM+FA+GGBFS+SF), hai đại lượng KHÔNG giống nhau.

## Hướng dẫn cập nhật một lần

1. Tải ZIP gói này và giải nén.
2. Tải repo GitHub UHPC-AI-Mix-Design về máy (hoặc dùng thư mục repo trên máy).
3. Chép **4 file** `ap_dung_sua_loi.py`, `hide_material_limits.py`, `uhpc_limits.py`, `uhpc_engineering.py` vào **cùng thư mục** có `app.py`, `uhpc_core.py`.
4. Trong terminal mở tại thư mục dự án, chạy:

```bash
python ap_dung_sua_loi.py
```

Hoặc gọi với đường dẫn tới thư mục repo:

```bash
python ap_dung_sua_loi.py "D:\\DuAn\\UHPC-AI-Mix-Design"
```

5. Chương trình chỉ ghi file sau khi kiểm tra cú pháp cả hai file, sao lưu bản cũ thành `*.before_engineering_fix.bak`. Nếu báo `Không tìm thấy ...` thì **mã repo đã khác**, script từ chối sửa; bạn cần cung cấp `app.py` + `uhpc_core.py` hiện tại để điều chỉnh đúng bản.
6. Kiểm tra `python -m py_compile app.py uhpc_core.py uhpc_engineering.py uhpc_limits.py` rồi chạy `streamlit run app.py` tại máy.
7. Cập nhật **`app.py`, `uhpc_core.py`, `uhpc_engineering.py`, `uhpc_limits.py`** lên GitHub cùng nhánh mà Streamlit dùng. Đừng commit file `.bak` hoặc script vá nếu không còn dùng.
8. Tại Streamlit Cloud kiểm tra phần **Manage app → Settings → Repository/Branch/Main file** có chỉ tới đúng commit và file `app.py`; reboot nếu cần để xóa cache.

## Test nhanh

```bash
python -m unittest discover -s tests -p 'test_guardrails.py' -v
```

Nên thử thực tế: mục tiêu 120 và 180 MPa; đồng thời thử tình huống dữ liệu huấn luyện có Xi măng max <600 để kiểm tra trạng thái **không khả thi**, và 60 kg xi măng ở bài toán thuận phải báo lỗi. Với yêu cầu 180 MPa, **mô hình vẫn cần dữ liệu thực nghiệm ở vùng 180 MPa**; ngưỡng >=600 không khiến AI mặc nhiên dự đoán đúng.

## Giới hạn / tương thích

Đã kiểm thử sửa mã trên một bản nguồn tham chiếu và cấu trúc GUI mô phỏng màn hình đã gửi; **chưa truy cập và chạy được bản mới nhất trực tiếp trên GitHub/Streamlit**. Nếu bố cục hàm tối ưu thay đổi, vá sẽ **dừng an toàn** thay vì phá mã. Chưa có phép kiểm thử với `.pkl`/CSV thực tế, nên không tuyên bố app đạt 180 MPa hoặc có độ chính xác cao hơn. Khi không tồn tại cấp phối hợp lệ, app phải báo không có kết quả đạt, không tự hạ giới hạn kỹ thuật.
