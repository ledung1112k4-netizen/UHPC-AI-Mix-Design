# UHPC Studio — ứng dụng Streamlit độc lập với Google Colab

## 1. Ứng dụng làm được gì?

- Bài toán thuận: nhập 9 lượng vật liệu tính theo kg/m³, tuổi nén, thông tin vật liệu → **gọi mô hình thật** để dự đoán cường độ nén; trình bày tỷ lệ W/B và bảng khối lượng mẻ.
- Bài toán ngược: nhập cường độ nén mục tiêu (mặc định **≥120 MPa, 28 ngày**), xi măng tối đa, lượng tối thiểu của tro bay và GGBFS, độ rỗng giả định và khoảng W/B → tìm cấp phối bằng **Differential Evolution**.
- Chạy "Nhanh" hoặc "Chuyên sâu". Mô hình chỉ được tải một lần, không huấn luyện lại mỗi lần bấm Thiết kế. Hàm mục tiêu dự đoán nhiều cấp phối cùng lúc.
- Hỗ trợ mẻ **50 kg** (mặc định, quy đổi theo tỷ lệ khối lượng) hoặc mẻ theo **m³**.
- Xuất kết quả theo **CSV**; có kiểm tra không ngoại suy từng đặc trưng, W/B và thể tích tuyệt đối.

**Tuyệt đối không có mô hình giả.** Notebook `.ipynb` đơn thuần KHÔNG lưu kèm `.pkl` đã huấn luyện, nên ứng dụng cần 2 tệp `assets/uhpc_model.pkl` và `assets/training_features.csv` từ Colab. Không có mô hình sẽ báo đúng trạng thái thiếu.

## 2. Xuất mô hình từ notebook `uhpc ggcolab(3).ipynb`

1. Tải hai CSV gốc đúng tên trong notebook lên Google Colab.
2. Chạy **cell 2** (chuẩn hóa, tạo `X_train`) rồi **cell 5** (huấn luyện LightGBM). Đừng chạy các cell ghi đè `uhpc_optimal_lightgbm_model.pkl` sau đó.
3. Tải tệp `COLAB_EXPORT.py` lên Colab. Tạo cell mới và chạy:

   ```python
   %run -i /content/COLAB_EXPORT.py
   ```

4. Colab tạo và cho tải về `uhpc_ai_assets.zip`, bên trong có `assets/uhpc_model.pkl` và `assets/training_features.csv`.
5. Giải nén và chép hai tệp đó vào đúng thư mục `assets/` của app. **Đừng chỉnh tên file.**

Vì notebook hiện hành dùng tập kiểm thử trong quá trình chọn mô hình và có rủi ro trùng mẫu, các chỉ số hiện có CHƯA là bằng chứng đủ mạnh cho thiết kế kết cấu. Bạn nên huấn luyện/kiểm chứng lại khi có dữ liệu được chuẩn hóa.

## 3. Chạy ứng dụng trên máy cá nhân

Python 3.10–3.12 được khuyến nghị. Trong thư mục chứa `app.py`:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Chương trình sẽ mở trang web trong trình duyệt. Truy cập thông qua địa chỉ mà Streamlit báo trong Terminal.

## 4. Triển khai không phụ thuộc Google Colab (Streamlit Community Cloud)

1. Tạo repository GitHub (riêng tư nếu dữ liệu/mô hình có giới hạn bản quyền).
2. Upload `app.py`, `uhpc_core.py`, `requirements.txt`, thư mục `.streamlit`, và các tệp trong `assets/`.
3. Mở https://share.streamlit.io/ và chọn **Create app** từ repository.
4. Main file path: **`app.py`**; chọn nhánh chứa dự án; bấm **Deploy**.
5. Khi app chạy, nhấn tab "Thiết kế cấp phối" để tối ưu và "Dự đoán cường độ" để kiểm tra.

Giới hạn phiên bản `requirements.txt` hiện nay chỉ là khung tương thích. Nếu tệp pickle phụ thuộc vào phiên bản `lightgbm`/`scikit-learn` của Colab, hãy cố định lại đúng phiên bản đã dùng khi huấn luyện (ví dụ kiểm tra bằng `import lightgbm, sklearn; print(lightgbm.__version__, sklearn.__version__)`). Không tải `.pkl` không tin cậy vì pickle có thể thực thi mã tùy ý.

## 5. Thuật toán và những điểm đã sửa so với notebook

- Giữ nguyên đủ **15 đặc trưng** và thứ tự từ tệp mô hình thực tế.
- Cố định tuổi do người dùng chọn (mặc định 28 ngày), không tối ưu tuổi.
- Bỏ `Ty_le_W_B` khỏi biến độc lập; tự tính theo công thức của notebook:
  `Ty_le_W_B = Nuoc/(Xi_mang + Tro_bay + Xil_lo_cao + Silica_fume + Bot_thach_anh)`.
- Tính thêm **W/B theo nghĩa thông thường** với mẫu số không có bột thạch anh.
- Giữ cố định `FlyAsh_SiO2`, `FlyAsh_Blaine`, `GGBFS_Grade`, `He_so_quy_doi_mau`; cho chỉnh theo thông tin vật liệu thực tế.
- Phạt chỉ khi dự đoán **thấp hơn** `target + margin`, không phạt cường độ lớn hơn mục tiêu.
- Ràng buộc cả **FA>0 và GGBFS>0**, W/B, tổng thể tích tuyệt đối, phạm vi min–max từng đặc trưng từ tập đã huấn luyện.
- **Kiểm tra sau tối ưu**; KHÔNG dán nhãn đạt cho kết quả dưới mục tiêu hoặc không đạt thể tích.
- Khối lượng riêng mặc định chỉ là giá trị **tham khảo**, người dùng phải sửa theo vật liệu và trạng thái ẩm thực tế. Lượng vật liệu kg/m³ cần phản ánh đúng quy ước cân ẩm/khô.
- Đẩy nhiều dòng đặc trưng qua `model.predict(DataFrame)` mỗi lần để giảm chi phí vòng lặp; dùng 25 vòng ở chế độ nhanh, 90 vòng ở chế độ chuyên sâu.

## 6. Những giới hạn không được bỏ qua

- Ứng dụng KHÔNG tự tạo dữ liệu thực nghiệm, KHÔNG cam kết mẫu đạt 120 MPa chỉ vì AI dự đoán đạt.
- Không có độ chảy, tính công tác, độ phân tán sợi, hóa học nước, chế độ dưỡng hộ (ngoài số ngày) và kết quả kiểm chứng mẫu mới → cần thực nghiệm trước khi áp dụng.
- `Sand_or_TotalAgg_kg_m3` ở một bộ dữ liệu cũ có thể là tổng cốt liệu chứ không chỉ cát thạch anh. Cần kiểm tra bản chất dữ liệu trước khi coi là cùng một biến.
- Miền min/max của từng biến chỉ là ràng buộc sơ bộ; không đảm bảo các kết hợp nằm gần phân bố dữ liệu huấn luyện. Cần nâng cấp đánh giá khoảng cách dữ liệu/độ bất định và validation theo nhóm bài báo cho phiên bản nghiên cứu chính thức.
- Khi thuật toán không tìm ra nghiệm khả thi, sẽ hiện cảnh báo đỏ và bảng **ứng viên không hợp lệ**, không cam kết nghiệm tối ưu toàn cục.
- Không dùng các thông số giới hạn sàng lọc trong app như thể chúng là các yêu cầu bắt buộc của một tiêu chuẩn cụ thể.

## 7. Cấu trúc thư mục

```text
UHPC_Streamlit_App/
├── app.py
├── uhpc_core.py
├── COLAB_EXPORT.py
├── HUONG_DAN.md
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── uhpc_model.pkl           <- bạn xuất từ Colab
│   └── training_features.csv   <- bạn xuất từ Colab
└── tests/
    └── test_core.py
```

## 8. Kiểm thử

```bash
pip install pytest
pytest -q
```

Lưu ý: kiểm thử tự động dùng **model giả lập trong phạm vi unittest**, KHÔNG chứa dữ liệu hoặc kết quả AI để đưa vào ứng dụng.
