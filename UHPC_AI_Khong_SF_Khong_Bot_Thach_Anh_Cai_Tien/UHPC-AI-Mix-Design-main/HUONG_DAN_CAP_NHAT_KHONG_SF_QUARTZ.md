# Cập nhật tìm cấp phối không SF và không bột thạch anh

- Trong tab Thiết kế ngược, chọn **Không sử dụng** cho Silica fume và Bột thạch anh (mặc định).
- Chọn **Khám phá có cảnh báo ngoại suy** nếu tập train không có đủ mẫu cùng tổ hợp vật liệu.
- Giữ tro bay và GGBFS theo nhu cầu, tuổi 28 ngày, mục tiêu 120 MPa.
- Kết quả là dự đoán của mô hình gốc, KHÔNG phải cường độ thí nghiệm. Không được kết luận đạt 120 MPa nếu chưa thử nghiệm.
- Dữ liệu 1307 hàng trong `assets/training_features.csv` có 83 hàng đồng thời SF=0 và bột thạch anh=0, nhưng khi thêm điều kiện FA/GGBFS và tuổi 28 ngày có thể còn 0 hàng.
- Chế độ ưu tiên độ tin cậy vẫn yêu cầu ít nhất 6 mẫu tương đồng và sẽ dừng nếu không đủ. Chế độ khám phá bỏ điều kiện phải có 6 mẫu cùng nhóm, vẫn đánh giá khoảng cách tới toàn bộ train và báo ngoại suy.
- Mô hình vẫn giữ nguyên 15 đặc trưng và tỷ lệ Ty_le_W_B tương thích với mô hình huấn luyện.
- Để triển khai: upload các file đã chỉnh lên GitHub nhánh main; Streamlit Community Cloud sẽ tự cập nhật app.
