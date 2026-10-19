"""Chạy nội dung file này trong 1 cell Google Colab, SAU KHI đã chạy cell huấn luyện.

Ví dụ trong Colab: %run -i /content/COLAB_EXPORT.py
File sinh ra: /content/uhpc_ai_assets.zip (tải về, giải nén vào thư mục app).
"""
from pathlib import Path
import pickle
import zipfile
import json
import numpy as np
import pandas as pd

MODEL_SOURCE = Path('/content/uhpc_optimal_lightgbm_model.pkl')
if not MODEL_SOURCE.exists():
    MODEL_SOURCE = Path('uhpc_optimal_lightgbm_model.pkl')
if not MODEL_SOURCE.exists():
    raise FileNotFoundError('Chưa có .pkl. Chạy cell huấn luyện và lưu mô hình trong notebook trước.')

# Chỉ đọc tệp .pkl mà CHÍNH BẠN tạo bằng notebook trong Colab.
with MODEL_SOURCE.open('rb') as handle:
    saved = pickle.load(handle)
if not isinstance(saved, dict) or not {'model', 'feature_cols'} <= saved.keys():
    raise ValueError('File mô hình không có model và feature_cols như notebook.')
model, features = saved['model'], list(saved['feature_cols'])
expected = [
    'Xi_mang', 'Tro_bay', 'Xil_lo_cao', 'Silica_fume', 'Bot_thach_anh',
    'Cat', 'Nuoc', 'Phu_gia_sieu_deo', 'Soi_thep', 'Tuoi_duong_ho',
    'Ty_le_W_B', 'FlyAsh_SiO2', 'FlyAsh_Blaine', 'GGBFS_Grade', 'He_so_quy_doi_mau',
]
if len(features) != 15 or set(features) != set(expected):
    raise ValueError(f'Model không có đủ 15 đặc trưng mong đợi: {features}')
if hasattr(model, 'n_features_in_') and model.n_features_in_ != 15:
    raise ValueError('Model .pkl và danh sách 15 đặc trưng không cùng cấu trúc')
names = getattr(model, 'feature_name_', None)
if names is None:
    names = getattr(model, 'feature_names_in_', None)
if names is not None and list(names) != features:
    raise ValueError('Danh sách đặc trưng trong .pkl không đúng thứ tự mô hình')

# Dải giới hạn phải đúng với dữ liệu ĐÃ HUẤN LUYỆN mô hình đang xuất.
# Notebook hiện nay chứa X_train sau cell 2; hãy chạy đúng thứ tự cell 2 rồi cell 5.
if 'X_train' not in globals() or not isinstance(X_train, pd.DataFrame):
    raise RuntimeError('Không tìm được X_train. Chạy cell 2 và cell 5 (huấn luyện) trước.')
if set(features) - set(X_train.columns):
    raise ValueError('X_train không khớp 15 đặc trưng đã lưu cùng mô hình.')
train = X_train.loc[:, features].apply(pd.to_numeric, errors='coerce').dropna()
if len(train) < 10:
    raise ValueError('X_train không đủ mẫu hợp lệ để xuất giới hạn.')

output = Path('/content/uhpc_ai_assets.zip')
package = {
    'model': model,
    'feature_cols': features,
    'feature_bounds': {k: (float(train[k].min()), float(train[k].max())) for k in features},
    'metadata': {
        'algorithm': type(model).__name__,
        'trained_observations_provided': len(train),
        'description': 'Model học từ notebook của người dùng. Không tái huấn luyện hoặc tạo số liệu dự đoán.',
        'validation_status': 'Chưa kiểm chứng độc lập tính tổng quát; kết quả test notebook có thể bị ảnh hưởng bởi quá trình lựa chọn mô hình.',
        'model_ratio_definition': 'Nuoc/(Xi_mang+Tro_bay+Xil_lo_cao+Silica_fume+Bot_thach_anh)',
        'training_range_warning': 'Phạm vi min/max không bảo đảm tổ hợp đầu vào mới nằm trong phân bố dữ liệu thực.',
    },
}
assets = Path('/content/uhpc_export_assets')
assets.mkdir(exist_ok=True)
model_file = assets / 'uhpc_model.pkl'
with model_file.open('wb') as handle:
    pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
train_file = assets / 'training_features.csv'
train.to_csv(train_file, index=False, encoding='utf-8-sig')
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(model_file, 'assets/uhpc_model.pkl')
    archive.write(train_file, 'assets/training_features.csv')

print(f'Xuất thành công: {output}')
print(f'Mô hình thực tế: {type(model).__name__}; số đặc trưng: {len(features)}; số mẫu train: {len(train)}')
print('Mở ZIP, chép thư mục assets/ vào cùng thư mục với app.py, rồi chạy Streamlit.')
print('Lưu ý: Không chia sẻ .pkl hay CSV nếu dữ liệu của bạn có hạn chế quyền sử dụng.')
try:
    from google.colab import files
    files.download(str(output))
except ImportError:
    print('Không ở trong Google Colab: mở đường dẫn ZIP ở trên để lấy kết quả.')
