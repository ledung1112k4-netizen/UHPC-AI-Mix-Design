"""Dọn giao diện giới hạn vật liệu cho app.py hiện tại của UHPC-AI-Mix-Design.

Chạy: python hide_material_limits.py
Tự tạo backup, từ chối sửa khi mã nguồn khác cấu trúc đã kiểm chứng.
Không chỉnh uhpc_core.py, .pkl hay dữ liệu training.
"""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
APP = HERE / "app.py"
CONFIG = HERE / "uhpc_limits.py"
MARKER = "UHPC_HIDDEN_LIMITS_2026_10_28"


def _replace_one(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"Mã nguồn không tương thích ở '{what}': cần 1 điểm neo, thấy {n}.")
    return src.replace(old, new, 1)


def _replace_span(src: str, start: str, end: str, replacement: str, what: str) -> str:
    if src.count(start) != 1 or src.count(end) != 1:
        raise RuntimeError(f"Mã nguồn không tương thích ở '{what}' (mốc start/end không duy nhất).")
    i = src.index(start)
    j = src.index(end)
    if j <= i:
        raise RuntimeError(f"Sai thứ tự đoạn mã '{what}'.")
    return src[:i] + replacement + src[j:]


def rewrite(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("app.py đã được xử lý; không chạy bản vá lần thứ hai.")
    if 'from uhpc_core import (' not in source:
        raise RuntimeError("Không thấy import uhpc_core trong app.py.")
    s = _replace_one(source, 'from uhpc_core import (',
        'from uhpc_limits import (INVERSE_MATERIAL_LIMITS, FORWARD_MATERIAL_LIMITS,\n'
        '                         MAX_STEEL_VOL_PERCENT, REQUIRED_COMPONENT_MIN, validate_limits)\n'
        'from uhpc_core import (', "Thêm cấu hình nội bộ")
    s = _replace_one(s,
        'with st.sidebar:\n',
        f'# {MARKER}: kiểm tra giới hạn nội bộ ngay khi khởi động app.\n'
        'validate_limits(MATERIALS)\n\nwith st.sidebar:\n',
        'Kiểm tra cấu hình')

    # 1. Bài toán ngược: chỉ để lại 2 thông số đầu vào thực sự cần chọn.
    s = _replace_one(s,
        '        left, mid, right = st.columns(3)\n'
        '        target = left.number_input("Cường độ nén mục tiêu (MPa)", min_value=40.0, max_value=250.0, value=120.0, step=5.0)\n'
        '        age = mid.number_input("Tuổi nén (ngày)", min_value=1, max_value=365, value=28, step=1)\n'
        '        min_cement = right.number_input("Xi măng tối thiểu (kg/m³)", min_value=1.0, max_value=1500.0, value=400.0, step=25.0)\n'
        '        max_cement = right.number_input("Xi măng tối đa (kg/m³)", min_value=1.0, max_value=1500.0, value=1000.0, step=25.0)\n',
        '        left, mid = st.columns(2)\n'
        '        target = left.number_input("Cường độ nén mục tiêu (MPa)", min_value=40.0, max_value=250.0, value=120.0, step=5.0)\n'
        '        age = mid.number_input("Tuổi nén (ngày)", min_value=1, max_value=365, value=28, step=1)\n'
        '        min_cement, max_cement = INVERSE_MATERIAL_LIMITS["Xi_mang"]\n',
        "Bỏ widget xi măng ngược")
    s = _replace_span(s,
        '            a2, b2 = st.columns(2)\n',
        '            c2, d2 = st.columns(2)\n',
        '            min_fa = min_gg = min_sf = min_qp = REQUIRED_COMPONENT_MIN\n',
        'Bỏ các widget lượng tối thiểu FA/GGBFS/SF/bột thạch anh')
    s = _replace_span(s,
        '        st.markdown("#### 📐 Biên tối thiểu – tối đa của từng vật liệu (kg/m³)")\n',
        '        st.markdown("#### 🧵 Giới hạn và ưu tiên giảm sợi thép")\n',
        '        material_limits_inverse = dict(INVERSE_MATERIAL_LIMITS)\n',
        'Bỏ bảng biên tất cả vật liệu ngược')
    s = _replace_span(s,
        '        st.markdown("#### 🧵 Giới hạn và ưu tiên giảm sợi thép")\n',
        '        priority_label = f_right.radio(\n',
        '        max_steel_pct = MAX_STEEL_VOL_PERCENT\n'
        '        st.markdown("#### 🧵 Mức ưu tiên giảm sợi thép")\n'
        '        f_right = st.container()\n',
        'Bỏ giới hạn sợi thép trên giao diện')
    s = _replace_span(s,
        '        st.caption(\n'
        '            f"Giới hạn cứng: {max_steel_pct:.2f}% thể tích = "\n',
        '        run_opt = st.form_submit_button(',
        '',
        'Bỏ chú thích hiển thị giới hạn sợi')
    s = _replace_one(s,
        '    st.caption(\n'
        '        "Giới hạn vật liệu, W/B và thể tích là điều kiện sàng lọc người dùng lựa chọn; "\n'
        '        "không mặc nhiên là yêu cầu tiêu chuẩn. Tổ hợp có mẫu huấn luyện không có nghĩa cấp phối mới đã được kiểm chứng."\n'
        '    )\n',
        '    st.caption("Cấp phối do AI đề xuất cần thí nghiệm xác nhận trước khi ứng dụng thực tế.")\n',
        'Bỏ giới thiệu giao diện biên vật liệu')

    # Cập nhật dấu vân tay ngay sau form; bản cũ chỉ theo dõi choice_key.
    # Tương thích cả với gói UHPC_Upgrade trước, nếu bản đó đã áp dụng.
    add_signature = (
        '    choice_key = (\n'
        '        choice_key, float(target), float(age), mode, similarity_policy,\n'
        '        float(wb_min), float(wb_max), float(margin),\n'
        '        float(air_percent), float(vol_tol), bool(prefer_near_target),\n'
        '        float(max_steel_pct), float(steel_weight),\n'
        '        tuple(sorted(material_limits_inverse.items())),\n'
        '        str(model_path), model_path.stat().st_mtime_ns if model_path else None,\n'
        '    )\n'
    )
    if '    inverse_input_key = (\n' in s:  # Gói nâng cấp trước tạo vân tay đầy đủ riêng
        s = _replace_one(s, '    inverse_input_key = (\n', add_signature + '    inverse_input_key = (\n',
                         'Vân tay trạng thái ngược với gói cũ')
    else:
        s = _replace_one(s, '    if run_opt:\n', add_signature + '    if run_opt:\n',
                         'Vân tay trạng thái ngược')

    # Bỏ chữ "giới hạn tối đa" khỏi bảng mô tả đầu ra, vẫn hiện lượng sợi thực tế.
    old_steel_caption = (
        '        st.caption(\n'
        '            f"Sợi thép đề xuất: {result[\'mix\'][\'Soi_thep\']:.2f} kg/m³ "\n'
        '            f"({result[\'steel_volume_percent\']:.2f}% thể tích); "\n'
        '            f"giới hạn tối đa: {result[\'steel_limit_kg_m3\']:.2f} kg/m³ "\n'
        '            f"({result[\'steel_limit_vol_percent\']:.2f}% thể tích). "\n'
        '            f"Trọng số ưu tiên giảm sợi: {result[\'steel_saving_weight\']:g}."\n'
        '        )\n'
    )
    new_steel_caption = (
        '        st.caption(\n'
        '            f"Sợi thép đề xuất: {result[\'mix\'][\'Soi_thep\']:.2f} kg/m³ "\n'
        '            f"({result[\'steel_volume_percent\']:.2f}% thể tích). "\n'
        '            f"Trọng số ưu tiên giảm sợi: {result[\'steel_saving_weight\']:g}."\n'
        '        )\n'
    )
    s = _replace_one(s, old_steel_caption, new_steel_caption, 'Ẩn giới hạn sợi thép kết quả')

    # 2. Thuận: thay toàn bộ phần thiết lập giới hạn từ UI bằng cấu hình module.
    s = _replace_span(s,
        '    st.markdown("#### Giới hạn xi măng cho bài toán thuận")\n',
        '    with st.form("forward"):\n',
        '    forward_limits = dict(FORWARD_MATERIAL_LIMITS)\n'
        '    forward_cement_min, forward_cement_max = forward_limits["Xi_mang"]\n'
        '    forward_agg_min, forward_agg_max = forward_limits["Cat"]\n'
        '    valid_cement_range = True  # được kiểm tra bởi validate_limits ở đầu app\n'
        '    valid_agg_range = True\n'
        '    invalid_forward_limits = []\n'
        '    if "forward_result" in st.session_state:\n'
        '        old_mix = st.session_state["forward_result"][0]\n'
        '        if any(not (lo <= old_mix[k] <= hi) for k, (lo, hi) in forward_limits.items()):\n'
        '            st.session_state.pop("forward_result", None)\n',
        'Bỏ toàn bộ bảng cài biên ở bài toán thuận')
    s = _replace_span(s,
        '    if submit_pred and not (forward_cement_min <= mix["Xi_mang"] <= forward_cement_max):\n',
        '    elif submit_pred:\n',
        '    if submit_pred and any(not (lo <= mix[k] <= hi) for k, (lo, hi) in forward_limits.items()):\n'
        '        st.session_state.pop("forward_result", None)\n'
        '        invalid = [LABELS[k] for k, (lo, hi) in forward_limits.items()\n'
        '                   if not (lo <= mix[k] <= hi)]\n'
        '        st.error("Các vật liệu ngoài phạm vi tính toán nội bộ: " + ", ".join(invalid))\n',
        'Bỏ số liệu biên trong thông báo lỗi thuận')

    # Tránh lưu kết quả cũ trong phiên khi thay cấu hình/tuổi/đầu vào.
    if '    forward_input_key = (\n' not in s:
        s = _replace_one(s,
            '    if submit_pred and any(not (lo <= mix[k] <= hi) for k, (lo, hi) in forward_limits.items()):\n',
            '    forward_input_key = (\n'
            '        tuple(sorted(forward_limits.items())),\n'
            '        tuple(sorted((k, float(v)) for k, v in fixed.items())),\n'
            '        tuple(sorted((k, float(v)) for k, v in densities.items())),\n'
            '        str(model_path), model_path.stat().st_mtime_ns if model_path else None,\n'
            '    )\n'
            '    if submit_pred and any(not (lo <= mix[k] <= hi) for k, (lo, hi) in forward_limits.items()):\n',
            'Thêm vân tay trạng thái thuận')
        s = _replace_one(s,
            '            st.session_state["forward_result"] = (mix.copy(), float(forward_age), yhat, warnings)\n',
            '            st.session_state["forward_result"] = (mix.copy(), float(forward_age), yhat, warnings)\n'
            '            st.session_state["forward_input_key"] = forward_input_key\n',
            'Lưu vân tay thuận')
        s = _replace_one(s,
            '    if "forward_result" in st.session_state:\n        cmix, cage, yhat, warnings = st.session_state["forward_result"]\n',
            '    if ("forward_result" in st.session_state and\n'
            '            st.session_state.get("forward_input_key") != forward_input_key):\n'
            '        st.session_state.pop("forward_result", None)\n'
            '    if "forward_result" in st.session_state:\n        cmix, cage, yhat, warnings = st.session_state["forward_result"]\n',
            'Không hiển thị dự đoán thuận cũ')
    else:
        # Module nâng cấp cũ đã thêm vân tay; mã này vẫn dùng lại phần biến forward_limits.
        pass

    # Phần mô tả sợi vẫn giữ ý nghĩa khoa học, không công khai biên số học.
    s = _replace_span(s,
        '    st.write("**Giới hạn sợi thép v6:**',
        '    st.write("**15 đặc trưng của notebook:**")',
        '    st.write("Sợi thép được kiểm tra nội bộ khi tối ưu; kết quả AI cần thí nghiệm xác nhận.")\n',
        'Ẩn giải thích giới hạn sợi thép')

    # Nếu đã dùng gói nâng cấp trước, giữ số dòng hỗ trợ SF/QP = 0,
    # nhưng bỏ các cột min/max của dataset khỏi giao diện.
    old_minmax = '''                    "Min train (kg/m³)": float(v.min()),
                    "Max train (kg/m³)": float(v.max()),
''' 
    if old_minmax in s:
        s = _replace_one(s, old_minmax, '', 'Ẩn cột min/max ở tab khảo sát')

    # 3. Không in min/max huấn luyện dưới mục phương pháp. Giữ cảnh báo ngoại suy.
    s = _replace_span(s,
        '    if training_bounds:\n'
        '        st.write("Miền huấn luyện (min/max từng đặc trưng; **không** bảo đảm mọi tổ hợp đều đã có trong dữ liệu):")\n',
        '    st.warning("Không có bảo đảm đạt cường độ thực tế ≥120 MPa.',
        '',
        'Ẩn bảng min/max dữ liệu train khỏi tab phương pháp')

    ast.parse(s)
    if '.number_input("Xi măng tối thiểu' in s or '"inverse_low_"' in s or '"forward_low_"' in s:
        raise RuntimeError("Vẫn còn widget giới hạn vật liệu sau vá; dừng an toàn.")
    return s


def main():
    if not APP.exists():
        raise SystemExit("Không thấy app.py. Chép 2 file của gói vào cùng thư mục với app.py rồi chạy lại.")
    if not CONFIG.exists():
        raise SystemExit("Thiếu uhpc_limits.py.")
    before = APP.read_text(encoding="utf-8")
    after = rewrite(before)  # chạy kiểm tra và parse trước khi thay file thật
    backup = APP.with_name("app.py.backup_before_hidden_limits")
    if backup.exists():
        raise SystemExit("Đã có backup trước đó, không ghi đè. Hãy kiểm tra app.py.")
    shutil.copy2(APP, backup)
    APP.write_text(after, encoding="utf-8")
    print("THÀNH CÔNG: các ô giới hạn vật liệu đã được chuyển vào uhpc_limits.py")
    print(f"Sao lưu: {backup}")
    print("Kiểm tra: python -m py_compile app.py uhpc_limits.py")


if __name__ == "__main__":
    main()
