from pathlib import Path
import ast
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hide_material_limits import rewrite
from uhpc_limits import (
    INVERSE_MATERIAL_LIMITS, FORWARD_MATERIAL_LIMITS,
    validate_limits,
)

MATERIALS = [
    "Xi_mang", "Tro_bay", "Xil_lo_cao", "Silica_fume",
    "Bot_thach_anh", "Cat", "Nuoc", "Phu_gia_sieu_deo", "Soi_thep",
]


def build_fixture():
    """Mô phỏng chính xác các đoạn giao diện đổi trong repo (nguồn nền là bản app đã đọc)."""
    s = Path('/mnt/data/uhpc_review/app.py').read_text(encoding='utf-8')
    old = '''        left, mid, right = st.columns(3)
        target = left.number_input("Cường độ nén mục tiêu (MPa)", min_value=40.0, max_value=250.0, value=120.0, step=5.0)
        age = mid.number_input("Tuổi nén (ngày)", min_value=1, max_value=365, value=28, step=1)
        max_cement = right.number_input("Xi măng tối đa (kg/m³)", min_value=1.0, max_value=1500.0, value=850.0, step=25.0)
'''
    new = '''        left, mid, right = st.columns(3)
        target = left.number_input("Cường độ nén mục tiêu (MPa)", min_value=40.0, max_value=250.0, value=120.0, step=5.0)
        age = mid.number_input("Tuổi nén (ngày)", min_value=1, max_value=365, value=28, step=1)
        min_cement = right.number_input("Xi măng tối thiểu (kg/m³)", min_value=1.0, max_value=1500.0, value=400.0, step=25.0)
        max_cement = right.number_input("Xi măng tối đa (kg/m³)", min_value=1.0, max_value=1500.0, value=1000.0, step=25.0)
'''
    assert s.count(old) == 1
    s=s.replace(old,new)
    old_inv='''        st.markdown("#### 🧵 Giới hạn và ưu tiên giảm sợi thép")
'''
    added_inv='''        st.markdown("#### 📐 Biên tối thiểu – tối đa của từng vật liệu (kg/m³)")
        st.caption("Các biên là khoảng khảo sát do người dùng đặt, không phải tiêu chuẩn. "
                   "Thuật toán lấy giao với miền huấn luyện; nếu không giao nhau sẽ báo không khả thi.")
        material_limits_inverse = {}
        for key in MATERIALS:
            if key == "Xi_mang":
                material_limits_inverse[key] = (float(min_cement), float(max_cement))
                continue
            default_lo, default_hi = {
                "Tro_bay": (0., 300.), "Xil_lo_cao": (0., 400.),
                "Silica_fume": (0., 250.), "Bot_thach_anh": (0., 350.),
                "Cat": (500., 1200.), "Nuoc": (100., 250.),
                "Phu_gia_sieu_deo": (0., 50.), "Soi_thep": (0., 160.),
            }[key]
            lcol, hcol = st.columns(2)
            low = lcol.number_input(f"{LABELS[key]} — tối thiểu", min_value=0., max_value=3000.,
                                    value=default_lo, step=5., key="inverse_low_" + key)
            high = hcol.number_input(f"{LABELS[key]} — tối đa", min_value=0., max_value=3000.,
                                     value=default_hi, step=5., key="inverse_high_" + key)
            material_limits_inverse[key] = (float(low), float(high))
'''
    assert s.count(old_inv)==1
    s=s.replace(old_inv,added_inv+old_inv)
    # Existing inverse references core rules in current repo (not vital to patch logic).
    target='''            rules = DesignRules(
'''
    assert s.count(target)==1
    s=s.replace(target,'''            rules = DesignRules(
                material_limits=material_limits_inverse,
''')
    # Replace original forward UI with the latest site's input panel (from live source)
    forward_intro='''    with st.form("forward"):
'''
    forward_body='''    st.markdown("#### Giới hạn xi măng cho bài toán thuận")
    cement_min_col, cement_max_col = st.columns(2)
    with cement_min_col:
        forward_cement_min = st.number_input("Xi măng tối thiểu (kg/m³)", min_value=0.0, max_value=1500.0, value=400.0, step=25.0, key="forward_cement_min")
    with cement_max_col:
        forward_cement_max = st.number_input("Xi măng tối đa (kg/m³)", min_value=0.0, max_value=1500.0, value=1000.0, step=25.0, key="forward_cement_max")
    valid_cement_range = forward_cement_min <= forward_cement_max
    if not valid_cement_range:
        st.error("Biên xi măng không hợp lệ: tối thiểu lớn hơn tối đa.")
    st.markdown("#### Giới hạn cốt liệu cho bài toán thuận")
    st.caption("Giới hạn do người dùng lựa chọn, đơn vị kg/m³; không mặc định là yêu cầu tiêu chuẩn. "
               "Áp dụng cho biến `Cat` (cát/cốt liệu), không bao gồm bột thạch anh và sợi thép.")
    agg_min_col, agg_max_col = st.columns(2)
    with agg_min_col:
        forward_agg_min = st.number_input("Cốt liệu tối thiểu (kg/m³)", min_value=0.0, max_value=3000.0,
            value=0.0, step=10.0, key="forward_agg_min")
    with agg_max_col:
        forward_agg_max = st.number_input("Cốt liệu tối đa (kg/m³)", min_value=0.0, max_value=3000.0,
            value=2000.0, step=10.0, key="forward_agg_max")
    valid_agg_range = forward_agg_min <= forward_agg_max
    if "forward_result" in st.session_state:
        old_mix = st.session_state["forward_result"][0]
        if (not valid_agg_range or not (forward_agg_min <= old_mix["Cat"] <= forward_agg_max)
                or not valid_cement_range or not (forward_cement_min <= old_mix["Xi_mang"] <= forward_cement_max)):
            st.session_state.pop("forward_result", None)
    if not valid_agg_range:
        st.error("Giới hạn cốt liệu không hợp lệ: tối thiểu phải nhỏ hơn hoặc bằng tối đa.")
    st.markdown("#### Biên cho tất cả vật liệu — bài toán thuận")
    st.caption("Xi măng và cốt liệu vẫn dùng các biên đã nhập phía trên. "
               "Các thành phần khác được kiểm tra trước khi dự đoán.")
    forward_limits = {"Xi_mang": (forward_cement_min, forward_cement_max),
                      "Cat": (forward_agg_min, forward_agg_max)}
    for key in MATERIALS:
        if key in forward_limits:
            continue
        default_lo, default_hi = {
            "Tro_bay": (0., 300.), "Xil_lo_cao": (0., 400.),
            "Silica_fume": (0., 250.), "Bot_thach_anh": (0., 350.),
            "Nuoc": (100., 250.), "Phu_gia_sieu_deo": (0., 50.),
            "Soi_thep": (0., 160.),
        }[key]
        lcol, hcol = st.columns(2)
        low = lcol.number_input(f"{LABELS[key]} — tối thiểu (thuận)", min_value=0., max_value=3000.,
                                value=default_lo, step=5., key="forward_low_" + key)
        high = hcol.number_input(f"{LABELS[key]} — tối đa (thuận)", min_value=0., max_value=3000.,
                                 value=default_hi, step=5., key="forward_high_" + key)
        forward_limits[key] = (float(low), float(high))
    invalid_forward_limits = [LABELS[k] for k, (lo, hi) in forward_limits.items() if lo > hi]
    if invalid_forward_limits:
        st.error("Giới hạn tối thiểu vượt tối đa: " + ", ".join(invalid_forward_limits))
'''
    assert s.count(forward_intro)==1
    s=s.replace(forward_intro,forward_body+forward_intro)
    start='''    if submit_pred:
        try:
            yhat = predict(model, mix, forward_age, fixed, features)
'''
    forward_check='''    if submit_pred and not (forward_cement_min <= mix["Xi_mang"] <= forward_cement_max):
        st.session_state.pop("forward_result", None)
        st.error(f"Xi măng {mix['Xi_mang']:,.2f} kg/m³ nằm ngoài biên [{forward_cement_min:,.2f}; {forward_cement_max:,.2f}] kg/m³.")
    elif submit_pred and not (forward_agg_min <= mix["Cat"] <= forward_agg_max):
        st.session_state.pop("forward_result", None)
        st.error(f"Cốt liệu nhập vào {mix['Cat']:,.2f} kg/m³ nằm ngoài biên "
                 f"[{forward_agg_min:,.2f}; {forward_agg_max:,.2f}] kg/m³. "
                 "Hãy chỉnh lại cấp phối hoặc biên trước khi dự đoán.")
    elif submit_pred and any(not (lo <= mix[key] <= hi) for key, (lo, hi) in forward_limits.items()):
        st.session_state.pop("forward_result", None)
        invalid = [f"{LABELS[key]}: {mix[key]:g} (biên {lo:g}–{hi:g})"
                   for key, (lo, hi) in forward_limits.items() if not (lo <= mix[key] <= hi)]
        st.error("Vật liệu nằm ngoài giới hạn: " + "; ".join(invalid))
    elif submit_pred:
        try:
            yhat = predict(model, mix, forward_age, fixed, features)
'''
    assert s.count(start)==1
    s=s.replace(start,forward_check)
    return s


class Tests(unittest.TestCase):
    def test_settings_valid(self):
        validate_limits(MATERIALS)
        self.assertEqual(INVERSE_MATERIAL_LIMITS['Xi_mang'], (400, 1000))
        self.assertEqual(FORWARD_MATERIAL_LIMITS['Cat'], (0, 2000))
        self.assertEqual(INVERSE_MATERIAL_LIMITS['Silica_fume'][0], 0)
        self.assertEqual(INVERSE_MATERIAL_LIMITS['Bot_thach_anh'][0], 0)

    def test_patch_compiles_and_hides_controls(self):
        text = rewrite(build_fixture())
        ast.parse(text)
        for forbidden in ('"inverse_low_"', '"forward_low_"', '"forward_cement_min"',
                          '"forward_agg_min"', '"Xi măng tối đa (kg/m³)"',
                          'st.markdown("#### Biên cho tất cả vật liệu'):
            self.assertNotIn(forbidden,text)
        self.assertIn('forward_limits = dict(FORWARD_MATERIAL_LIMITS)', text)
        self.assertIn('material_limits_inverse = dict(INVERSE_MATERIAL_LIMITS)', text)
        self.assertIn('MAX_STEEL_VOL_PERCENT', text)
        self.assertIn('st.session_state["forward_input_key"]', text)

    def test_patch_fails_closed_on_wrong_source(self):
        with self.assertRaises(RuntimeError):
            rewrite('import streamlit as st\n')

    def test_no_double_patch(self):
        fixed=rewrite(build_fixture())
        with self.assertRaises(RuntimeError):
            rewrite(fixed)


if __name__ == '__main__':
    unittest.main()
