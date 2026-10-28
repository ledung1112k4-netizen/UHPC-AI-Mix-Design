"""Giới hạn khối lượng nội bộ (kg/m³), KHÔNG hiển thị thành widget Streamlit.

Đây là cấu hình nghiên cứu ban đầu của repository, KHÔNG phải giới hạn TCVN.
Tự hiệu chỉnh dựa trên dữ liệu huấn luyện và kết quả kiểm chứng thực tế.
"""

# Thiết kế ngược: giới hạn tìm kiếm được giao với các biên huấn luyện
INVERSE_MATERIAL_LIMITS = {
    "Xi_mang": (400.0, 1000.0),
    "Tro_bay": (0.0, 300.0),
    "Xil_lo_cao": (0.0, 400.0),
    "Silica_fume": (0.0, 250.0),
    "Bot_thach_anh": (0.0, 350.0),
    "Cat": (500.0, 1200.0),
    "Nuoc": (100.0, 250.0),
    "Phu_gia_sieu_deo": (0.0, 50.0),
    "Soi_thep": (0.0, 160.0),
}

# Dự đoán thuận: phạm vi nhập cần kiểm tra trước khi gọi mô hình
FORWARD_MATERIAL_LIMITS = {
    **INVERSE_MATERIAL_LIMITS,
    "Cat": (0.0, 2000.0),  # kế thừa mặc định cốt liệu của app cũ
}

MAX_STEEL_VOL_PERCENT = 2.0
REQUIRED_COMPONENT_MIN = 20.0  # áp dụng khi chính sách vật liệu là 'required'


def validate_limits(materials):
    """Bắt lỗi cấu hình thiếu khóa, giá trị không hợp lệ hoặc biên đảo."""
    import math
    expected = set(materials)
    for group_name, limits in (
        ("INVERSE_MATERIAL_LIMITS", INVERSE_MATERIAL_LIMITS),
        ("FORWARD_MATERIAL_LIMITS", FORWARD_MATERIAL_LIMITS),
    ):
        if set(limits) != expected:
            raise ValueError(f"{group_name}: khóa khác với MATERIALS của mô hình: "
                             f"thiếu={expected - set(limits)}, thừa={set(limits) - expected}")
        for key, pair in limits.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"{group_name}.{key}: cần (min, max)")
            low, high = pair
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
                raise ValueError(f"{group_name}.{key}: biên phải hữu hạn và 0 <= min <= max")
    if not (math.isfinite(MAX_STEEL_VOL_PERCENT) and 0 <= MAX_STEEL_VOL_PERCENT <= 100):
        raise ValueError("MAX_STEEL_VOL_PERCENT không hợp lệ")
    if not (math.isfinite(REQUIRED_COMPONENT_MIN) and REQUIRED_COMPONENT_MIN > 0):
        raise ValueError("REQUIRED_COMPONENT_MIN phải dương")
