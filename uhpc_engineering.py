"""Bộ lọc NỘI BỘ cho nghiên cứu cấp phối UHPC, KHÔNG phải tiêu chuẩn hoặc bảo đảm cường độ.

Các ngưỡng theo cấp cường độ là chính sách nghiên cứu thận trọng; cần hiệu chỉnh với
nguyên liệu, điều kiện dưỡng hộ và dữ liệu thí nghiệm của người sử dụng.
W/B vật lý KHÔNG tính bột thạch anh; đặc trưng W/B của notebook vẫn giữ nguyên.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

MATERIAL_LIMITS = {
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


@dataclass(frozen=True)
class StrengthTier:
    cement_min: float
    binder_min: float
    conventional_wb_max: float


def tier_for_target(target_mpa: float) -> StrengthTier:
    """Giới hạn sàng lọc *giả định*, không suy ra cường độ từ lượng xi măng."""
    if not math.isfinite(float(target_mpa)) or target_mpa <= 0:
        raise ValueError("Cường độ mục tiêu phải hữu hạn và dương.")
    if target_mpa >= 170:
        return StrengthTier(600.0, 850.0, 0.22)
    if target_mpa >= 140:
        return StrengthTier(500.0, 750.0, 0.26)
    return StrengthTier(400.0, 650.0, 0.30)


def tighten_bounds(bounds, material_names, target_mpa: float):
    """Giao biên train + thiết kế + ngưỡng cấp mục tiêu, không nới biên train."""
    names = list(material_names)
    if len(bounds) != len(names) or set(names) != set(MATERIAL_LIMITS):
        raise ValueError("Số biến tối ưu không khớp bộ 9 vật liệu UHPC.")
    tier = tier_for_target(target_mpa)
    updated = []
    for key, (raw_lo, raw_hi) in zip(names, bounds):
        hard_lo, hard_hi = MATERIAL_LIMITS[key]
        if key == 'Xi_mang':
            hard_lo = max(hard_lo, tier.cement_min)
        low = max(float(raw_lo), hard_lo)
        high = min(float(raw_hi), hard_hi)
        if low > high + 1e-8:
            raise ValueError(
                f"Không thể tối ưu {target_mpa:g} MPa: giới hạn nội bộ của {key} "
                "không giao với miền huấn luyện hoặc lựa chọn vật liệu. "
                "Không tự nới biên và không tạo cấp phối giả."
            )
        updated.append((low, max(low, high)))
    return updated


def _array(mix_matrix, material_names):
    arr = np.asarray(mix_matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    names = list(material_names)
    if arr.shape[1] != len(names):
        raise ValueError("Số cột vật liệu không khớp ma trận tối ưu.")
    index = {k: i for i, k in enumerate(names)}
    if not set(MATERIAL_LIMITS).issubset(index):
        raise ValueError("Thiếu biến vật liệu trong thiết kế cấp phối.")
    return arr, index


def assess_population(mix_matrix, material_names, target_mpa: float):
    """Trả mask hợp lệ và penalty vector theo các quan hệ vật lý-định lượng.

    Penalty chỉ dẫn hướng tối ưu; mask kiểm tra cứng, KHÔNG đánh đồng với AI.
    """
    a, idx = _array(mix_matrix, material_names)
    tier = tier_for_target(target_mpa)
    cem = a[:, idx['Xi_mang']]
    fa = a[:, idx['Tro_bay']]
    slag = a[:, idx['Xil_lo_cao']]
    sf = a[:, idx['Silica_fume']]
    water = a[:, idx['Nuoc']]
    binder = cem + fa + slag + sf
    replacement = fa + slag + sf
    safe_binder = np.maximum(binder, 1e-9)
    w_b = water / safe_binder
    replacement_fraction = replacement / safe_binder
    silica_fraction = sf / safe_binder

    # Tổng chất kết dính, W/B thường và tỷ lệ thay thế đều là rào kỹ thuật.
    violations = [
        np.maximum(tier.binder_min - binder, 0.0),
        1000 * np.maximum(w_b - tier.conventional_wb_max, 0.0),
        1000 * np.maximum(0.10 - w_b, 0.0),
        1000 * np.maximum(replacement_fraction - 0.65, 0.0),
        1000 * np.maximum(silica_fraction - 0.30, 0.0),
    ]
    for name, (lo, hi) in MATERIAL_LIMITS.items():
        if name == 'Xi_mang':
            lo = max(lo, tier.cement_min)
        value = a[:, idx[name]]
        violations.append(np.maximum(lo - value, 0.0))
        violations.append(np.maximum(value - hi, 0.0))
    valid = np.isfinite(a).all(axis=1) & (a >= -1e-8).all(axis=1)
    for v in violations:
        valid &= v <= 1e-6
    # Sufficient to penalize strongly; a finite loss is important for DE.
    penalty = 2.0e4 * sum(v * v for v in violations)
    penalty = np.where(valid, 0.0, penalty + 1e6)
    penalty = np.nan_to_num(penalty, nan=1e12, posinf=1e12)
    return valid, penalty


def mix_checks(mix: dict[str, float], target_mpa: float) -> dict[str, bool]:
    names = list(MATERIAL_LIMITS)
    matrix = np.asarray([[float(mix[k]) for k in names]])
    a, idx = _array(matrix, names)
    tier = tier_for_target(target_mpa)
    binder = float(sum(mix[k] for k in ('Xi_mang', 'Tro_bay', 'Xil_lo_cao', 'Silica_fume')))
    w_b = float(mix['Nuoc']) / binder if binder > 0 else float('inf')
    ratio = float(mix['Tro_bay'] + mix['Xil_lo_cao'] + mix['Silica_fume']) / binder if binder > 0 else float('inf')
    sf_ratio = float(mix['Silica_fume']) / binder if binder > 0 else float('inf')
    within = all(lo - 1e-7 <= mix[k] <= hi + 1e-7 for k, (lo, hi) in MATERIAL_LIMITS.items())
    return {
        'Xi măng đáp ứng ngưỡng sàng lọc nội bộ': float(mix['Xi_mang']) >= tier.cement_min - 1e-7 and float(mix['Xi_mang']) <= 1000 + 1e-7,
        'Tổng chất kết dính đáp ứng ngưỡng sàng lọc': binder >= tier.binder_min - 1e-7,
        'Nước/chất kết dính thông thường trong khoảng nghiên cứu': 0.10 - 1e-7 <= w_b <= tier.conventional_wb_max + 1e-7,
        'Tổng FA + GGBFS + SF không vượt 65% chất kết dính': ratio <= 0.65 + 1e-7,
        'Silica fume không vượt 30% chất kết dính': sf_ratio <= 0.30 + 1e-7,
        'Tất cả vật liệu nằm trong biên nội bộ': within,
    }


def forward_issues(mix: dict[str, float]) -> list[str]:
    """Đối với bài toán thuận, chặn cấp phối quá xa định hướng nghiên cứu.

    Không đặt điều kiện theo cường độ DỰ ĐOÁN; nhằm tránh việc AI hợp thức hóa
    cấp phối vô lý. Không bắt buộc khép kín thể tích vì đầu vào có thể chưa đủ.
    """
    result = []
    binder = sum(float(mix[k]) for k in ('Xi_mang', 'Tro_bay', 'Xil_lo_cao', 'Silica_fume'))
    for key, (lo, hi) in MATERIAL_LIMITS.items():
        value = float(mix[key])
        if not math.isfinite(value) or value < lo - 1e-7 or value > hi + 1e-7:
            result.append(f"{key}: {value:g} kg/m³ ngoài phạm vi nghiên cứu nội bộ")
    if binder < 650:
        result.append('Tổng chất kết dính dưới 650 kg/m³')
    if binder <= 0:
        result.append('Không có chất kết dính hợp lệ')
    else:
        w_b = float(mix['Nuoc']) / binder
        if not 0.10 <= w_b <= 0.30:
            result.append(f'W/B thông thường = {w_b:.3f} ngoài khoảng sàng lọc')
    return result
