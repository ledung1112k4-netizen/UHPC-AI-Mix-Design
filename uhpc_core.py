"""Tính toán và tối ưu cấp phối UHPC từ mô hình AI đã huấn luyện.

Không huấn luyện mô hình, không tự sinh cường độ thực nghiệm hoặc cường độ AI.
Đầu ra tối ưu là phương án sàng lọc, không phải cấp phối đã nghiệm thu.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

MATERIALS = [
    "Xi_mang", "Tro_bay", "Xil_lo_cao", "Silica_fume", "Bot_thach_anh",
    "Cat", "Nuoc", "Phu_gia_sieu_deo", "Soi_thep",
]
FEATURES_15 = MATERIALS + [
    "Tuoi_duong_ho", "Ty_le_W_B", "FlyAsh_SiO2", "FlyAsh_Blaine",
    "GGBFS_Grade", "He_so_quy_doi_mau",
]
LABELS = {
    "Xi_mang": "Xi măng", "Tro_bay": "Tro bay (FA)",
    "Xil_lo_cao": "Xỉ lò cao (GGBFS)", "Silica_fume": "Silica fume",
    "Bot_thach_anh": "Bột thạch anh", "Cat": "Cát / cốt liệu",
    "Nuoc": "Nước", "Phu_gia_sieu_deo": "Phụ gia siêu dẻo",
    "Soi_thep": "Sợi thép",
}
# Các giá trị dưới đây là giá trị minh họa để bắt đầu nhập, KHÔNG là cấp phối đã được chứng thực.
# Chúng được chép từ 'test_mix' trong cell 23 của notebook do người dùng cung cấp.
NOTEBOOK_TEST_MIX = {
    "Xi_mang": 650.0, "Tro_bay": 120.0, "Xil_lo_cao": 80.0,
    "Silica_fume": 100.0, "Bot_thach_anh": 60.0, "Cat": 980.0,
    "Nuoc": 155.0, "Phu_gia_sieu_deo": 25.0, "Soi_thep": 120.0,
}
# Tỷ trọng khối lượng riêng tham khảo có thể chỉnh theo phiếu kiểm nghiệm vật liệu (kg/m³).
DEFAULT_DENSITIES = {
    "Xi_mang": 3150.0, "Tro_bay": 2300.0, "Xil_lo_cao": 2900.0,
    "Silica_fume": 2200.0, "Bot_thach_anh": 2650.0,
    "Cat": 2650.0, "Nuoc": 1000.0, "Phu_gia_sieu_deo": 1100.0,
    "Soi_thep": 7850.0,
}
BINDER_CHOICES = {
    "BOTH": "Xi măng + Tro bay + GGBFS",
    "FA_ONLY": "Xi măng + Tro bay (không GGBFS)",
    "GGBFS_ONLY": "Xi măng + GGBFS (không tro bay)",
    "CEMENT_ONLY": "Chỉ xi măng (không FA/GGBFS)",
    "CUSTOM": "Tùy chỉnh thành phần chất liên kết",
}
POLICY_CHOICES = {"required": "Bắt buộc", "optional": "Tùy chọn", "off": "Không sử dụng"}
BINDER_POLICIES = {
    "BOTH": ("required", "required"),
    "FA_ONLY": ("required", "off"),
    "GGBFS_ONLY": ("off", "required"),
    "CEMENT_ONLY": ("off", "off"),
}

DEFAULT_FIXED = {
    "FlyAsh_SiO2": 60.0, "FlyAsh_Blaine": 3500.0,
    "GGBFS_Grade": 100.0, "He_so_quy_doi_mau": 1.0,
}


@dataclass(frozen=True)
class DesignRules:
    wb_min: float = 0.14
    wb_max: float = 0.24
    air_fraction: float = 0.02
    volume_tolerance: float = 0.02
    min_flyash: float = 20.0
    min_ggbfs: float = 20.0
    margin_mpa: float = 0.0
    binder_mode: str = "BOTH"
    custom_fa: str = "required"
    custom_ggbfs: str = "required"
    silica_mode: str = "optional"
    quartz_mode: str = "optional"
    min_silica: float = 20.0
    min_quartz: float = 20.0

    def validate(self) -> None:
        if not (0 < self.wb_min <= self.wb_max < 1):
            raise ValueError("Giới hạn W/B không hợp lệ.")
        if not (0 <= self.air_fraction < 0.2):
            raise ValueError("Hàm lượng khí dự kiến cần nằm trong 0–20%.")
        if not (0 < self.volume_tolerance <= 0.1):
            raise ValueError("Sai số thể tích phải lớn hơn 0 và không quá 0,1 m³.")
        if self.binder_mode not in BINDER_CHOICES:
            raise ValueError("Tổ hợp chất liên kết không hợp lệ.")
        if any(x not in POLICY_CHOICES for x in (
            self.custom_fa, self.custom_ggbfs, self.silica_mode, self.quartz_mode
        )):
            raise ValueError("Trạng thái vật liệu phải là bắt buộc/tùy chọn/không sử dụng.")
        minima = (self.min_flyash, self.min_ggbfs, self.min_silica, self.min_quartz)
        if not all(np.isfinite(v) and v >= 0 for v in minima):
            raise ValueError("Hàm lượng tối thiểu các vật liệu phải không âm và hữu hạn.")
        for k, policy in material_policies(self).items():
            if policy == "required" and material_minimum(self, k) <= 0:
                raise ValueError(f"{LABELS[k]} bắt buộc phải có lượng tối thiểu > 0.")
        if not np.isfinite(self.margin_mpa) or self.margin_mpa < 0:
            raise ValueError("Biên dự phòng cường độ không được âm.")


def material_policies(rules: DesignRules) -> dict[str, str]:
    """Quy tắc có/không/tùy chọn cho 4 thành phần; QP là filler, không phải binder."""
    if rules.binder_mode == "CUSTOM":
        fa, gg = rules.custom_fa, rules.custom_ggbfs
    else:
        fa, gg = BINDER_POLICIES[rules.binder_mode]
    return {
        "Tro_bay": fa,
        "Xil_lo_cao": gg,
        "Silica_fume": rules.silica_mode,
        "Bot_thach_anh": rules.quartz_mode,
    }


def material_minimum(rules: DesignRules, key: str) -> float:
    return {
        "Tro_bay": rules.min_flyash,
        "Xil_lo_cao": rules.min_ggbfs,
        "Silica_fume": rules.min_silica,
        "Bot_thach_anh": rules.min_quartz,
    }[key]


def count_training_matches(training: pd.DataFrame, rules: DesignRules) -> int:
    """Số dòng huấn luyện có trạng thái vật liệu tương ứng, không là đánh giá độ tin cậy.

    Sự xuất hiện một mẫu cùng tổ hợp KHÔNG đảm bảo cấp phối mới nằm trong phân bố train.
    """
    rules.validate()
    policies = material_policies(rules)
    missing = set(policies) - set(training.columns)
    if missing:
        raise ValueError("Dữ liệu huấn luyện thiếu: " + ", ".join(sorted(missing)))
    mask = np.ones(len(training), dtype=bool)
    for k, policy in policies.items():
        values = pd.to_numeric(training[k], errors="coerce").to_numpy(dtype=float)
        mask &= np.isfinite(values)
        if policy == "off":
            mask &= np.abs(values) <= 1e-8
        elif policy == "required":
            mask &= values >= material_minimum(rules, k) - 1e-8
    return int(np.sum(mask))


def material_search_bounds(
    training_bounds: dict[str, tuple[float, float]],
    rules: DesignRules,
    max_cement: float,
) -> list[tuple[float, float]]:
    """Áp ràng buộc từng vật liệu trực tiếp vào vector tối ưu DE."""
    rules.validate()
    policies = material_policies(rules)
    bds = []
    for k in MATERIALS:
        lo, hi = training_bounds[k]
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Biên dữ liệu {LABELS[k]} không hữu hạn.")
        lo = max(lo, 0.0)
        if k == "Xi_mang":
            lo = max(lo, 1.0)  # vẫn là UHPC xi măng; không tối ưu nghiệm không xi măng
            hi = min(hi, max_cement)
        if k in policies:
            if policies[k] == "off":
                if training_bounds[k][0] > 1e-8 or training_bounds[k][1] < -1e-8:
                    raise ValueError(
                        f"Không thể đặt {LABELS[k]} = 0: mô hình chưa học giá trị 0 "
                        f"trong dải [{training_bounds[k][0]:g}, {training_bounds[k][1]:g}]."
                    )
                lo, hi = 0.0, 0.0
            elif policies[k] == "required":
                lo = max(lo, material_minimum(rules, k))
        if lo > hi:
            raise ValueError(
                f"Ràng buộc {LABELS[k]} không khả thi trong dữ liệu huấn luyện "
                f"[{training_bounds[k][0]:g}, {training_bounds[k][1]:g}]."
            )
        bds.append((lo, hi))
    return bds


def check_model_columns(model: Any, feature_cols: list[str]) -> None:
    if len(feature_cols) != 15 or len(set(feature_cols)) != 15 or set(feature_cols) != set(FEATURES_15):
        raise ValueError("Mô hình phải có chính xác 15 đặc trưng giống notebook đã gửi.")
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != 15:
        raise ValueError(f"Mô hình thực tế chỉ có {n_features} đặc trưng, không tương thích.")
    names = getattr(model, "feature_name_", None)
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    if names is not None:
        names_list = list(names)
        if names_list != list(feature_cols):
            raise ValueError("Thứ tự đặc trưng của mô hình không trùng với danh sách đã lưu.")


def feature_frame(mixes: pd.DataFrame, age: float, fixed: dict[str, float], feature_cols: list[str]) -> pd.DataFrame:
    missing = set(MATERIALS) - set(mixes.columns)
    if missing:
        raise ValueError(f"Thiếu cột vật liệu: {sorted(missing)}")
    mx = mixes.loc[:, MATERIALS].astype(float)
    if not np.isfinite(mx.to_numpy()).all() or (mx.to_numpy() < 0).any():
        raise ValueError("Khối lượng vật liệu phải là các số hữu hạn không âm.")
    if not np.isfinite(age) or age <= 0:
        raise ValueError("Tuổi dưỡng hộ phải lớn hơn 0.")
    for k in DEFAULT_FIXED:
        if k not in fixed or not np.isfinite(fixed[k]):
            raise ValueError(f"Thiếu thuộc tính vật liệu hợp lệ: {k}")
    # Đồng nhất 100% với đặc trưng đã dùng trong notebook, kể cả QP ở mẫu số.
    total_powder = mx[["Xi_mang", "Tro_bay", "Xil_lo_cao", "Silica_fume", "Bot_thach_anh"]].sum(axis=1)
    if (total_powder <= 0).any():
        raise ValueError("Tổng lượng bột phải lớn hơn 0.")
    out = mx.copy()
    out["Tuoi_duong_ho"] = float(age)
    out["Ty_le_W_B"] = mx["Nuoc"] / total_powder
    for k in DEFAULT_FIXED:
        out[k] = float(fixed[k])
    return out.loc[:, feature_cols]


def physical_quantities(mixes: pd.DataFrame, densities: dict[str, float]) -> pd.DataFrame:
    mx = mixes.loc[:, MATERIALS].astype(float)
    dens = np.array([densities[k] for k in MATERIALS], dtype=float)
    if not np.isfinite(dens).all() or (dens <= 0).any():
        raise ValueError("Khối lượng riêng của từng vật liệu phải dương và hữu hạn.")
    binder = mx[["Xi_mang", "Tro_bay", "Xil_lo_cao", "Silica_fume"]].sum(axis=1).to_numpy()
    powder = binder + mx["Bot_thach_anh"].to_numpy()
    water = mx["Nuoc"].to_numpy()
    return pd.DataFrame({
        "W_B": np.divide(water, binder, out=np.full(len(mx), np.inf), where=binder > 0),
        "W_powder": np.divide(water, powder, out=np.full(len(mx), np.inf), where=powder > 0),
        "absolute_volume": (mx.to_numpy() / dens).sum(axis=1),
        "mass_kg_m3": mx.sum(axis=1).to_numpy(),
        "steel_vol_fraction": mx["Soi_thep"].to_numpy() / densities["Soi_thep"],
    }, index=mx.index)


def predict(model: Any, mix: dict[str, float], age: float, fixed: dict[str, float], feature_cols: list[str]) -> float:
    check_model_columns(model, feature_cols)
    df = pd.DataFrame([mix], columns=MATERIALS)
    x = feature_frame(df, age, fixed, feature_cols)
    result = np.asarray(model.predict(x), dtype=float).reshape(-1)
    if result.size != 1 or not np.isfinite(result[0]):
        raise ValueError("Mô hình không trả về cường độ nén hữu hạn.")
    return float(result[0])


def batch_table(mix: dict[str, float], mode: str, amount: float) -> tuple[pd.DataFrame, float]:
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError("Khối lượng/thể tích mẻ trộn phải lớn hơn 0.")
    total_mass = sum(float(mix[k]) for k in MATERIALS)
    if total_mass <= 0:
        raise ValueError("Tổng khối lượng vật liệu phải lớn hơn 0.")
    if mode == "kg":
        coefficient = float(amount) / total_mass
    elif mode == "m3":
        coefficient = float(amount)
    else:
        raise ValueError("Đơn vị mẻ trộn không hợp lệ.")
    table = pd.DataFrame({
        "Vật liệu": [LABELS[k] for k in MATERIALS],
        "Định lượng (kg/m³)": [float(mix[k]) for k in MATERIALS],
        "Khối lượng mẻ (kg)": [float(mix[k]) * coefficient for k in MATERIALS],
    })
    return table, coefficient


def training_bounds_from_df(training: pd.DataFrame, feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    missing = set(feature_cols) - set(training.columns)
    if missing:
        raise ValueError("Dữ liệu đặc trưng huấn luyện thiếu: " + ", ".join(sorted(missing)))
    df = training.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError("Không có dữ liệu huấn luyện hợp lệ để xác định phạm vi.")
    bounds = {}
    for k in feature_cols:
        lo, hi = float(df[k].min()), float(df[k].max())
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Giới hạn {k} không hợp lệ")
        bounds[k] = (lo, hi)
    return bounds


def bound_warnings(features: pd.DataFrame, training_bounds: dict[str, tuple[float, float]]) -> list[str]:
    warnings = []
    for k, (lo, hi) in training_bounds.items():
        if k not in features:
            continue
        values = features[k].to_numpy(dtype=float)
        if np.any(values < lo - 1e-8) or np.any(values > hi + 1e-8):
            warnings.append(f"{k} nằm ngoài dải dữ liệu huấn luyện [{lo:.3g}, {hi:.3g}].")
    return warnings


def optimize_mix(
    model: Any,
    feature_cols: list[str],
    training_bounds: dict[str, tuple[float, float]],
    age: float,
    fixed: dict[str, float],
    densities: dict[str, float],
    rules: DesignRules,
    target_mpa: float,
    max_cement: float,
    mode: str = "Nhanh",
    seed: int = 42,
    training: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Tối ưu DE theo lô với tổ hợp chất liên kết cố định và kiểm tra hậu tối ưu.

    Trả về ứng viên hợp lệ có xi măng thấp nhất nếu đã gặp trong khi tìm kiếm;
    không xác nhận cường độ thực nghiệm hay khả năng thi công.
    """
    check_model_columns(model, feature_cols)
    rules.validate()
    if not (np.isfinite(target_mpa) and target_mpa > 0):
        raise ValueError("Cường độ mục tiêu phải dương.")
    if not (np.isfinite(age) and age > 0):
        raise ValueError("Tuổi dưỡng hộ phải dương.")
    if not np.isfinite(max_cement) or max_cement <= 0:
        raise ValueError("Giới hạn xi măng phải dương.")
    if mode not in {"Nhanh", "Chuyên sâu"}:
        raise ValueError("Chế độ tối ưu không hợp lệ.")
    missing = set(FEATURES_15) - set(training_bounds)
    if missing:
        raise ValueError("Chưa có giới hạn huấn luyện cho: " + ", ".join(sorted(missing)))
    for k in ["Tuoi_duong_ho", *DEFAULT_FIXED]:
        val = age if k == "Tuoi_duong_ho" else fixed[k]
        lo, hi = training_bounds[k]
        if not lo - 1e-8 <= val <= hi + 1e-8:
            raise ValueError(f"{k}={val:g} nằm ngoài miền huấn luyện [{lo:g}, {hi:g}]; không tối ưu ngoại suy.")

    # Nếu không có dữ liệu train, không được giả định tổ hợp chưa thấy là đã được học.
    support_count = None
    if training is not None:
        support_count = count_training_matches(training, rules)
        if support_count == 0:
            raise ValueError(
                "Không có mẫu huấn luyện nào khớp tổ hợp chất liên kết và ngưỡng tối thiểu "
                "đang chọn. Hãy chọn tổ hợp có dữ liệu thực, hoặc bổ sung dữ liệu và huấn luyện lại; "
                "không tối ưu ngoài phân bố này."
            )
    elif rules.binder_mode != "BOTH" or rules.silica_mode != "optional" or rules.quartz_mode != "optional":
        raise ValueError(
            "Cần assets/training_features.csv để xác minh dữ liệu cho tổ hợp chất liên kết "
            "đã chọn. Hiện chỉ giữ chế độ mặc định khi chưa có tệp này."
        )

    bds = material_search_bounds(training_bounds, rules, max_cement)
    maxiter, popsize = (25, 7) if mode == "Nhanh" else (90, 12)
    dens_arr = np.array([densities[k] for k in MATERIALS], dtype=float)
    if not np.isfinite(dens_arr).all() or np.any(dens_arr <= 0):
        raise ValueError("Dữ liệu khối lượng riêng không hợp lệ.")
    target_volume = 1.0 - rules.air_fraction
    index = {k: i for i, k in enumerate(MATERIALS)}
    policies = material_policies(rules)
    total_call = [0]
    best_feasible = {"cement": float("inf"), "x": None, "count": 0}
    training_keys = list(training_bounds)

    def outside(y: np.ndarray, low: float, high: float) -> np.ndarray:
        return np.maximum(low - y, 0.0) + np.maximum(y - high, 0.0)

    def objective(a: np.ndarray) -> np.ndarray:
        # Scipy vectorized=True truyền (N, S); chuyển về (S, N).
        mix_arr = np.atleast_2d(a.T)
        df = pd.DataFrame(mix_arr, columns=MATERIALS)
        features_frame = feature_frame(df, age, fixed, feature_cols)
        predictions = np.asarray(model.predict(features_frame), dtype=float).reshape(-1)
        if len(predictions) != len(df) or not np.isfinite(predictions).all():
            raise ValueError("Mô hình trả về dự đoán không hợp lệ.")
        quantities = physical_quantities(df, densities)
        shortage = np.maximum(target_mpa + rules.margin_mpa - predictions, 0.0)
        wb = quantities["W_B"].to_numpy()
        vol = quantities["absolute_volume"].to_numpy()
        model_r = quantities["W_powder"].to_numpy()
        lo_wb, hi_wb = training_bounds["Ty_le_W_B"]
        loss = (
            mix_arr[:, index["Xi_mang"]]
            + 1.0e4 * shortage ** 2
            + 1.0e6 * outside(wb, rules.wb_min, rules.wb_max) ** 2
            + 1.0e6 * outside(
                vol, target_volume - rules.volume_tolerance,
                target_volume + rules.volume_tolerance
            ) ** 2
            + 1.0e6 * outside(model_r, lo_wb, hi_wb) ** 2
        )
        in_bounds = np.ones(len(df), dtype=bool)
        for k in training_keys:
            low, high = training_bounds[k]
            v = features_frame[k].to_numpy()
            in_bounds &= (v >= low - 1e-8) & (v <= high + 1e-8)
        feasible = (
            (shortage <= 1e-7)
            & (outside(wb, rules.wb_min, rules.wb_max) <= 1e-7)
            & (abs(vol - target_volume) <= rules.volume_tolerance + 1e-7)
            & in_bounds
        )
        total_call[0] += len(df)
        best_feasible["count"] += int(feasible.sum())
        if feasible.any():
            choices = np.flatnonzero(feasible)
            lowest = choices[np.argmin(mix_arr[choices, index["Xi_mang"]])]
            cement = mix_arr[lowest, index["Xi_mang"]]
            if cement < best_feasible["cement"]:
                best_feasible["cement"] = float(cement)
                best_feasible["x"] = mix_arr[lowest].copy()
        return loss

    result = differential_evolution(
        objective, bounds=bds, maxiter=maxiter, popsize=popsize,
        seed=seed, polish=False, vectorized=True, updating="deferred",
        tol=0.01, workers=1,
    )
    selected_feasible = best_feasible["x"] is not None
    selected = best_feasible["x"] if selected_feasible else result.x
    proposal = dict(zip(MATERIALS, map(float, selected)))
    predicted = predict(model, proposal, age, fixed, feature_cols)
    measures = physical_quantities(pd.DataFrame([proposal]), densities).iloc[0]
    extra = bound_warnings(
        feature_frame(pd.DataFrame([proposal]), age, fixed, feature_cols), training_bounds
    )
    physical_checks = {
        "Cường độ AI ≥ mục tiêu + biên dự phòng": predicted + 1e-7 >= target_mpa + rules.margin_mpa,
        "W/B trong giới hạn": rules.wb_min - 1e-7 <= measures["W_B"] <= rules.wb_max + 1e-7,
        "Thể tích tuyệt đối hợp lệ": abs(measures["absolute_volume"] - target_volume) <= rules.volume_tolerance + 1e-7,
        "Không vượt miền từng đặc trưng": not extra,
    }
    for k, policy in policies.items():
        if policy == "off":
            ok = abs(proposal[k]) <= 1e-8
            label = f"{LABELS[k]} = 0 (không sử dụng)"
        elif policy == "required":
            ok = proposal[k] >= material_minimum(rules, k) - 1e-7
            label = f"{LABELS[k]} ≥ {material_minimum(rules, k):g} kg/m³"
        else:
            ok = proposal[k] >= -1e-8
            label = f"{LABELS[k]} tùy chọn"
        physical_checks[label] = ok
    warnings = list(extra)
    if support_count is not None and support_count < 10:
        warnings.append(
            f"Chỉ có {support_count} mẫu huấn luyện khớp tổ hợp đang chọn; "
            "độ tin cậy của tối ưu có thể thấp."
        )
    if proposal["Tro_bay"] == 0:
        warnings.append("Mô hình vẫn nhận các đặc trưng hóa lý tro bay cố định kể cả khi FA = 0; kiểm chứng ngoại suy là cần thiết.")
    if proposal["Xil_lo_cao"] == 0:
        warnings.append("Mô hình vẫn nhận đặc trưng mác GGBFS cố định kể cả khi GGBFS = 0; kiểm chứng ngoại suy là cần thiết.")
    return {
        "feasible": bool(all(physical_checks.values())),
        "mix": proposal, "prediction": predicted,
        "age": age, "target": target_mpa,
        "checks": physical_checks, "warnings": warnings,
        "quantities": measures.to_dict(),
        "iterations": result.nit, "evaluations": total_call[0],
        "optimizer_success": bool(result.success),
        "note": result.message,
        "binder_mode": rules.binder_mode,
        "binder_label": BINDER_CHOICES[rules.binder_mode],
        "material_policies": policies,
        "training_support_count": support_count,
        "feasible_candidates_seen": best_feasible["count"],
        "selected_from_feasible_candidates": selected_feasible,
    }
