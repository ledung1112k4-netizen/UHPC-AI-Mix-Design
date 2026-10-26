"""Independent, leakage-aware generalization audit for UHPC model predictions.

Real measured 28-day data are REQUIRED. Never fabricate measurements or model accuracy.
An interval computed here only has a standard conformal interpretation for data
exchangeable with the held-out calibration mixtures; OOD coverage is not guaranteed.
"""
from __future__ import annotations
from typing import Any
import math
import numpy as np
import pandas as pd

from uhpc_core import FEATURES_15, MATERIALS
from uhpc_quality import AuditError, STRENGTH_NAMES, distance_reference


def material_group_ids(frame: pd.DataFrame, decimals: int = 3) -> pd.Series:
    """Group a recipe across ages and specimen sizes using material masses.

    Equality is based on rounding to 0.001 kg/m3 to avoid float export noise;
    this is NOT a tolerant fuzzy grouping of nearly similar mix recipes.
    """
    if not set(MATERIALS).issubset(frame.columns):
        raise AuditError("Thiếu 9 thành phần vật liệu để nhóm cấp phối.")
    x = frame.loc[:, MATERIALS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise AuditError("Có khối lượng vật liệu không hợp lệ.")
    return pd.Series(list(map(tuple, np.round(x.to_numpy(dtype=float), decimals))), index=frame.index)


def conformal_quantile(scores, coverage: float = .90) -> float | None:
    """Finite-sample higher order statistic, no interpolated empirical quantile."""
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or (a < 0).any():
        return None
    if not 0 < coverage < 1:
        raise AuditError("Mức bao phủ kỳ vọng không hợp lệ.")
    k = math.ceil((len(a) + 1) * coverage)
    if k > len(a):
        return None
    return float(np.partition(a, k - 1)[k - 1])


def audit_external_validation(model: Any, cols: list[str], heldout: pd.DataFrame,
                              train: pd.DataFrame, *, age_days: float = 28,
                              target_mpa: float = 120,
                              min_calibration_groups: int = 20) -> tuple[dict, pd.DataFrame]:
    """Evaluate genuinely measured validation rows (not retraining).

    The caller is responsible for confirming the rows are from a separate,
    never-used calibration/validation set. Exact recipe overlap is detected,
    but source lineage / near duplicates cannot be proved from CSV alone.
    """
    target_col = next((c for c in STRENGTH_NAMES if c in heldout.columns), None)
    if target_col is None:
        raise AuditError("CSV chưa có cột cường độ nén thực nghiệm.")
    if not set(FEATURES_15).issubset(heldout.columns) or not set(FEATURES_15).issubset(train.columns):
        raise AuditError("Thiếu 15 đặc trưng để kiểm toán ngoại suy.")
    d = heldout.copy()
    for k in [*FEATURES_15, target_col]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES_15, target_col])
    d = d[(d[MATERIALS] >= 0).all(axis=1)]
    d = d[np.isclose(d["Tuoi_duong_ho"], age_days)].copy()
    if not len(d):
        raise AuditError(f"Không có dòng hợp lệ ở đúng {age_days:g} ngày trong CSV.")
    d["_group"] = material_group_ids(d)
    train_groups = set(material_group_ids(train))
    seen = d["_group"].isin(train_groups).to_numpy(dtype=bool)
    x = d.loc[:, cols]
    preds = np.asarray(model.predict(x), dtype=float).reshape(-1)
    if len(preds) != len(d) or not np.isfinite(preds).all():
        raise AuditError("Mô hình trả về dự báo không hợp lệ trên CSV kiểm định.")
    reference = distance_reference(train)
    if reference is not None:
        lo, scale, nn, p95 = reference
        arr = d.loc[:, FEATURES_15].to_numpy(dtype=float)
        dist = nn.kneighbors((arr - lo)/scale, n_neighbors=1, return_distance=True)[0][:, 0]
        train_low, train_high = train[FEATURES_15].min().to_numpy(), train[FEATURES_15].max().to_numpy()
        outside = np.any((arr < train_low - 1e-9) | (arr > train_high + 1e-9), axis=1)
        domain = (~outside) & (dist <= p95)
    else:
        dist = np.full(len(d), float("nan")); domain = np.zeros(len(d), bool)
    report = pd.DataFrame({
        "Đo thực nghiệm (MPa)": d[target_col].to_numpy(dtype=float),
        "AI dự báo (MPa)": preds,
        "Sai số AI - đo (MPa)": preds - d[target_col].to_numpy(dtype=float),
        "Cấp phối trùng train": seen,
        "Vùng gần train (P95 NN)": domain,
        "Khoảng cách NN": dist,
        "_group": d["_group"].to_numpy(),
    })
    report["Sai số tuyệt đối (MPa)"] = report["Sai số AI - đo (MPa)"].abs()
    unseen = report.loc[~report["Cấp phối trùng train"]].copy()
    def err_summary(table: pd.DataFrame) -> dict:
        if table.empty: return {"n":0,"unique_mixes":0,"MAE":None,"RMSE":None,"bias":None}
        e=table["Sai số AI - đo (MPa)"].to_numpy(dtype=float)
        return {"n":int(len(table)), "unique_mixes":int(table["_group"].nunique()),
                "MAE":float(np.mean(np.abs(e))),"RMSE":float(np.sqrt(np.mean(e**2))),
                "bias":float(np.mean(e))}
    high = unseen.loc[unseen["Đo thực nghiệm (MPa)"] >= target_mpa]
    outside_unseen = unseen.loc[~unseen["Vùng gần train (P95 NN)"]]
    # Within-mixture repeat rows cannot be treated as independent examples.
    # Pool conservatively with max absolute residual for each new recipe.
    max_errors = unseen.groupby("_group")["Sai số tuyệt đối (MPa)"].max().to_numpy(dtype=float)
    n_unseen = int(len(max_errors))
    q90 = conformal_quantile(max_errors) if n_unseen >= min_calibration_groups else None
    diagnostics = {
        "age_days":float(age_days), "target_mpa":float(target_mpa),
        "all":err_summary(report),"unseen":err_summary(unseen),
        "unseen_high_strength":err_summary(high),
        "unseen_outside_typical_region":err_summary(outside_unseen),
        "train_overlap_rows":int(seen.sum()),
        "train_overlap_unique_mixes":int(report.loc[seen,"_group"].nunique()),
        "n_unique_new_mixes":n_unseen,
        "calibration_q90_abs_error_mpa":q90,
        "calibration_eligible":bool(q90 is not None and n_unseen>=min_calibration_groups),
        "calibration_interpretation":"Chỉ khi CSV độc lập và cấp phối mới trao đổi được với tập hiệu chuẩn; không đảm bảo ngoại suy xa.",
        "distance_p95_not_coverage_probability":True,
        "material_group_round_decimals":3,
    }
    return diagnostics, report.drop(columns=["_group"])
