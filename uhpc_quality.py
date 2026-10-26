"""Independent, auditable engineering and ML guards for UHPC Studio.

This module never fabricates mixtures, strengths or calibration statistics.
Inputs and material densities are research assumptions, not a design standard.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from uhpc_core import MATERIALS, FEATURES_15, LABELS, DEFAULT_DENSITIES, DEFAULT_FIXED

BINDER = ("Xi_mang", "Tro_bay", "Xil_lo_cao", "Silica_fume")
POWDER = (*BINDER, "Bot_thach_anh")
OTHER = ("Tuoi_duong_ho", "Ty_le_W_B", *DEFAULT_FIXED)
STRENGTH_NAMES = ("Cuong_do_nen_MPa", "Cuong_do_nen", "compressive_strength", "strength_mpa")


class AuditError(ValueError):
    pass


def check_mix(mix: Mapping[str, float]) -> dict[str, float]:
    missing = [k for k in MATERIALS if k not in mix]
    if missing:
        raise AuditError("Thiếu vật liệu: " + ", ".join(missing))
    v = {k: float(mix[k]) for k in MATERIALS}
    if not all(np.isfinite(x) and x >= 0 for x in v.values()):
        raise AuditError("Định lượng kg/m³ phải hữu hạn và không âm.")
    if sum(v.values()) <= 0 or sum(v[k] for k in BINDER) <= 0:
        raise AuditError("Tổng vật liệu và lượng chất kết dính phải lớn hơn 0.")
    return v


def physical(mix: Mapping[str, float], densities: Mapping[str, float], air: float = 0) -> dict[str, float]:
    v = check_mix(mix)
    if not 0 <= air < 0.2:
        raise AuditError("Hàm lượng khí dự kiến không hợp lệ.")
    rho = {k: float(densities[k]) for k in MATERIALS}
    if any(not np.isfinite(x) or x <= 0 for x in rho.values()):
        raise AuditError("Khối lượng riêng phải là số dương hữu hạn.")
    b = sum(v[k] for k in BINDER)
    p = sum(v[k] for k in POWDER)
    return {
        "W_B": v["Nuoc"] / b,
        "W_P_notebook": v["Nuoc"] / p,
        "absolute_volume": sum(v[k] / rho[k] for k in MATERIALS),
        "volume_residual": 1.0 - air - sum(v[k] / rho[k] for k in MATERIALS),
        "steel_vol_percent": 100 * v["Soi_thep"] / rho["Soi_thep"],
        "mass_kg_m3": sum(v.values()),
        "equiv_density_kg_m3": sum(v.values()) / max(1e-9, sum(v[k] / rho[k] for k in MATERIALS) + air),
    }


def verify_design(mix, densities, *, wb_min=.14, wb_max=.24,
                  air=.02, volume_tol=.02, steel_limit_percent=2.,
                  require_fa=True, require_ggbfs=True, require_quartz=False,
                  cement_max=None, target=None, predicted=None, margin=0):
    q = physical(mix, densities, air)
    checks = {
        "W/B trong giới hạn": wb_min <= q["W_B"] <= wb_max,
        "Thể tích tuyệt đối (có xét khí dự kiến)": abs(q["volume_residual"]) <= volume_tol,
        "Sợi thép trong giới hạn % thể tích": q["steel_vol_percent"] <= steel_limit_percent + 1e-9,
        "Có tro bay (FA)": not require_fa or mix["Tro_bay"] > 0,
        "Có GGBFS": not require_ggbfs or mix["Xil_lo_cao"] > 0,
        "Có bột thạch anh": not require_quartz or mix["Bot_thach_anh"] > 0,
        "Xi măng không vượt giới hạn": cement_max is None or mix["Xi_mang"] <= cement_max + 1e-9,
    }
    if target is not None:
        checks["Dự đoán AI ≥ mục tiêu + dự phòng"] = predicted is not None and np.isfinite(predicted) and predicted + 1e-8 >= target + margin
    return q, checks


def batch_table(mix, batch_mode, amount):
    v = check_mix(mix)
    if not np.isfinite(amount) or amount <= 0:
        raise AuditError("Quy mô mẻ phải dương.")
    total = sum(v.values())
    if batch_mode == "kg":
        factor = amount / total
    elif batch_mode == "m³":
        factor = amount
    else:
        raise AuditError("Đơn vị mẻ trộn không hợp lệ.")
    table = pd.DataFrame({
        "Vật liệu": [LABELS.get(k,k) for k in MATERIALS],
        "kg/m³": [v[k] for k in MATERIALS],
        "Mẻ trộn (kg)": [v[k]*factor for k in MATERIALS],
        "Tỷ lệ khối lượng (%)": [v[k]/total*100 for k in MATERIALS],
    })
    return table


def feature_frame(mixes: pd.DataFrame, age: float, fixed: Mapping[str,float], columns=FEATURES_15):
    missing = set(MATERIALS) - set(mixes)
    if missing:
        raise AuditError("Thiếu cột dữ liệu: "+", ".join(sorted(missing)))
    a = mixes.loc[:, MATERIALS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(a.to_numpy(dtype=float)).all() or (a < 0).any().any():
        raise AuditError("Tồn tại giá trị vật liệu không hợp lệ.")
    if not np.isfinite(age) or age <= 0:
        raise AuditError("Tuổi nén không hợp lệ.")
    denom = a[list(POWDER)].sum(axis=1)
    if (denom <= 0).any():
        raise AuditError("Tổng bột phải lớn hơn 0.")
    out = a.copy()
    out["Tuoi_duong_ho"] = age
    out["Ty_le_W_B"] = a["Nuoc"] / denom  # exact notebook definition, NOT conventional W/B
    for k in DEFAULT_FIXED:
        val = float(fixed[k])
        if not np.isfinite(val):
            raise AuditError(f"{k} không hợp lệ.")
        out[k] = val
    if len(columns) != 15 or len(set(columns)) != 15 or set(columns) != set(FEATURES_15):
        raise AuditError("Bộ đặc trưng của mô hình khác 15 cột gốc.")
    return out.loc[:,list(columns)]


def load_bundle(path: str | Path):
    """Only read pickle already deployed by the repo owner; never unpickle user upload."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        model = obj.get("model")
        if model is None: model = obj.get("best_model")
        cols = obj.get("feature_cols")
        if cols is None: cols = obj.get("feature_names")
        meta = {k:obj.get(k) for k in ("validation", "metrics", "runtime_versions") if k in obj}
    else:
        model, cols, meta = obj, None, {}
    if model is None or not callable(getattr(model,"predict",None)):
        raise AuditError("Tệp .pkl không chứa mô hình dự đoán hợp lệ.")
    if cols is None:
        cols = getattr(model, "feature_name_", None)
    if cols is None:
        cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        # Legacy model; must match exact 15 input features.
        cols = list(FEATURES_15)
    cols = list(cols)
    if set(cols) != set(FEATURES_15) or len(cols) != 15:
        raise AuditError("Tên đặc trưng mô hình không phù hợp 15 cột UHPC.")
    names = getattr(model,"feature_name_",None)
    if names is None:
        names = getattr(model,"feature_names_in_",None)
    if names is not None and list(names) != cols:
        raise AuditError("Thứ tự đặc trưng đã lưu không trùng mô hình.")
    if getattr(model,"n_features_in_",15) != 15:
        raise AuditError("Số đặc trưng mô hình không phải 15.")
    return model, cols, meta


def load_training(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(FEATURES_15) - set(df.columns)
    if missing:
        raise AuditError("CSV train thiếu: "+", ".join(sorted(missing)))
    for k in FEATURES_15:
        df[k] = pd.to_numeric(df[k],errors="coerce")
    df = df.dropna(subset=FEATURES_15).reset_index(drop=True)
    df = df[np.isfinite(df[FEATURES_15]).all(axis=1)]
    df = df[(df[MATERIALS] >= 0).all(axis=1)]
    if df.empty:
        raise AuditError("Không có mẫu huấn luyện hợp lệ.")
    return df.reset_index(drop=True)


def predict(model, cols, mix, age, fixed):
    x = feature_frame(pd.DataFrame([check_mix(mix)]), age, fixed, cols)
    y = np.asarray(model.predict(x),dtype=float).reshape(-1)
    if len(y)!=1 or not np.isfinite(y[0]):
        raise AuditError("Mô hình không trả về cường độ hợp lệ.")
    return float(y[0])


@dataclass
class Proximity:
    near_distance: float
    train_p95: float
    close: bool
    outside_features: list[str]
    neighbors_count: int
    age_matches: int


def distance_reference(train: pd.DataFrame):
    """Reference: robust-scaled L2 leave-one-out nearest distance on unique train rows.
    P95 is a proximity diagnostic only, NEVER 95% prediction certainty.
    """
    data = train[FEATURES_15].drop_duplicates().astype(float)
    if len(data)<3:
        return None
    lo = data.quantile(.05).to_numpy(dtype=float)
    hi = data.quantile(.95).to_numpy(dtype=float)
    scale = np.where(hi-lo > 1e-9, hi-lo, 1.0)
    z = (data.to_numpy(dtype=float) - lo)/scale
    knn = NearestNeighbors(n_neighbors=2).fit(z)
    dist = knn.kneighbors(z, return_distance=True)[0][:,1]
    return (lo, scale, knn, float(np.quantile(dist,.95)))


def distance_check(features: pd.DataFrame, train: pd.DataFrame, reference=None):
    missing = set(FEATURES_15)-set(features)
    if missing: raise AuditError("Thiếu đặc trưng để kiểm tra ngoại suy.")
    minvals, maxvals = train[FEATURES_15].min(), train[FEATURES_15].max()
    outside = [k for k in FEATURES_15 if (features[k].lt(minvals[k]-1e-9) | features[k].gt(maxvals[k]+1e-9)).any()]
    age_matches = int(np.sum(np.isclose(train["Tuoi_duong_ho"], float(features["Tuoi_duong_ho"].iloc[0]))))
    ref = reference if reference is not None else distance_reference(train)
    if ref is None:
        return Proximity(float("nan"),float("nan"),False,outside,len(train),age_matches)
    lo, scale, knn, p95 = ref
    z = (features.loc[:,FEATURES_15].to_numpy(dtype=float)-lo)/scale
    d = knn.kneighbors(z, n_neighbors=1,return_distance=True)[0][:,0]
    return Proximity(float(d.max()),p95,bool(not outside and np.all(d<=p95)),outside,len(train),age_matches)


def cost_table(mix: Mapping[str,float], prices:Mapping[str,float], *, mode="m³", amount=1.0):
    v = check_mix(mix)
    if mode not in ("kg","m³") or amount<=0:
        raise AuditError("Quy mô dự toán không hợp lệ.")
    coeff = amount / sum(v.values()) if mode == "kg" else amount
    missing = [k for k in MATERIALS if v[k]>0 and (k not in prices or prices[k] is None or not np.isfinite(prices[k]) or prices[k]<0)]
    if missing:
        raise AuditError("Thiếu/không hợp lệ đơn giá VND/kg: "+", ".join(LABELS.get(k,k) for k in missing))
    rows=[]
    for k in MATERIALS:
        supplied = prices.get(k)
        price = float(supplied) if supplied is not None and np.isfinite(supplied) else 0.0
        rows.append({"Vật liệu":LABELS.get(k,k),"Khối lượng (kg)":v[k]*coeff,"Đơn giá (VND/kg)":price,"Thành tiền (VND)":v[k]*coeff*price})
    return pd.DataFrame(rows)


def evaluate_validation(model, cols, val:pd.DataFrame, train:pd.DataFrame | None = None):
    true_col = next((k for k in STRENGTH_NAMES if k in val),None)
    if not true_col:
        raise AuditError("CSV kiểm định thiếu cột cường độ đo được: "+", ".join(STRENGTH_NAMES))
    if not set(FEATURES_15).issubset(val.columns):
        raise AuditError("CSV kiểm định phải có đủ 15 đặc trưng đầu vào.")
    x = val[cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(val[true_col], errors="coerce")
    mask = x.notna().all(axis=1)&np.isfinite(x).all(axis=1)&y.notna()&np.isfinite(y)
    if int(mask.sum()) < 3:
        raise AuditError("Cần ít nhất 3 dòng kiểm định có nhãn cường độ thật.")
    x, y = x.loc[mask], y.loc[mask]
    pred = np.asarray(model.predict(x),dtype=float).reshape(-1)
    if len(pred)!=len(x) or not np.isfinite(pred).all():
        raise AuditError("Không thể đánh giá mô hình trên CSV kiểm định.")
    results = pd.DataFrame({"Đo thực nghiệm (MPa)":y.to_numpy(),"AI dự báo (MPa)":pred,"Sai số (MPa)":pred-y.to_numpy(),"Tuổi nén (ngày)":x["Tuoi_duong_ho"].to_numpy()})
    rmse = float(np.sqrt(mean_squared_error(y,pred)))
    metrics={"n":len(results), "MAE":float(mean_absolute_error(y,pred)),"RMSE":rmse,"R²":float(r2_score(y,pred))}
    if train is not None and set(FEATURES_15).issubset(train.columns):
        # Exact matches identify certain reuse; absence does NOT prove independence.
        merged=x.loc[:,FEATURES_15].merge(train[FEATURES_15].drop_duplicates(),on=FEATURES_15,how="left",indicator=True)
        metrics["exact_train_overlap"] = int((merged["_merge"]=="both").sum())
    return metrics, results


def strength_target_interval(target: float, mode="min_close", tolerance=0.01, margin=0.0):
    """Accepted AI *point-prediction* band, not a physical strength confidence interval.

    min_close: target+margin <= prediction <= target+margin+tolerance
    band: target-tolerance <= prediction <= target+tolerance (may be below target)
    minimum_only: prediction >= target+margin
    """
    values=(target,tolerance,margin)
    if not all(np.isfinite(float(v)) for v in values) or target <= 0 or tolerance < 0 or margin < 0:
        raise AuditError("Mục tiêu, dung sai và biên dự phòng phải hợp lệ.")
    if mode == "min_close":
        return float(target+margin),float(target+margin+tolerance)
    if mode == "band":
        if target-tolerance < 0:
            raise AuditError("Dung sai ± phải nhỏ hơn hoặc bằng cường độ mục tiêu.")
        return float(target-tolerance),float(target+tolerance)
    if mode == "minimum_only":
        return float(target+margin),float("inf")
    raise AuditError("Chế độ bám sát cường độ không hợp lệ.")


def strength_alignment(prediction, target, *, mode="min_close", tolerance=0.01, margin=0.0):
    """Report prediction-vs-command only: not model accuracy vs lab observation."""
    low,high=strength_target_interval(target,mode,tolerance,margin)
    if not np.isfinite(prediction):
        raise AuditError("Cường độ AI dự đoán không hợp lệ.")
    delta=float(prediction-target)
    return {
        "target_mpa":float(target),"predicted_mpa":float(prediction),
        "delta_mpa":delta,"absolute_delta_mpa":abs(delta),
        "absolute_delta_percent":abs(delta)/float(target)*100,
        "same_to_2_decimals": bool(round(float(prediction),2)==round(float(target),2)),
        "exact_numeric_match":bool(abs(delta)<=1e-10),
        "lower_mpa":low,"upper_mpa":high if np.isfinite(high) else None,
        "meets_policy":bool(low-1e-8<=prediction<=high+1e-8),
        "meets_minimum":bool(prediction+1e-8>=target),
        "mode":mode,"tolerance_mpa":float(tolerance),"margin_mpa":float(margin),
    }


def validation_near_target(results:pd.DataFrame,target:float,age:float,window:float=10.0, tolerance:float=2.0):
    """Real-label validation restricted by measured strength and specified age.

    Merely a descriptive slice of uploaded data, never an individual prediction bound.
    """
    if not all(np.isfinite(float(x)) for x in (target,age,window,tolerance)) or target <= 0 or age <= 0 or window < 0 or tolerance < 0:
        raise AuditError("Thông số vùng kiểm định không hợp lệ.")
    measured=results["Đo thực nghiệm (MPa)"].astype(float).to_numpy()
    yhat=results["AI dự báo (MPa)"].astype(float).to_numpy()
    ages=results["Tuổi nén (ngày)"].astype(float).to_numpy()
    good=np.isfinite(measured)&np.isfinite(yhat)&np.isfinite(ages)&np.isclose(ages,age)&(np.abs(measured-target)<=window+1e-9)
    actual,predicted=measured[good],yhat[good]
    if len(actual)==0:return None
    diff=predicted-actual
    return {"n":int(len(actual)),"mae_mpa":float(np.mean(np.abs(diff))),"rmse_mpa":float(np.sqrt(np.mean(diff**2))),
            "bias_mpa":float(np.mean(diff)),"within_tolerance_count":int(np.sum(np.abs(diff)<=tolerance+1e-8)),
            "within_tolerance_percent":float(np.mean(np.abs(diff)<=tolerance+1e-8)*100),
            "target_mpa":float(target),"age_days":float(age),"window_mpa":float(window),"tolerance_mpa":float(tolerance)}


def request_id(payload):
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,default=float).encode()).hexdigest()


def optimize(model,cols,train,age,fixed,densities,target, *, wb_min=.14,wb_max=.24,
             air=.02, volume_tol=.02, steel_limit_percent=2.0, require_fa=True,
             require_ggbfs=True, require_quartz=False, cement_max=1000,
             min_present_kg=5., margin=0., use_distance=True, objective_mode="cement",
             target_mode="min_close", target_tolerance=0.01,
             prices=None, seed=42,maxiter=38,popsize=7):
    """Numerical search; valid numerical candidate != experimentally verified UHPC.

    DE searches inside per-material training quantile ranges, and chooses only
    candidates passing every final constraint. Returns infeasible status if none.
    """
    if not wb_min>0 or not wb_max>=wb_min or cement_max<=0:
        raise AuditError("Giới hạn thiết kế không hợp lệ.")
    target_lower,target_upper=strength_target_interval(target,target_mode,target_tolerance,margin)
    if objective_mode not in ("cement","cost","near_target"):
        raise AuditError("Mục tiêu tối ưu không hợp lệ.")
    sub=train[np.isclose(train["Tuoi_duong_ho"],age)]
    if sub.empty:
        raise AuditError("Không có dữ liệu ở đúng tuổi nén được chọn; không tối ưu ngoại suy tuổi.")
    for k,required in [("Tro_bay",require_fa),("Xil_lo_cao",require_ggbfs),("Bot_thach_anh",require_quartz)]:
        if required:
            sub=sub[sub[k]>=min_present_kg]
    if len(sub)<5:
        raise AuditError(f"Chỉ còn {len(sub)} mẫu khớp tuổi và tổ hợp vật liệu; cần ít nhất 5 mẫu để tối ưu.")
    for k in DEFAULT_FIXED:
        f = float(fixed[k])
        if f < train[k].min()-1e-9 or f>train[k].max()+1e-9:
            raise AuditError(f"{k} ngoài phạm vi mô hình, không chạy tối ưu.")
    low = sub[MATERIALS].quantile(.05)
    high = sub[MATERIALS].quantile(.95)
    limits=[]
    for k in MATERIALS:
        l,h=max(0,float(low[k])),float(high[k])
        if k=="Xi_mang": l=max(l,1.); h=min(h,cement_max)
        if k=="Soi_thep": h=min(h, float(densities[k])*steel_limit_percent/100.)
        if k in ["Tro_bay","Xil_lo_cao","Bot_thach_anh"]:
            required = {"Tro_bay":require_fa,"Xil_lo_cao":require_ggbfs,"Bot_thach_anh":require_quartz}[k]
            if required: l=max(l,min_present_kg)
        if h < l: raise AuditError(f"Không có miền khả thi cho {LABELS.get(k,k)}.")
        limits.append((l,h))
    # Fixed zero-width dimensions are removed to support SciPy all versions.
    free=[i for i,(a,b) in enumerate(limits) if b>a+1e-9]
    if not free:raise AuditError("Không có biến vật liệu nào có thể tối ưu.")
    if objective_mode=="cost":
        if prices is None:raise AuditError("Chưa cung cấp đơn giá vật liệu.")
        possible = {k:float(prices.get(k,np.nan)) for k in MATERIALS}
        bad = [k for i,k in enumerate(MATERIALS) if limits[i][1]>0 and not np.isfinite(possible[k])]
        if bad: raise AuditError("Thiếu giá để tối ưu: "+", ".join(bad))
        if any(val<0 for val in possible.values() if np.isfinite(val)):
            raise AuditError("Đơn giá không được âm.")
        # Price for an impossible/zero-mass component cannot contaminate vector dot products.
        prices = {k:(float(possible[k]) if np.isfinite(possible[k]) else 0.) for k in MATERIALS}
        baseline_cost=sum(float(sub[k].median())*prices[k] for k in MATERIALS)
        if baseline_cost<=0:
            raise AuditError("Không thể tối ưu chi phí khi tổng chi phí tham chiếu bằng 0.")
    ref = distance_reference(train)
    group_ref = distance_reference(sub)
    if use_distance and (ref is None or group_ref is None):
        raise AuditError("Chưa đủ mẫu train phân biệt để kiểm tra khoảng cách toàn tập và nhóm cùng điều kiện.")
    train_lo=train[FEATURES_15].min().to_numpy(dtype=float)
    train_hi=train[FEATURES_15].max().to_numpy(dtype=float)
    den=np.asarray([densities[k] for k in MATERIALS],dtype=float)
    if not np.isfinite(den).all() or (den<=0).any():raise AuditError("Khối lượng riêng không hợp lệ.")
    # Three candidate tiers: passes everything; passes safety/physics but misses
    # strength target; lowest objective (may violate physical constraints).
    # A missing exact match never authorizes fabricating an exact prediction.
    best = {"loss":float("inf"),"mix":None,"prediction":None,"distance":None,"origin":None}
    best_safe={"loss":float("inf"),"mix":None,"prediction":None,"distance":None,"origin":None}
    best_raw={"loss":float("inf"),"mix":None,"prediction":None,"distance":None,"origin":None}
    eval_count=[0]
    source=["Tối ưu số"]
    def expand(a):
        z=np.asarray(a,dtype=float)
        if z.ndim==1:z=z.reshape(-1,1)
        out=np.tile(np.array([lim[0] for lim in limits]),(z.shape[1],1))
        out[:,free]=z.T
        return out
    def objective(a):
        samples=expand(a)
        n=len(samples)
        f=feature_frame(pd.DataFrame(samples,columns=MATERIALS),age,fixed,cols)
        predicted=np.asarray(model.predict(f),dtype=float).reshape(-1)
        if len(predicted)!=n or not np.isfinite(predicted).all():raise AuditError("Model trả kết quả tối ưu không hợp lệ.")
        v=samples/den
        volume=v.sum(axis=1)
        binder=samples[:,[MATERIALS.index(k) for k in BINDER]].sum(axis=1)
        wb=samples[:,MATERIALS.index("Nuoc")]/np.maximum(1e-9,binder)
        steel=100*samples[:,MATERIALS.index("Soi_thep")]/den[MATERIALS.index("Soi_thep")]
        shortfall=np.maximum(target_lower-predicted,0)
        overrun=np.maximum(predicted-target_upper,0)
        # Reject both under-strength AND overshooting the selected strength band.
        # Linear + quadratic terms discourage near-boundary violations.
        strength_violation=shortfall+overrun
        penalty=2e5*strength_violation+1e6*strength_violation**2
        penalty+=1e7*(np.maximum(wb_min-wb,0)**2+np.maximum(wb-wb_max,0)**2)
        penalty+=1e7*np.maximum(np.abs(volume-(1-air))-volume_tol,0)**2
        penalty+=1e5*np.maximum(steel-steel_limit_percent,0)**2
        farr=f.loc[:,FEATURES_15].to_numpy(dtype=float)
        outside=np.maximum(train_lo-farr,0)+np.maximum(farr-train_hi,0)
        denom=np.maximum(train_hi-train_lo,1.0)
        penalty+=1e5*np.sum((outside/denom)**2,axis=1)
        if ref:
            rlo, rscale, knn, p95=ref
            dist=knn.kneighbors((farr-rlo)/rscale,n_neighbors=1,return_distance=True)[0][:,0]
            if use_distance:
                penalty+=1e5*np.maximum(dist-p95,0)**2
        else: dist=np.full(n,np.nan)
        if group_ref:
            glo, gscale, gknn, gp95=group_ref
            group_dist=gknn.kneighbors((farr-glo)/gscale,n_neighbors=1,return_distance=True)[0][:,0]
            if use_distance:
                penalty+=1e5*np.maximum(group_dist-gp95,0)**2
        else: group_dist=np.full(n,np.nan)
        if objective_mode=="cost":
            raw=samples @ np.array([prices[k] for k in MATERIALS],dtype=float)
            # Fixed normalizer: optimization objective must NOT depend on current population.
            raw=raw/baseline_cost * 500
        elif objective_mode=="near_target":
            # Predicted strength deviation dominates cement minimization by design.
            # Material saving is only a small tie-breaker, not a second main goal.
            raw=np.abs(predicted-target)*1000 + samples[:,0]*.0001
        else: raw=samples[:,0]+samples[:,MATERIALS.index("Soi_thep")]*.25
        loss=raw+penalty
        # Explicitly require every suggested candidate to remain inside the
        # numerical search bounds, including when evaluating actual training rows.
        search_lo=np.array([b[0] for b in limits])
        search_hi=np.array([b[1] for b in limits])
        inside_search=((samples>=search_lo-1e-8)&(samples<=search_hi+1e-8)).all(axis=1)
        required_materials=(
            (samples[:,MATERIALS.index("Tro_bay")]>=min_present_kg-1e-8 if require_fa else np.ones(n,dtype=bool))
            & (samples[:,MATERIALS.index("Xil_lo_cao")]>=min_present_kg-1e-8 if require_ggbfs else np.ones(n,dtype=bool))
            & (samples[:,MATERIALS.index("Bot_thach_anh")]>=min_present_kg-1e-8 if require_quartz else np.ones(n,dtype=bool))
        )
        safe=inside_search & required_materials & (wb>=wb_min-1e-8)&(wb<=wb_max+1e-8)&(np.abs(volume-(1-air))<=volume_tol+1e-8)&(steel<=steel_limit_percent+1e-8)&(outside<=1e-8).all(axis=1)
        if use_distance:safe &= (dist<=ref[3]+1e-8)&(group_dist<=group_ref[3]+1e-8)
        valid=safe & (shortfall<=1e-8) & (overrun<=1e-8)
        if valid.any():
            inds=np.flatnonzero(valid)
            k=inds[np.argmin(raw[inds])]
            if raw[k]<best["loss"]:
                best.update(loss=float(raw[k]),mix=samples[k].copy(),prediction=float(predicted[k]),distance=float(dist[k]),origin=source[0])
        if safe.any():
            inds=np.flatnonzero(safe)
            # Even if no sample meets a very tight requested tolerance,
            # return the closest physically admissible prediction as a
            # *failed* candidate, not an unsafe low-penalty proposal.
            near_score=np.abs(predicted[inds]-target)+samples[inds,0]*1e-8
            k=inds[np.argmin(near_score)]
            score=float(near_score[np.argmin(near_score)])
            if score<best_safe["loss"]:
                best_safe.update(loss=score,mix=samples[k].copy(),prediction=float(predicted[k]),distance=float(dist[k]),origin=source[0])
        j=int(np.argmin(loss))
        if loss[j]<best_raw["loss"]:
            best_raw.update(loss=float(loss[j]),mix=samples[j].copy(),prediction=float(predicted[j]),distance=float(dist[j]),origin=source[0])
        eval_count[0]+=n
        return loss
    # Check existing rows at the selected age as candidate starting points.
    # They do NOT carry a measured strength label here; all outputs are
    # re-predicted by the loaded model and subjected to the same checks.
    prior=sub.loc[:,MATERIALS].drop_duplicates().reset_index(drop=True)
    if len(prior)>1000:
        prior=prior.sample(n=1000,random_state=int(seed)).reset_index(drop=True)
    source[0]="Dòng dữ liệu huấn luyện (dự báo lại)"
    for start in range(0,len(prior),250):
        chunk=prior.iloc[start:start+250].to_numpy(dtype=float)
        objective(chunk[:,free].T)
    source[0]="Tối ưu số"
    de = differential_evolution(objective,[limits[i] for i in free],seed=int(seed),maxiter=int(maxiter),popsize=int(popsize),tol=.01,polish=False,vectorized=True,updating="deferred")
    chosen=best if best["mix"] is not None else (best_safe if best_safe["mix"] is not None else best_raw)
    if chosen["mix"] is None:raise AuditError("Bộ giải không đánh giá được ứng viên nào.")
    mix=dict(zip(MATERIALS,map(float,chosen["mix"])))
    pred=predict(model,cols,mix,age,fixed)
    phys, checks=verify_design(mix,densities,wb_min=wb_min,wb_max=wb_max,air=air,volume_tol=volume_tol,steel_limit_percent=steel_limit_percent,require_fa=require_fa,require_ggbfs=require_ggbfs,require_quartz=require_quartz,cement_max=cement_max,target=None)
    alignment=strength_alignment(pred,target,mode=target_mode,tolerance=target_tolerance,margin=margin)
    checks["Đáp ứng khoảng cường độ AI đã chọn"]=alignment["meets_policy"]
    if target_mode in ("min_close","minimum_only"):
        # Retain the original diagnostic for older consumers and test runners.
        checks["Dự đoán AI ≥ mục tiêu + dự phòng"] = alignment["predicted_mpa"] + 1e-8 >= target + margin
    x_final=feature_frame(pd.DataFrame([mix]),age,fixed,cols)
    prox=distance_check(x_final,train,ref)
    group_prox=distance_check(x_final,sub,group_ref)
    checks["Trong min–max từng đặc trưng train"]=not bool(prox.outside_features)
    if use_distance:
        checks["Gần dữ liệu train toàn tập (P95 NN)"]=prox.close
        checks["Gần train cùng tuổi/tổ hợp (P95 NN)"]=group_prox.close
    return {"mix":mix,"prediction":pred,"physical":phys,"checks":checks,"proximity":prox,"group_proximity":group_prox,"feasible":bool(all(checks.values())),"optimizer_success":bool(de.success),"evaluations":eval_count[0],"training_matches":len(sub),"iterations":int(de.nit),"mode":objective_mode,
            "target_mode":target_mode,"target_tolerance":float(target_tolerance),"alignment":alignment,
            "candidate_origin":chosen["origin"],"candidate_tier":("in_target" if best["mix"] is not None else "safe_outside_target" if best_safe["mix"] is not None else "no_safe_candidate"),
            "match_is_model_prediction_only":True}
