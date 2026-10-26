"""UHPC Studio — Streamlit app độc lập với Google Colab.

Mô hình dự đoán PHẢI đến từ notebook của người dùng; không dùng kết quả giả.
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from uhpc_core import (
    BINDER_CHOICES, POLICY_CHOICES, DEFAULT_DENSITIES, DEFAULT_FIXED, FEATURES_15, LABELS, MATERIALS,
    NOTEBOOK_TEST_MIX, DesignRules, batch_table, bound_warnings, mass_percentages,
    check_model_columns, count_training_matches, feature_frame, feature_importance_table, material_policies, optimize_mix, physical_quantities,
    predict, training_bounds_from_df, steel_limit_kg_m3, steel_volume_percent,
    training_similarity, training_data_audit,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATHS = [
    ROOT / "assets" / "uhpc_model.pkl",
    ROOT / "assets" / "uhpc_optimal_lightgbm_model.pkl",
    ROOT / "uhpc_optimal_lightgbm_model.pkl",
]
TRAINING_PATHS = [ROOT / "assets" / "training_features.csv", ROOT / "training_features.csv"]

st.set_page_config(page_title="UHPC Studio | Thiết kế cấp phối AI", page_icon="🧪", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px}
.hero {background: linear-gradient(110deg,#133b39,#176e63); border-radius:16px;
 padding:26px 30px; color:white; margin-bottom:20px}
.hero small {letter-spacing:.15em; color:#bbebe3; font-weight:700}
.hero h1 {font-weight:750; margin:6px 0 8px; font-size:2.15rem; color:white}
.hero p {font-size:1.02rem; max-width:850px; color:#ebf9f7; margin-bottom:0}
.stButton>button[kind="primary"] {font-weight:700; border-radius:9px}
[data-testid="stMetric"] {background:#eef5f4; padding:16px; border-radius:12px}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def read_model(path_text: str, modification: int):
    """Chỉ nạp pickle từ thư mục dự án do chính chủ sở hữu cung cấp.

    Tuyệt đối không tải file .pkl từ người lạ: pickle có thể thực thi mã tùy ý.
    """
    with open(path_text, "rb") as f:
        bundle = pickle.load(f)  # noqa: S301 — tệp local tin cậy do người dùng xuất từ Colab
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError("Tệp .pkl phải có các khóa 'model' và 'feature_cols'.")
    model = bundle["model"]
    cols = list(bundle["feature_cols"])
    check_model_columns(model, cols)
    return bundle, model, cols


@st.cache_data(show_spinner=False)
def load_training(path_text: str, modification: int) -> pd.DataFrame:
    return pd.read_csv(path_text, encoding="utf-8-sig")


def find_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.is_file()), None)


def float_display(x: float, precision: int = 3) -> str:
    return f"{x:,.{precision}f}"


def display_mix(mix: dict[str, float], mode: str, amount: float, suffix: str) -> None:
    """Hiển thị khối lượng từng vật liệu và tỷ trọng cốt liệu trên TỔNG hỗn hợp."""
    tab, _ = batch_table(mix, mode, amount)
    df = tab.copy()
    percentages = mass_percentages(mix)
    pct_aggregate = float(percentages["Cat"])
    aggregate_mass = float(df.loc[df["Vật liệu"].eq(LABELS["Cat"]), "Khối lượng mẻ (kg)"].iloc[0])
    total_kg_m3 = float(df["Định lượng (kg/m³)"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Cốt liệu / tổng khối lượng", f"{pct_aggregate:.2f}%", help="Tính từ trường Cat / tổng 9 vật liệu theo kg/m³.")
    c2.metric("Cốt liệu trong mẻ", f"{aggregate_mass:,.3f} kg")
    c3.metric("Tổng khối lượng cấp phối", f"{total_kg_m3:,.2f} kg/m³")
    st.progress(pct_aggregate / 100.0, text=f"Cát/cốt liệu chiếm {pct_aggregate:.2f}% tổng khối lượng vật liệu")
    st.caption("% khối lượng = khối lượng vật liệu (kg/m³) ÷ tổng khối lượng của 9 vật liệu (kg/m³) × 100. "
               "Cốt liệu = biến `Cat` theo dữ liệu gốc; không tính bột thạch anh (chất độn) hoặc sợi thép (cốt gia cường). "
               "Đây là tỷ lệ theo khối lượng, không phải tỷ lệ theo thể tích hay % độ quan trọng AI.")

    df = pd.concat([df, pd.DataFrame([{
        "Vật liệu": "TỔNG", "Định lượng (kg/m³)": df["Định lượng (kg/m³)"].sum(),
        "Khối lượng mẻ (kg)": df["Khối lượng mẻ (kg)"].sum(),
        "Tỷ lệ khối lượng (%)": 100.0,
    }])], ignore_index=True)
    st.dataframe(df.style.format({
        "Định lượng (kg/m³)": "{:,.2f}", "Khối lượng mẻ (kg)": "{:,.3f}",
        "Tỷ lệ khối lượng (%)": "{:.2f}%",
    }), use_container_width=True, hide_index=True)
    with st.expander("📊 Xem biểu đồ % khối lượng các vật liệu", expanded=False):
        st.bar_chart(tab.set_index("Vật liệu")[["Tỷ lệ khối lượng (%)"]], horizontal=True)
        st.caption("Biểu đồ này là tỷ lệ vật liệu trong cấp phối thực tế, KHÁC với tab 'Ảnh hưởng vật liệu' về Feature Importance.")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Xuất bảng cấp phối CSV (có % khối lượng)", data=csv_bytes,
                       file_name=f"UHPC_cap_phoi_{suffix}.csv", mime="text/csv", key=f"dl_{suffix}")


def show_ratios(mix: dict[str, float], densities: dict[str, float], age: float, model_ratio: float | None = None):
    physical = physical_quantities(pd.DataFrame([mix]), densities).iloc[0]
    cs = st.columns(5)
    cs[0].metric("W/B (không tính bột thạch anh)", f"{physical['W_B']:.3f}")
    cs[1].metric("W/Tổng bột (theo AI)", f"{physical['W_powder']:.3f}")
    cs[2].metric("Σ thể tích tuyệt đối", f"{physical['absolute_volume']:.3f} m³")
    cs[3].metric("Sợi thép (% thể tích)", f"{steel_volume_percent(mix['Soi_thep'], densities['Soi_thep']):.2f}%")
    cs[4].metric("Tuổi dự đoán", f"{age:g} ngày")
    st.caption("W/Tổng bột là đặc trưng `Ty_le_W_B` có trong mô hình notebook; W/B thông thường KHÔNG tính bột thạch anh. Khối lượng riêng đang dùng là giá trị tham khảo và phải hiệu chỉnh theo vật liệu thực tế.")
    return physical


st.markdown("""
<div class="hero">
<small>UHPC STUDIO • AI-ASSISTED MIX DESIGN</small>
<h1>Thiết kế cấp phối bê tông UHPC</h1>
<p>Dự đoán cường độ nén và đề xuất cấp phối có tro bay &amp; xỉ lò cao, cường độ nén >= 120Mpa.</p>
</div>
""", unsafe_allow_html=True)

model_path = find_existing(MODEL_PATHS)
model = None
bundle = {}
features = FEATURES_15
load_error = None
if model_path:
    try:
        bundle, model, features = read_model(str(model_path), model_path.stat().st_mtime_ns)
    except Exception as exc:
        load_error = str(exc)

training_path = find_existing(TRAINING_PATHS)
training_bounds = None
training = None
training_error = None
if model is not None:
    try:
        if training_path:
            training = load_training(str(training_path), training_path.stat().st_mtime_ns)
        if "feature_bounds" in bundle:
            training_bounds = {str(k): tuple(map(float, v)) for k, v in bundle["feature_bounds"].items()}
        elif training is not None:
            training_bounds = training_bounds_from_df(training, features)
    except Exception as exc:
        training_error = str(exc)

if model is None:
    st.warning("**Chưa có mô hình AI đã huấn luyện.** Giao diện và công thức tính mẻ vẫn dùng được; các chức năng AI tạm khóa. Đọc `HUONG_DAN.md` và chạy `COLAB_EXPORT.py` trong notebook để xuất mô hình vào `assets/uhpc_model.pkl`.")
    if load_error:
        st.error("Không nạp được mô hình: " + load_error)
else:
    name = type(model).__name__
    st.success(f"Đã nạp mô hình **{name}** từ `{model_path.relative_to(ROOT)}` ({len(features)} đặc trưng). Chỉ hiển thị dự đoán do mô hình này trả về.")
    if training_bounds is None:
        st.info("Chưa có giới hạn dữ liệu huấn luyện. Có thể dự đoán theo cấp phối nhập vào, nhưng **chưa cho tối ưu ngược** để tránh tự ý tìm kiếm ngoài phạm vi dữ liệu.")
    if training_error:
        st.error("Không đọc được phạm vi dữ liệu huấn luyện: " + training_error)

def show_data_similarity(info: dict | None) -> None:
    if not info:
        st.info("Chưa có dữ liệu đầy đủ để kiểm tra vùng lân cận của cấp phối.")
        return
    st.markdown("#### Khoảng cách đến dữ liệu huấn luyện")
    c1, c2, c3 = st.columns(3)
    c1.metric("Khoảng cách lân cận (chuẩn hóa)", f"{info['distance_k']:.3f}")
    c2.metric("Mốc P95 dữ liệu train", f"{info['reference_p95']:.3f}")
    c3.metric("Số mẫu train duy nhất", f"{info['unique_training_mixes']:,}")
    if info["outside_typical_region"]:
        st.warning("⚠️ Cấp phối nằm xa hơn vùng lân cận của phần lớn dữ liệu huấn luyện. "
                   "Kết quả AI có nguy cơ ngoại suy; không coi là cấp phối đã xác nhận.")
    else:
        st.info("Cấp phối tương đối gần các mẫu train theo chỉ số đã chọn. "
                "**Điều này không bảo đảm cường độ nén hoặc độ chính xác dự đoán.**")
    st.caption(f"Tập tham chiếu: {info.get('scope', 'Toàn bộ train')}. "
               "Khoảng cách Euclid sau chuẩn hóa min-max, "
               f"láng giềng thứ {info['k']}; mốc P95 chỉ là ngưỡng sàng lọc từ train. "
               "KHÔNG phải độ tin cậy 95%, không kiểm chứng độ chảy hoặc độ bền thực tế.")


def show_calibration(bundle: dict, prediction: float) -> None:
    validation = bundle.get("validation", {}) if isinstance(bundle, dict) else {}
    q90 = validation.get("calibration_q90_abs_error_mpa")
    if q90 is not None and isinstance(q90, (int, float)) and 0 <= float(q90) < 1000:
        low, high = max(0.0, prediction - float(q90)), prediction + float(q90)
        st.info(f"Khoảng dự đoán tham khảo từ tập calibration độc lập (~90%, theo giả định): "
                f"**{low:.2f} – {high:.2f} MPa**. Chỉ áp dụng có điều kiện khi dữ liệu mới "
                "cùng phân bố, không bảo đảm bao phủ trường hợp ngoại suy.")
    else:
        st.caption("Mô hình hiện chưa có kết quả hiệu chuẩn sai số từ tập calibration độc lập. "
                   "Không thể hiển thị khoảng dự đoán có căn cứ.")


with st.sidebar:
    st.header("⚙️ Cài đặt chung")
    batch_kind = st.radio("Cách tính mẻ trộn", ["Khối lượng (kg)", "Thể tích (m³)"], horizontal=False)
    if batch_kind == "Khối lượng (kg)":
        batch_mode = "kg"
        batch_amount = st.number_input("Khối lượng mẻ (kg)", 0.1, 10000.0, 50.0, 1.0)
    else:
        batch_mode = "m3"
        batch_amount = st.number_input("Thể tích mẻ (m³)", 0.001, 100.0, 0.02, 0.001, format="%.3f")
    st.divider()
    st.subheader("Thuộc tính vật liệu")
    st.caption("Notebook đang sử dụng các giá trị mặc định sau. Thay bằng kết quả kiểm nghiệm thực tế nếu có.")
    fixed = {}
    fixed["FlyAsh_SiO2"] = st.number_input("SiO₂ tro bay (%)", 0.0, 100.0, 60.0, 0.5)
    fixed["FlyAsh_Blaine"] = st.number_input("Blaine tro bay (đơn vị theo dữ liệu gốc)", 1.0, 20000.0, 3500.0, 100.0)
    fixed["GGBFS_Grade"] = st.number_input("Mác GGBFS (theo dữ liệu gốc)", 0.0, 500.0, 100.0, 5.0)
    fixed["He_so_quy_doi_mau"] = st.number_input("Hệ số quy đổi mẫu", 0.1, 3.0, 1.0, 0.05)
    with st.expander("Khối lượng riêng vật liệu (kg/m³)"):
        st.caption("Các giá trị khởi tạo chỉ là giả định để kiểm tra thể tích tuyệt đối.")
        densities = {}
        for k in MATERIALS:
            densities[k] = st.number_input(LABELS[k], min_value=100.0,
                max_value=15000.0, value=float(DEFAULT_DENSITIES[k]), step=50.0, key="density_" + k)
    st.divider()
    st.caption("Dự đoán AI ≠ kết quả thí nghiệm. Chưa xác thực khả năng thi công, độ chảy, cường độ thực tế hay độ bền lâu.")

tab_inverse, tab_forward, tab_importance, tab_dataset, tab_method = st.tabs([
    "🎯 Thiết kế cấp phối", "🧪 Dự đoán cường độ",
    "📊 Ảnh hưởng vật liệu", "🔍 Kiểm tra dữ liệu", "📘 Mô hình & phạm vi sử dụng",
])

with tab_inverse:
    st.subheader("Thiết kế ngược — lựa chọn tổ hợp chất liên kết")
    st.write(
        "**Differential Evolution theo lô** tìm cấp phối có cường độ **dự đoán ≥ mức yêu cầu**, "
        "kiểm tra W/B, thể tích và sợi thép. Bản mới sàng lọc độ gần dữ liệu train NGAY KHI "
        "tối ưu, thay vì chỉ cảnh báo sau khi chọn nghiệm. Không chỉnh sửa số dự đoán để ép "
        "về đúng cường độ mục tiêu."
    )
    binder_mode = st.selectbox(
        "**Lựa chọn tổ hợp chất liên kết**",
        options=list(BINDER_CHOICES),
        format_func=lambda value: BINDER_CHOICES[value],
        key="binder_mode_selector",
        help="Tùy chọn Chỉ xi măng có nghĩa FA=0 và GGBFS=0; silica fume được đặt riêng bên dưới.",
    )
    custom_fa = "required"
    custom_gg = "required"
    if binder_mode == "CUSTOM":
        cfa, cgg = st.columns(2)
        custom_fa = cfa.selectbox(
            "Trạng thái tro bay", list(POLICY_CHOICES),
            format_func=lambda value: POLICY_CHOICES[value], key="custom_fa_selector",
        )
        custom_gg = cgg.selectbox(
            "Trạng thái GGBFS", list(POLICY_CHOICES),
            format_func=lambda value: POLICY_CHOICES[value], key="custom_gg_selector",
        )
    with st.expander("Silica fume và bột thạch anh — cài đặt riêng", expanded=True):
        csf, cqp = st.columns(2)
        silica_mode = csf.selectbox(
            "Silica fume", list(POLICY_CHOICES), index=1,
            format_func=lambda value: POLICY_CHOICES[value], key="silica_policy_selector",
        )
        quartz_mode = cqp.selectbox(
            "Bột thạch anh (chất độn)", list(POLICY_CHOICES), index=1,
            format_func=lambda value: POLICY_CHOICES[value], key="quartz_policy_selector",
        )
        st.caption("Bột thạch anh là chất độn, không được tính là chất liên kết trong W/B quy ước. Chọn 'Không sử dụng' sẽ cố định lượng vật liệu bằng 0 kg/m³.")

    # Ẩn kết quả cũ nếu cài đặt thuộc tính vật liệu đã thay đổi.
    choice_key = (binder_mode, custom_fa, custom_gg, silica_mode, quartz_mode,
                  tuple((k, float(densities[k])) for k in MATERIALS),
                  tuple((k, float(fixed[k])) for k in DEFAULT_FIXED))
    if binder_mode == "BOTH":
        fa_policy, gg_policy = "required", "required"
    elif binder_mode == "FA_ONLY":
        fa_policy, gg_policy = "required", "off"
    elif binder_mode == "GGBFS_ONLY":
        fa_policy, gg_policy = "off", "required"
    elif binder_mode == "CEMENT_ONLY":
        fa_policy, gg_policy = "off", "off"
    else:
        fa_policy, gg_policy = custom_fa, custom_gg

    with st.form("optimization"):
        left, mid, right = st.columns(3)
        target = left.number_input("Cường độ nén mục tiêu (MPa)", min_value=40.0, max_value=250.0, value=120.0, step=5.0)
        age = mid.number_input("Tuổi nén (ngày)", min_value=1, max_value=365, value=28, step=1)
        max_cement = right.number_input("Xi măng tối đa (kg/m³)", min_value=1.0, max_value=1500.0, value=850.0, step=25.0)
        mode = st.radio("Chế độ tìm kiếm", ["Nhanh", "Chuyên sâu"], horizontal=True)
        st.markdown("#### 🔎 Kiểm soát ngoại suy trong lúc tối ưu")
        safe_mode = st.radio(
            "Phương pháp chọn cấp phối",
            ["Ưu tiên độ tin cậy (khuyến nghị)", "Chỉ cảnh báo ngoại suy (đối chiếu v6)"],
            horizontal=False,
            help="Chế độ khuyến nghị buộc ứng viên phải có khoảng cách láng giềng "
                 "không vượt mốc P95 của các mẫu train cùng tuổi và cùng tổ hợp vật liệu. "
                 "Đây chỉ là cách sàng lọc, không phải bảo đảm cường độ 95%.",
        )
        similarity_policy = ("strict" if safe_mode.startswith("Ưu tiên") else "report_only")
        prefer_near_target = st.checkbox(
            "Ưu tiên cường độ dự đoán gần mức yêu cầu (không ép bằng mục tiêu)",
            value=True,
            help="Đối với yêu cầu 130 MPa, thuật toán chỉ ưu tiên ứng viên cao hơn 130 MPa "
                 "ít nhất có thể; luôn hiển thị kết quả model tính thật. "
                 "Nếu chưa có nghiệm gần mục tiêu, vẫn báo chính xác kết quả tìm được.",
        )
        st.caption("Nếu dữ liệu cùng tổ hợp quá ít hoặc không có nghiệm nằm trong vùng P95, "
                   "app sẽ báo 'chưa tìm được' thay vì tự tạo ra cấp phối đạt.")
        with st.expander("Ràng buộc kỹ thuật — điều chỉnh theo đề tài và vật liệu thực tế"):
            a, b, c = st.columns(3)
            wb_min = a.number_input("W/B tối thiểu", min_value=0.05, max_value=0.50, value=0.14, step=0.01, format="%.2f")
            wb_max = b.number_input("W/B tối đa", min_value=0.05, max_value=0.50, value=0.24, step=0.01, format="%.2f")
            margin = c.number_input("Biên dự phòng AI (MPa)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            a2, b2 = st.columns(2)
            min_fa = a2.number_input("Tro bay tối thiểu (kg/m³)", min_value=0.01, max_value=1000.0, value=20.0, step=5.0) if fa_policy == "required" else 20.0
            min_gg = b2.number_input("GGBFS tối thiểu (kg/m³)", min_value=0.01, max_value=1000.0, value=20.0, step=5.0) if gg_policy == "required" else 20.0
            a3, b3 = st.columns(2)
            min_sf = a3.number_input("Silica fume tối thiểu (kg/m³)", min_value=0.01, max_value=1000.0, value=20.0, step=5.0) if silica_mode == "required" else 20.0
            min_qp = b3.number_input("Bột thạch anh tối thiểu (kg/m³)", min_value=0.01, max_value=1000.0, value=20.0, step=5.0) if quartz_mode == "required" else 20.0
            c2, d2 = st.columns(2)
            air_percent = c2.number_input("Khí giả định (%)", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
            vol_tol = d2.number_input("Sai số thể tích (m³)", min_value=0.005, max_value=0.1, value=0.02, step=0.005, format="%.3f")
        st.markdown("#### 🧵 Giới hạn và ưu tiên giảm sợi thép")
        f_left, f_right = st.columns([1, 2])
        max_steel_pct = f_left.slider(
            "Sợi thép tối đa (% thể tích)", min_value=0.0, max_value=3.0,
            value=2.0, step=0.25,
            help="0% = không dùng sợi; 2% thể tích với ρ = 7.850 kg/m³ tương ứng 157 kg/m³.",
        )
        priority_label = f_right.radio(
            "Mức ưu tiên tiết kiệm sợi thép",
            ["Cân bằng", "Ưu tiên mạnh", "Không ưu tiên"],
            horizontal=True,
            help="Điểm tối ưu = kg xi măng/m³ + trọng số × kg sợi/m³. "
                 "Trọng số kỹ thuật, không phải đơn giá hay bằng chứng nhân quả.",
        )
        steel_weight = {"Không ưu tiên": 0.0, "Cân bằng": 1.0, "Ưu tiên mạnh": 3.0}[priority_label]
        st.caption(
            f"Giới hạn cứng: {max_steel_pct:.2f}% thể tích = "
            f"{max_steel_pct / 100.0 * densities['Soi_thep']:.2f} kg sợi/m³ "
            f"(ρ sợi = {densities['Soi_thep']:,.0f} kg/m³). "
            "Không tự nới giới hạn nếu AI chưa tìm được cấp phối hợp lệ."
        )
        run_opt = st.form_submit_button("🔎 Tìm cấp phối với mô hình AI", type="primary", disabled=(model is None or training_bounds is None), use_container_width=True)
    st.caption(
        "Giới hạn vật liệu, W/B và thể tích là điều kiện sàng lọc người dùng lựa chọn; "
        "không mặc nhiên là yêu cầu tiêu chuẩn. Tổ hợp có mẫu huấn luyện không có nghĩa cấp phối mới đã được kiểm chứng."
    )
    if run_opt:
        try:
            rules = DesignRules(
                wb_min=wb_min, wb_max=wb_max, air_fraction=air_percent / 100.0,
                volume_tolerance=vol_tol, min_flyash=min_fa, min_ggbfs=min_gg,
                margin_mpa=margin, binder_mode=binder_mode,
                custom_fa=custom_fa, custom_ggbfs=custom_gg,
                silica_mode=silica_mode, quartz_mode=quartz_mode,
                min_silica=min_sf, min_quartz=min_qp,
                max_steel_vol_percent=max_steel_pct, steel_saving_weight=steel_weight,
                similarity_policy=similarity_policy, prefer_near_target=prefer_near_target,
            )
            rules.validate()
            if training is not None:
                support = count_training_matches(training, rules, densities["Soi_thep"])
                st.info(f"Dữ liệu huấn luyện có **{support} tổ hợp đặc trưng khác nhau** khớp trạng thái vật liệu và ngưỡng tối thiểu. Không phải số thí nghiệm độc lập hoặc đánh giá độ tin cậy.")
            with st.spinner("Đang dự đoán theo lô và tìm cấp phối phù hợp tổ hợp đã chọn..."):
                result = optimize_mix(
                    model, features, training_bounds, float(age), fixed, densities,
                    rules, target, max_cement, mode=mode, training=training,
                )
            st.session_state["inverse_result"] = result
            st.session_state["inverse_choice_key"] = choice_key
        except Exception as exc:
            st.session_state.pop("inverse_result", None)
            st.session_state.pop("inverse_choice_key", None)
            st.error(f"Không thực hiện được tối ưu: {exc}")
    result = st.session_state.get("inverse_result")
    if result is not None and st.session_state.get("inverse_choice_key") != choice_key:
        st.info("Bạn đã đổi tổ hợp vật liệu. Nhấn **Tìm cấp phối với mô hình AI** để tính lại; kết quả cũ đã được ẩn.")
        result = None
    if result is not None:
        st.markdown(f"**Tổ hợp đã tối ưu:** {result['binder_label']}")
        if result["training_support_count"] is not None:
            st.caption(
                f"Mẫu train duy nhất có cùng tuổi và tổ hợp vật liệu: "
                f"{result['training_support_count']}. Nghiệm qua ràng buộc tìm được: "
                f"{result['feasible_candidates_seen']}. "
                "Đây không phải số thí nghiệm xác nhận cường độ cấp phối mới."
            )
        st.caption("**Chiến lược:** " + (
            "Ràng buộc khoảng cách trong tối ưu" if result.get("similarity_policy") == "strict"
            else "Chỉ cảnh báo khoảng cách (cách cũ, để đối chiếu)"
        ))
        if result["feasible"]:
            validation = bundle.get("validation", {})
            q90 = validation.get("calibration_q90_abs_error_mpa") if isinstance(validation, dict) else None
            uncertain_target = (
                isinstance(q90, (int, float)) and 0 <= float(q90) < 1000
                and result["prediction"] - float(q90) < result["target"]
            )
            if ((result.get("training_similarity") and result["training_similarity"]["outside_typical_region"]) or
                (result.get("training_similarity_local") and result["training_similarity_local"]["outside_typical_region"])):
                st.warning("Ứng viên thỏa các ràng buộc số học nhưng **xa dữ liệu huấn luyện**; "
                           "không nên sử dụng như cấp phối đã được kiểm chứng.")
            elif uncertain_target:
                st.warning("Ứng viên thỏa ràng buộc dự đoán điểm nhưng **cận dưới của khoảng "
                           "dự đoán tham khảo thấp hơn mục tiêu**. Chưa có đủ căn cứ để tin "
                           "cường độ thực tế sẽ đạt yêu cầu.")
            else:
                st.success("Đã tìm thấy **phương án thỏa các ràng buộc đang chọn theo mô hình AI**. Chưa phải cấp phối được thí nghiệm xác nhận.")
        else:
            st.error("**Chưa tìm được cấp phối đồng thời đáp ứng cường độ dự đoán, "
                     "điều kiện vật lý và giới hạn ngoại suy đã chọn.** "
                     "Bảng bên dưới chỉ là ứng viên chẩn đoán; KHÔNG phải cấp phối đạt yêu cầu.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cường độ AI dự đoán", f"{result['prediction']:.2f} MPa")
        col2.metric("Cường độ yêu cầu", f"{result['target']:.1f} MPa")
        col3.metric("Số phép dự đoán", f"{result['evaluations']:,}")
        st.caption(
            f"Chênh lệch **dự đoán – mục tiêu** = "
            f"{result['prediction']-result['target']:+.2f} MPa. "
            "Cường độ tối thiểu là điều kiện >=, không yêu cầu dự đoán phải "
            "bằng chính xác giá trị nhập."
        )
        show_calibration(bundle, result["prediction"])
        st.markdown("#### Đánh giá vùng dữ liệu toàn tập")
        show_data_similarity(result.get("training_similarity"))
        st.markdown("#### Đánh giá nhóm có cùng tuổi và tổ hợp vật liệu")
        show_data_similarity(result.get("training_similarity_local"))
        show_ratios(result["mix"], densities, result["age"])
        st.caption(
            f"Sợi thép đề xuất: {result['mix']['Soi_thep']:.2f} kg/m³ "
            f"({result['steel_volume_percent']:.2f}% thể tích); "
            f"giới hạn tối đa: {result['steel_limit_kg_m3']:.2f} kg/m³ "
            f"({result['steel_limit_vol_percent']:.2f}% thể tích). "
            f"Trọng số ưu tiên giảm sợi: {result['steel_saving_weight']:g}."
        )
        st.markdown("#### Bảng cấp phối đề xuất / ứng viên")
        display_mix(result["mix"], batch_mode, batch_amount, "de_xuat")
        st.markdown("#### Kiểm tra điều kiện đầu ra")
        st.dataframe(pd.DataFrame([{"Điều kiện": k, "Đạt": "✓" if ok else "✗"}
                                   for k, ok in result["checks"].items()]),
                     use_container_width=True, hide_index=True)
        for warning in result["warnings"]:
            st.warning(warning)
        st.caption("Thuật toán chỉ kiểm tra điều kiện mô hình/giả định. Cần thử độ chảy, phân tán sợi, mẫu nén và độ bền lâu trước khi áp dụng thực tế.")
        chart = pd.DataFrame({"Vật liệu": [LABELS[k] for k in MATERIALS], "kg/m³": [result["mix"][k] for k in MATERIALS]})
        st.bar_chart(chart.set_index("Vật liệu"))

with tab_forward:
    st.subheader("Bài toán thuận — dự đoán từ cấp phối nhập vào")
    st.info("Bộ giá trị sẵn bên dưới là **cấp phối kiểm thử trong cell 23 của notebook**, không phải kết quả thực nghiệm được xác nhận. Bạn có thể sửa từng vật liệu.")
    with st.form("forward"):
        mix = {}
        for i, key in enumerate(MATERIALS):
            if i % 3 == 0:
                cols = st.columns(3)
            mix[key] = cols[i%3].number_input(f"{LABELS[key]} (kg/m³)",
                min_value=0.0, max_value=3000.0, value=float(NOTEBOOK_TEST_MIX[key]),
                step=5.0, key="mix_" + key)
        forward_age = st.number_input("Tuổi dự đoán (ngày)", 1, 365, 28, 1)
        submit_pred = st.form_submit_button("🧠 Dự đoán cường độ", type="primary", disabled=(model is None), use_container_width=True)
    if submit_pred:
        try:
            yhat = predict(model, mix, forward_age, fixed, features)
            feat = feature_frame(pd.DataFrame([mix]), float(forward_age), fixed, features)
            warnings = bound_warnings(feat, training_bounds) if training_bounds else []
            st.session_state["forward_result"] = (mix.copy(), float(forward_age), yhat, warnings)
        except Exception as exc:
            st.session_state.pop("forward_result", None)
            st.error(f"Không dự đoán được: {exc}")
    if "forward_result" in st.session_state:
        cmix, cage, yhat, warnings = st.session_state["forward_result"]
        st.metric("Cường độ nén AI dự đoán", f"{yhat:.2f} MPa", help="Không phải số đo thí nghiệm.")
        show_calibration(bundle, yhat)
        if training is not None:
            try:
                show_data_similarity(training_similarity(cmix, cage, fixed, features, training))
            except ValueError as exc:
                st.warning(f"Không thể đánh giá độ gần dữ liệu: {exc}")
        show_ratios(cmix, densities, cage)
        if not training_bounds:
            st.warning("Không có dải huấn luyện để kiểm tra ngoại suy cho cấp phối này.")
        for warning in warnings:
            st.warning(warning)
        display_mix(cmix, batch_mode, batch_amount, "du_doan")

with tab_importance:
    st.subheader("📊 Vật liệu nào quan trọng nhất trong dự đoán cường độ UHPC?")
    st.caption(
        "Biểu đồ lấy số liệu trực tiếp từ mô hình AI **đang được nạp trên website**; "
        "không sử dụng các tỷ lệ giả định và không cần chạy Colab lại."
    )
    if model is None:
        st.warning("Chưa có mô hình đã huấn luyện trong `assets/uhpc_model.pkl`; không thể vẽ biểu đồ thực.")
    else:
        try:
            importance, importance_method = feature_importance_table(model, features)
            if importance_method == "gain":
                st.info(
                    "**Phương pháp LightGBM Gain:** đo tổng mức cải thiện hàm mất mát "
                    "tại những nút cây đã sử dụng đặc trưng. Điểm càng cao → mô hình "
                    "đã dựa vào đặc trưng đó càng nhiều khi dự đoán. **Không chứng minh "
                    "vật liệu đó gây tăng hay giảm cường độ.**"
                )
            else:
                st.warning(
                    f"Mô hình đang nạp là **{type(model).__name__}**, không phải LightGBM. "
                    "Biểu đồ dùng impurity-based importance, KHÔNG phải gain của LightGBM. "
                    "Để có biểu đồ Gain, hãy xuất lại mô hình LightGBM từ Colab."
                )

            materials_chart = importance[importance["Nhóm"] == "Vật liệu"].copy()
            materials_chart = materials_chart.sort_values("Tỷ trọng trong 9 vật liệu (%)", ascending=False)
            all_chart = importance.sort_values("Tỷ trọng toàn mô hình (%)", ascending=False)
            if float(importance["Điểm gốc"].sum()) <= 0:
                st.warning("Mô hình trả về điểm quan trọng bằng 0 cho mọi đặc trưng; không thể xếp hạng có ý nghĩa.")
            elif float(materials_chart["Điểm gốc"].sum()) <= 0:
                st.warning("9 vật liệu không có điểm importance dương. Mô hình đang dựa vào các đặc trưng khác; không thể kết luận thứ hạng vật liệu.")
            else:
                top = materials_chart.iloc[0]
                st.metric("Vật liệu quan trọng nhất theo mô hình", top["Tên hiển thị"])
                st.markdown("#### 1. Xếp hạng 9 vật liệu trong cấp phối")
                st.bar_chart(
                    materials_chart.set_index("Tên hiển thị")[["Tỷ trọng trong 9 vật liệu (%)"]],
                    horizontal=True, height=370, x_label="% trong tổng điểm của 9 vật liệu",
                    y_label="Vật liệu", use_container_width=True,
                )
                st.caption(
                    "Các tỷ lệ của biểu đồ này được **chuẩn hóa trong 9 vật liệu**, tổng = 100%. "
                    "Chúng không phải % tăng cường độ nén hoặc % thành phần cấp phối."
                )
            if float(importance["Điểm gốc"].sum()) > 0:
                st.markdown("#### 2. Đóng góp của toàn bộ 15 đặc trưng")
                st.bar_chart(
                    all_chart.set_index("Tên hiển thị")[["Tỷ trọng toàn mô hình (%)"]],
                    horizontal=True, height=500, x_label="% tổng điểm importance của 15 đặc trưng",
                    y_label="Đặc trưng", use_container_width=True,
                )
                st.caption(
                    "Biểu đồ gồm 9 vật liệu và 6 đặc trưng khác (tuổi, tỷ lệ nước/tổng bột, "
                    "chất lượng nguyên liệu, hệ số mẫu). Đặc trưng có tương quan với nhau "
                    "có thể chia sẻ tầm quan trọng, chẳng hạn Nước và tỷ lệ nước/tổng bột."
                )
            st.markdown("#### Bảng số liệu từ mô hình")
            cols = ["Tên hiển thị", "Nhóm", "Tỷ trọng toàn mô hình (%)", "Tỷ trọng trong 9 vật liệu (%)", "Điểm gốc"]
            st.dataframe(
                importance[cols].style.format({
                    "Tỷ trọng toàn mô hình (%)": "{:.2f}",
                    "Tỷ trọng trong 9 vật liệu (%)": "{:.2f}",
                    "Điểm gốc": "{:,.3f}",
                }, na_rep="—"),
                hide_index=True, use_container_width=True,
            )
            st.download_button(
                "⬇️ Tải bảng ảnh hưởng vật liệu (CSV)",
                data=importance.to_csv(index=False).encode("utf-8-sig"),
                file_name="UHPC_LightGBM_feature_importance.csv" if importance_method == "gain" else "UHPC_feature_importance.csv",
                mime="text/csv",
            )
            st.warning(
                "**Giới hạn khoa học:** Feature Importance không chỉ ra dấu tác động (+/−), "
                "không có nghĩa tăng vật liệu này sẽ làm cường độ tăng. Đây là mức quan trọng "
                "đối với dự đoán của mô hình trong dữ liệu huấn luyện; không phải kết quả thí nghiệm độc lập."
            )
        except Exception as exc:
            st.error(f"Không tính được biểu đồ tầm quan trọng từ mô hình: {exc}")


with tab_dataset:
    st.subheader("Kiểm tra chất lượng dữ liệu được đóng gói cùng mô hình")
    st.caption("Tệp assets/training_features.csv chỉ gồm X_train; không có cường độ đo, "
               "nguồn bài báo hay dữ liệu test độc lập. Không thể suy ra R², MAE, RMSE từ tệp này.")
    if training is None:
        st.warning("Không tìm thấy tệp dữ liệu huấn luyện; không thể kiểm tra.")
    else:
        try:
            report = training_data_audit(training, features)
            c1, c2, c3 = st.columns(3)
            c1.metric("Dòng train", f"{report['total_rows']:,}")
            c2.metric("Dòng đặc trưng trùng", f"{report['exact_duplicate_features']:,}")
            c3.metric("Có FA & GGBFS, 28 ngày", f"{report['with_fa_and_ggbfs_28d']:,}")
            st.write("**Đặc trưng không biến thiên:** " +
                     (", ".join(report['constant_features']) if report['constant_features'] else "Không có"))
            st.warning("Các bản ghi giống X chưa chắc là dữ liệu đo trùng: có thể có mẫu lặp "
                       "khác cường độ. Chỉ loại trùng sau khi kiểm tra y và nguồn bài báo. "
                       "Các đặc trưng hằng không thể dùng để kết luận vai trò hóa lý của vật liệu.")
            st.markdown("**Việc cần chạy trong Google Colab:** đánh giá dữ liệu có nhãn y, "
                        "chia train/calibration/test theo nhóm, xuất R²/MAE/RMSE độc lập "
                        "và mô hình LightGBM mới khi đã kiểm chứng.")
        except Exception as exc:
            st.error(f"Không kiểm tra được dữ liệu: {exc}")


with tab_method:
    st.subheader("Thông tin mô hình và giới hạn sử dụng")
    st.write("**Tổ hợp chất liên kết:** 5 chế độ (hai vật liệu, chỉ FA, chỉ GGBFS, không FA/GGBFS, tùy chỉnh). Các lựa chọn không thuộc dữ liệu huấn luyện bị từ chối tối ưu. Silica fume được cấu hình riêng, bột thạch anh là chất độn.")
    st.write("**Giới hạn sợi thép v6:** sợi thép là biến độc lập trong mô hình nhưng bị chặn cứng theo % thể tích ở bài toán tối ưu ngược. Mặc định 2% thể tích = 157 kg/m³ nếu khối lượng riêng sợi = 7.850 kg/m³. Không đồng nghĩa đây là mức tiêu chuẩn cho mọi UHPC. Hàm mục tiêu ưu tiên giảm cả lượng xi măng và lượng sợi với trọng số kỹ thuật (không phải giá thị trường); cấp phối chỉ được chấp nhận sau khi thỏa giới hạn cứng, điều kiện W/B, thể tích và cường độ dự đoán. Bài toán dự đoán thuận cho phép nhập tự do nhưng hiển thị % sợi để người dùng tự kiểm tra.")
    st.write("**15 đặc trưng của notebook:**")
    st.code(", ".join(FEATURES_15), language="text")
    st.write("**Quan trọng:** `Ty_le_W_B = Nước / (Xi măng + FA + GGBFS + Silica fume + Bột thạch anh)` là định nghĩa *theo mô hình cũ*. Ứng dụng giữ nguyên để tương thích, đồng thời tính W/B quy ước riêng không chứa bột thạch anh.")
    if model is not None:
        st.write("Mô hình đang nạp:", type(model).__name__)
        st.write("Thứ tự 15 đặc trưng thực tế:", features)
    if training_bounds:
        st.write("Miền huấn luyện (min/max từng đặc trưng; **không** bảo đảm mọi tổ hợp đều đã có trong dữ liệu):")
        st.dataframe(pd.DataFrame([{"Đặc trưng": k, "Nhỏ nhất": v[0], "Lớn nhất": v[1]}
                                   for k, v in training_bounds.items()]), use_container_width=True, hide_index=True)
    st.warning("Không có bảo đảm đạt cường độ thực tế ≥120 MPa. Dải giới hạn vật liệu là dải từ dữ liệu huấn luyện, không phải giới hạn tiêu chuẩn UHPC. Kết quả cần kiểm nghiệm độc lập.")
    st.markdown("**Thiếu dữ liệu thí nghiệm đặc thù:** tình trạng dưỡng hộ, tính công tác, đặc điểm sợi và nguồn cốt liệu có thể khác giữa các tài liệu. Không dùng ứng dụng này làm hồ sơ thiết kế cấp phối được phê duyệt khi chưa hiệu chuẩn và thí nghiệm.")
st.markdown("""
<style>
:root{
    --bg-main: #020817;
    --text-main: #F8FAFC;
    --text-muted: #CBD5E1;

    --card-bg: #F8FAFC;
    --card-title: #64748B;
    --card-value: #0F172A;
    --card-sub: #334155;

    --success-bg: #14532D;
    --success-text: #DCFCE7;

    --info-bg: #0F2747;
    --info-text: #DBEAFE;
}

/* Nền tổng */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-main);
    color: var(--text-main);
}

/* Tiêu đề section */
h1, h2, h3, h4 {
    color: var(--text-main) !important;
    font-weight: 700 !important;
}

/* Chữ mô tả */
p, li, span, label, div {
    color: var(--text-main);
}

/* Chữ phụ */
.small-text, .muted-text {
    color: var(--text-muted) !important;
}

/* Card số liệu */
.metric-card {
    background: var(--card-bg) !important;
    border-radius: 18px;
    padding: 22px 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
    border: 1px solid rgba(15, 23, 42, 0.06);
}

.metric-card .metric-label {
    color: var(--card-title) !important;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 10px;
}

.metric-card .metric-value {
    color: var(--card-value) !important;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.15;
}

.metric-card .metric-sub {
    color: var(--card-sub) !important;
    font-size: 14px;
    font-weight: 500;
    margin-top: 6px;
}

/* Banner xanh lá */
.success-banner {
    background: var(--success-bg) !important;
    color: var(--success-text) !important;
    border-radius: 14px;
    padding: 16px 18px;
    font-size: 16px;
    font-weight: 700;
}

/* Banner xanh dương */
.info-banner {
    background: var(--info-bg) !important;
    color: var(--info-text) !important;
    border-radius: 14px;
    padding: 16px 18px;
    font-size: 16px;
    font-weight: 600;
}

/* Nội dung markdown phụ trên nền tối */
[data-testid="stMarkdownContainer"] p {
    color: var(--text-muted);
    font-size: 16px;
}

/* Làm nổi số */
.big-number {
    color: #0F172A !important;
    font-weight: 800 !important;
    letter-spacing: 0.2px;
}
</style>
""", unsafe_allow_html=True)
