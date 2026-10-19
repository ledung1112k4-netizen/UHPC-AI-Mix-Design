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
    NOTEBOOK_TEST_MIX, DesignRules, batch_table, bound_warnings,
    check_model_columns, count_training_matches, feature_frame, material_policies, optimize_mix, physical_quantities,
    predict, training_bounds_from_df,
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
    tab, _ = batch_table(mix, mode, amount)
    df = tab.copy()
    total = pd.DataFrame([{
        "Vật liệu": "TỔNG", "Định lượng (kg/m³)": df["Định lượng (kg/m³)"].sum(),
        "Khối lượng mẻ (kg)": df["Khối lượng mẻ (kg)"].sum(),
    }])
    df = pd.concat([df, total], ignore_index=True)
    st.dataframe(df.style.format({"Định lượng (kg/m³)": "{:,.2f}", "Khối lượng mẻ (kg)": "{:,.3f}"}),
                 use_container_width=True, hide_index=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Xuất bảng cấp phối CSV", data=csv_bytes,
                       file_name=f"UHPC_cap_phoi_{suffix}.csv", mime="text/csv", key=f"dl_{suffix}")


def show_ratios(mix: dict[str, float], densities: dict[str, float], age: float, model_ratio: float | None = None):
    physical = physical_quantities(pd.DataFrame([mix]), densities).iloc[0]
    cs = st.columns(4)
    cs[0].metric("W/B (không tính bột thạch anh)", f"{physical['W_B']:.3f}")
    cs[1].metric("W/Tổng bột (theo AI)", f"{physical['W_powder']:.3f}")
    cs[2].metric("Σ thể tích tuyệt đối", f"{physical['absolute_volume']:.3f} m³")
    cs[3].metric("Tuổi dự đoán", f"{age:g} ngày")
    st.caption("W/Tổng bột là đặc trưng `Ty_le_W_B` có trong mô hình notebook; W/B thông thường KHÔNG tính bột thạch anh. Khối lượng riêng đang dùng là giá trị tham khảo và phải hiệu chỉnh theo vật liệu thực tế.")
    return physical


st.markdown("""
<div class="hero">
<small>UHPC STUDIO • AI-ASSISTED MIX DESIGN</small>
<h1>Thiết kế cấp phối bê tông UHPC</h1>
<p>Dự đoán cường độ nén và đề xuất cấp phối có tro bay &amp; xỉ lò cao,
lựa chọn tổ hợp chất liên kết linh hoạt, ưu tiên 120 MPa tại 28 ngày. Không cần mở Google Colab sau khi triển khai.</p>
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

tab_inverse, tab_forward, tab_method = st.tabs([
    "🎯 Thiết kế cấp phối", "🧪 Dự đoán cường độ", "📘 Mô hình & phạm vi sử dụng",
])

with tab_inverse:
    st.subheader("Thiết kế ngược — lựa chọn tổ hợp chất liên kết")
    st.write(
        "**Differential Evolution theo lô** tối ưu lượng xi măng, đồng thời kiểm tra "
        "cường độ AI mục tiêu, W/B, thể tích và đúng vật liệu đã chọn. "
        "Các chế độ ít dữ liệu thực nghiệm có thể bị từ chối tối ưu để tránh ngoại suy."
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

    choice_key = (binder_mode, custom_fa, custom_gg, silica_mode, quartz_mode)
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
            )
            rules.validate()
            if training is not None:
                support = count_training_matches(training, rules)
                st.info(f"Dữ liệu huấn luyện có **{support} mẫu** khớp trạng thái vật liệu và ngưỡng tối thiểu đã chọn. Đây không phải đánh giá độ tin cậy của AI.")
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
            st.caption(f"Số mẫu huấn luyện khớp điều kiện vật liệu: {result['training_support_count']}. Số nghiệm hợp lệ từng gặp trong tìm kiếm: {result['feasible_candidates_seen']}.")
        if result["feasible"]:
            st.success("Đã tìm thấy **phương án thỏa các ràng buộc đang chọn theo mô hình AI**. Chưa phải cấp phối được thí nghiệm xác nhận.")
        else:
            st.error("**Chưa tìm được cấp phối thỏa toàn bộ điều kiện** trong ngân sách tối ưu này. Bảng bên dưới chỉ là ứng viên, **không được xem là cấp phối đạt yêu cầu**.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Cường độ AI dự đoán", f"{result['prediction']:.2f} MPa")
        col2.metric("Cường độ yêu cầu", f"{result['target']:.1f} MPa")
        col3.metric("Số phép dự đoán", f"{result['evaluations']:,}")
        show_ratios(result["mix"], densities, result["age"])
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
        show_ratios(cmix, densities, cage)
        if not training_bounds:
            st.warning("Không có dải huấn luyện để kiểm tra ngoại suy cho cấp phối này.")
        for warning in warnings:
            st.warning(warning)
        display_mix(cmix, batch_mode, batch_amount, "du_doan")

with tab_method:
    st.subheader("Thông tin mô hình và giới hạn sử dụng")
    st.write("**Tổ hợp chất liên kết:** 5 chế độ (hai vật liệu, chỉ FA, chỉ GGBFS, không FA/GGBFS, tùy chỉnh). Các lựa chọn không thuộc dữ liệu huấn luyện bị từ chối tối ưu. Silica fume được cấu hình riêng, bột thạch anh là chất độn.")
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
