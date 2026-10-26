"""UHPC AI STUDIO 2.2: no invented reference mixtures, predictions or validation.

GitHub drop-in: this app reads existing assets/uhpc_model.pkl and
assets/training_features.csv and keeps your existing uhpc_core.py untouched.
"""
from __future__ import annotations

from pathlib import Path
import io
import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from uhpc_core import MATERIALS, FEATURES_15, LABELS, DEFAULT_FIXED, DEFAULT_DENSITIES
from uhpc_quality import (
    AuditError, batch_table, check_mix, cost_table, distance_check,
    distance_reference, evaluate_validation, feature_frame, load_bundle,
    load_training, optimize, physical, predict, request_id, verify_design,
    strength_alignment, validation_near_target,
)

ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [ROOT/"assets"/"uhpc_model.pkl", ROOT/"model"/"uhpc_model.pkl"]
TRAIN_CANDIDATES = [ROOT/"assets"/"training_features.csv", ROOT/"data"/"training_features.csv", ROOT/"data"/"uhpc_reference.csv"]
MODEL_PATH = next((x for x in MODEL_CANDIDATES if x.exists()),MODEL_CANDIDATES[0])
TRAIN_PATH = next((x for x in TRAIN_CANDIDATES if x.exists()),TRAIN_CANDIDATES[0])

st.set_page_config(page_title="UHPC AI Studio 2.2", page_icon="🧱",layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"], [data-testid="stAppViewContainer"] {font-family:'Be Vietnam Pro','Segoe UI',sans-serif;}
[data-testid="stAppViewContainer"]{background:#f5f8fb;}
[data-testid="stSidebar"]{background:#102b41;}
[data-testid="stSidebar"] * {color:#f1f7fb;}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] small{color:#e0edf5!important;}
[data-testid="stSidebar"] input{color:#163247!important;}
.block-container{max-width:1480px;padding-top:1.5rem;}
h1,h2,h3{color:#193248;letter-spacing:-.02em;}
.top{background:linear-gradient(112deg,#10273d 0%,#10455a 70%,#08777e 100%);padding:26px 28px;border-radius:17px;color:#fff;margin-bottom:18px;}
.top h1{color:#fff!important;font-size:30px;margin:0 0 9px 0;}
.top p{color:#d5eaef;font-size:14px;margin:0;line-height:1.65;}
.muted{font-size:12px;color:#667c8e;}
[data-testid="stMetric"]{background:#fff;border:1px solid #e2eaf0;border-radius:13px;padding:15px;}
[data-testid="stForm"]{background:#fff;border:1px solid #e2eaf0;border-radius:13px;padding:14px;}
</style>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model(path, mtime):
    return load_bundle(path)

@st.cache_data(show_spinner=False)
def get_training(path,mtime):
    return load_training(path)

model, cols, metadata, model_error = None, None, {}, None
train, train_error = None, None
try:
    model, cols, metadata = get_model(str(MODEL_PATH),MODEL_PATH.stat().st_mtime)
except Exception as exc:
    model_error = f"{type(exc).__name__}: {exc}"
try:
    train = get_training(str(TRAIN_PATH),TRAIN_PATH.stat().st_mtime)
except Exception as exc:
    train_error = f"{type(exc).__name__}: {exc}"

with st.sidebar:
    st.markdown("## 🧱 UHPC AI STUDIO")
    st.caption("PHÒNG THÍ NGHIỆM THIẾT KẾ CẤP PHỐI · 2.2")
    age=st.number_input("Tuổi nén (ngày)",min_value=1,max_value=365,value=28,step=1)
    target=st.number_input("Cường độ mục tiêu (MPa)",min_value=1.,max_value=300.,value=120.,step=5.)
    match_options={
        "Sát mục tiêu, không thấp hơn":"min_close",
        "Sát mục tiêu trong khoảng ±":"band",
        "Chỉ cần đạt tối thiểu":"minimum_only",
    }
    target_mode_label=st.selectbox("Cách so sánh với cường độ nhập",list(match_options),index=0,
                                   help="Mặc định ưu tiên dự báo nằm ngay phía trên cường độ nhập, không chấp nhận vượt quá dung sai.")
    target_mode=match_options[target_mode_label]
    target_tolerance=st.number_input("Dung sai khớp dự báo (MPa)",min_value=0.001,max_value=100.,value=0.01,step=.01,format="%.3f",
                                     help="Mặc định 0,01 MPa: nhập 120 → khoảng 120–120,01 MPa khi chọn 'không thấp hơn'. Đây KHÔNG phải độ chính xác thực nghiệm.")
    margin=st.number_input("Biên dự phòng số học (MPa)",min_value=0.,max_value=50.,value=0.,step=.5,help="Áp dụng khi cần đạt tối thiểu hoặc sát từ phía trên. Không phải khoảng tin cậy.")
    if target_mode=="band":
        st.caption(f"Khoảng chấp nhận dự báo: {target-target_tolerance:g}–{target+target_tolerance:g} MPa. Có thể thấp hơn mục tiêu; biên dự phòng không áp dụng ở chế độ ±.")
    elif target_mode=="min_close":
        st.caption(f"Khoảng chấp nhận dự báo: {target+margin:g}–{target+margin+target_tolerance:g} MPa.")
    else:
        st.caption(f"Chấp nhận mọi mức dự báo ≥ {target+margin:g} MPa; không ràng buộc độ sát.")
    batch_mode=st.radio("Đơn vị mẻ",["kg","m³"],horizontal=True)
    batch_size=st.number_input("Quy mô mẻ",min_value=.001,max_value=100000.,value=50. if batch_mode=="kg" else 1.,step=5. if batch_mode=="kg" else .1)
    st.divider()
    st.markdown("**Ràng buộc thiết kế nghiên cứu**")
    wb_min,wb_max=st.slider("Giới hạn W/B (không gồm bột thạch anh)",.05,.45,(.14,.24),.01)
    cement_max=st.number_input("Xi măng tối đa (kg/m³)",min_value=1.,max_value=2000.,value=1000.,step=50.)
    steel_limit=st.number_input("Sợi thép tối đa (% thể tích)",min_value=0.,max_value=5.,value=2.,step=.25)
    air_percent=st.number_input("Khí dự kiến (% thể tích)",min_value=0.,max_value=10.,value=2.,step=.5)
    vol_tol=st.number_input("Dung sai thể tích tuyệt đối (m³/m³)",min_value=.001,max_value=.1,value=.02,step=.005,format="%.3f")
    require_fa=st.checkbox("Yêu cầu tro bay > 0",value=True)
    require_ggbfs=st.checkbox("Yêu cầu GGBFS > 0",value=True)
    require_quartz=st.checkbox("Yêu cầu bột thạch anh > 0",value=True)
    st.divider()
    with st.expander("Khối lượng riêng vật liệu (kg/m³)"):
        st.caption("Mặc định là giá trị tham khảo; cần thay bằng phiếu thử vật liệu thực tế.")
        densities={k:st.number_input(LABELS.get(k,k),min_value=100.,max_value=15000.,value=float(DEFAULT_DENSITIES[k]),key=f"rho_{k}") for k in MATERIALS}
    st.caption("🟢 Có mô hình" if model is not None else "🔴 Chưa nạp mô hình")
    st.caption(f"🟢 {len(train):,} dòng train" if train is not None else "🔴 Thiếu dữ liệu train")

st.markdown("""<div class="top"><h1>Thiết kế cấp phối bê tông UHPC bằng AI</h1>
<p>Đề xuất phối liệu theo mục tiêu cường độ, kiểm soát ràng buộc vật lý và phạm vi dữ liệu, đánh giá sai số từ số liệu thực nghiệm và quy đổi mẻ trộn. Mọi kết quả là <b>dự đoán nghiên cứu</b>, không thay thế thí nghiệm cường độ nén.</p></div>""",unsafe_allow_html=True)

m1,m2,m3,m4=st.columns(4)
m1.metric("Cường độ mục tiêu",f"{target:g} MPa")
m2.metric("Tuổi nén",f"{age} ngày")
m3.metric("Quy mô mẻ",f"{batch_size:g} {batch_mode}")
m4.metric("Dữ liệu huấn luyện",f"{len(train):,} mẫu" if train is not None else "Chưa có")
if model_error: st.error("Không nạp được mô hình thật: "+model_error)
if train_error: st.warning("Không nạp được train: "+train_error)

st.caption("**Phân biệt:** W/B = nước / (xi măng + tro bay + GGBFS + silica fume); W/P* = nước / (4 chất trên + bột thạch anh). Tính năng dự báo luôn giữ W/P* theo notebook 15 đặc trưng.")


def common_payload():
    return dict(age=age,target=target,margin=margin,target_mode=target_mode,
                target_tolerance=target_tolerance,batch_size=batch_size,batch_mode=batch_mode,
                wb_min=wb_min,wb_max=wb_max,steel_limit=steel_limit,air_percent=air_percent,
                vol_tol=vol_tol,cement_max=cement_max,require_fa=require_fa,
                require_ggbfs=require_ggbfs,require_quartz=require_quartz,densities=densities,
                model_version=MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None,
                train_version=TRAIN_PATH.stat().st_mtime if TRAIN_PATH.exists() else None)


def get_fixed(row=None, prefix="fixed"):
    fixed={}
    with st.expander("Thông số bổ sung cho mô hình (4 biến)"):
        st.caption("Các giá trị ban đầu đến từ notebook, không phải kết quả thí nghiệm của vật liệu đang sử dụng.")
        quadrants=st.columns(2)
        readable={"FlyAsh_SiO2":"SiO₂ tro bay (%)", "FlyAsh_Blaine":"Blaine tro bay", "GGBFS_Grade":"Mác GGBFS", "He_so_quy_doi_mau":"Hệ số quy đổi mẫu"}
        for i,k in enumerate(DEFAULT_FIXED):
            with quadrants[i%2]:
                fixed[k]=st.number_input(readable[k],value=float(row[k]) if row is not None else float(DEFAULT_FIXED[k]),key=f"{prefix}_{k}",format="%.3f")
    return fixed


def render_result(result, *, provenance):
    mix=result["mix"]
    prediction=float(result["prediction"])
    q,checks=verify_design(mix,densities,wb_min=wb_min,wb_max=wb_max,air=air_percent/100,volume_tol=vol_tol,
                           steel_limit_percent=steel_limit,require_fa=require_fa,require_ggbfs=require_ggbfs,
                           require_quartz=require_quartz,cement_max=cement_max,target=None)
    alignment=strength_alignment(prediction,target,mode=target_mode,tolerance=target_tolerance,margin=margin)
    checks["Cường độ AI nằm trong khoảng cho phép"]=alignment["meets_policy"]
    proximity=None
    if train is not None:
        proximity=distance_check(feature_frame(pd.DataFrame([mix]),age,result["fixed"],cols),train)
        checks["15 đặc trưng trong min/max train"]=not bool(proximity.outside_features)
        if result.get("enforce_distance",False):checks["Gần train theo chỉ số P95"]=proximity.close
    if result.get("checks"):
        checks.update(result["checks"])
    all_pass=all(checks.values())
    st.subheader("Báo cáo cấp phối")
    st.caption(provenance+" · Kết quả AI, chưa kiểm chứng thực nghiệm độc lập.")
    s1,s2,s3,s4,s5=st.columns(5)
    s1.metric("AI dự báo (không ép số)",f"{prediction:.4f} MPa",f"{alignment['delta_mpa']:+.4f} MPa so với lệnh")
    s2.metric("Độ lệch so với lệnh",f"{alignment['absolute_delta_mpa']:.4f} MPa",f"{alignment['absolute_delta_percent']:.4f}% mục tiêu",delta_color="off")
    s3.metric("W/B",f"{q['W_B']:.3f}")
    s4.metric("W/P* mô hình",f"{q['W_P_notebook']:.3f}")
    s5.metric("Sợi thép",f"{q['steel_vol_percent']:.2f}% thể tích")
    accepted=(f"{alignment['lower_mpa']:.2f}–{alignment['upper_mpa']:.2f} MPa"
              if alignment['upper_mpa'] is not None else f"≥ {alignment['lower_mpa']:.2f} MPa")
    st.caption(f"Khoảng cường độ AI chấp nhận: **{accepted}**. Độ lệch ở đây so với cường độ bạn NHẬP, không phải sai số so với thí nghiệm thực tế.")
    if result.get("candidate_tier")=="safe_outside_target":
        st.warning("Không có nghiệm đúng dung sai trong số ứng viên đã kiểm tra. Đây là cấp phối hợp lệ về ràng buộc được tìm thấy gần mục tiêu nhất, nhưng **không khớp cường độ yêu cầu**.")
    elif result.get("candidate_tier")=="no_safe_candidate":
        st.error("Không tìm thấy ứng viên qua kiểm tra kỹ thuật. Không nên sử dụng cấp phối này.")
    if result.get("candidate_origin"):
        st.caption("Nguồn ứng viên: " + result["candidate_origin"] + ". Giá trị cường độ là dự báo của mô hình, không phải giá trị đo.")
    if alignment["same_to_2_decimals"] and not alignment["exact_numeric_match"]:
        st.caption("Cường độ dự báo trùng mục tiêu khi **làm tròn 2 chữ số**, không phải bằng nhau tuyệt đối.")
    if all_pass:
        if target_mode=="band" and not alignment["meets_minimum"]:
            st.warning("Đã khớp dải ± dung sai và các kiểm tra số học, NHƯNG thấp hơn cường độ nhập; không công nhận đạt điều kiện cường độ tối thiểu.")
        elif target_mode=="minimum_only" and alignment["absolute_delta_mpa"]>target_tolerance:
            st.warning("Đạt cường độ tối thiểu theo dự báo nhưng chưa sát trong dung sai đang nhập; hãy chọn 'Sát mục tiêu, không thấp hơn' nếu muốn giới hạn phần vượt.")
        else:
            st.success("Dự báo AI đạt khoảng mục tiêu và các điều kiện số học đã chọn. Chưa có xác nhận cường độ thực nghiệm.")
    else:
        st.error("Kết quả CHƯA khớp đủ điều kiện cường độ hoặc kỹ thuật đã chọn. Không được công nhận là cấp phối đạt.")
    c1,c2=st.columns([1.15,.85])
    with c1:
        table=batch_table(mix,batch_mode,batch_size)
        st.dataframe(table.style.format({"kg/m³":"{:,.2f}","Mẻ trộn (kg)":"{:,.3f}","Tỷ lệ khối lượng (%)":"{:.2f}"}),hide_index=True,use_container_width=True)
        st.caption(f"Tổng mẻ = {table['Mẻ trộn (kg)'].sum():,.3f} kg; tổng định lượng = {q['mass_kg_m3']:,.1f} kg/m³; thể tích tuyệt đối = {q['absolute_volume']:.4f} m³/m³ (không kể khí).")
    with c2:
        fig=px.pie(table,values="Mẻ trộn (kg)",names="Vật liệu",hole=.55)
        fig.update_layout(height=320,margin=dict(l=2,r=2,t=5,b=5),legend=dict(orientation="h",y=-.2),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("**Bảng kiểm kỹ thuật**")
    check_df=pd.DataFrame([{"Điều kiện":k,"Kết quả":"Đạt theo số học" if v else "KHÔNG ĐẠT"} for k,v in checks.items()])
    st.dataframe(check_df,hide_index=True,use_container_width=True)
    if result.get("group_proximity"):
        gp=result["group_proximity"]
        st.caption(f"Khoảng cách train cùng tuổi/tổ hợp: {gp.near_distance:.3f}; P95 nhóm {gp.train_p95:.3f}.")
    if proximity:
        st.caption(f"Khoảng cách NN đã chuẩn hóa: {proximity.near_distance:.3f}; mốc P95 train: {proximity.train_p95:.3f}. Có {proximity.age_matches} dòng train tại đúng {age} ngày. P95 là ngưỡng khoảng cách, không phải xác suất tin cậy 95%.")
        if proximity.outside_features: st.warning("Ngoài biên đặc trưng: "+", ".join(proximity.outside_features))
        if not proximity.close: st.warning("Cấp phối ngoài vùng gần dữ liệu huấn luyện theo phép đo khoảng cách; độ tin cậy có thể thấp.")
    else: st.warning("Chưa có dữ liệu train để kiểm tra ngoại suy.")
    if result.get("optimizer_success") is False:
        st.warning("Bộ giải không xác nhận hội tụ. Ứng viên được tìm thấy không phải nghiệm tối ưu toàn cục được chứng minh.")
    out=table.copy()
    out.loc[len(out)] = ["TỔNG CỘNG",out["kg/m³"].sum(),out["Mẻ trộn (kg)"].sum(),100.]
    export={"origin":provenance,"age_days":int(age),"target_mpa":float(target),"predicted_mpa":prediction,
            "strength_alignment":alignment,"validated_in_lab":False,
            "pass_numerical_checks":bool(all_pass),"checks":checks,"physical_quantities":q,"mix_kg_m3":mix,
            "model_features":list(cols),"proximity_p95_is_not_confidence":True}
    a,b=st.columns(2)
    a.download_button("⬇ Tải cấp phối CSV",out.to_csv(index=False).encode("utf-8-sig"),"UHPC_cap_phoi.csv", "text/csv",use_container_width=True)
    b.download_button("⬇ Tải báo cáo JSON",json.dumps(export,ensure_ascii=False,indent=2).encode("utf-8"),"UHPC_bao_cao.json","application/json",use_container_width=True)
    st.caption("Định lượng chưa tự điều chỉnh độ ẩm cốt liệu, nước thực hữu hiệu, hao hụt mẻ, độ chảy, chất lượng sợi và điều kiện dưỡng hộ.")


def render_if_fresh(name, fingerprint, provenance):
    record=st.session_state.get(name)
    if not record:return
    if record["fingerprint"] != fingerprint:
        st.info("Thông số đã thay đổi sau lần tính trước. Kết quả cũ được ẩn; vui lòng chạy lại.")
    else:
        render_result(record["data"],provenance=provenance)


t1,t2,t3,t4,t5=st.tabs(["🎯 Thiết kế ngược","🧪 Dự báo cấp phối","📊 Dữ liệu & kiểm định","💰 Dự toán vật liệu","📖 Phương pháp"])

with t1:
    st.subheader("Thiết kế theo cường độ nén mục tiêu")
    st.write("Thuật toán xem xét cấp phối từ dữ liệu huấn luyện và tìm kiếm bằng **Differential Evolution**, trong dải P5–P95 vật liệu cùng tuổi. Khi không có nghiệm chính xác, hiển thị **gần nhất tìm được** có qua kiểm tra kỹ thuật, không gán cường độ bằng giá trị nhập.")
    st.info("Nhập 120 MPa, mặc định dung sai 0,01 MPa: app cố tìm 120,0000–120,0100 MPa và ưu tiên gần 120,0000 nhất. Nếu mô hình không dự báo được trong khoảng này, app hiển thị phương án hợp lệ gần nhất tìm được và báo KHÔNG KHỚP; tuyệt đối không đổi số dự báo thành 120.")
    with st.form("opt_form"):
        o1,o2=st.columns(2)
        with o1:
            opt_mode=st.selectbox("Mục tiêu tối ưu",["Dự đoán sát cường độ nhập","Giảm xi măng và sợi","Giảm chi phí vật liệu"],index=0,
                                  help="Ưu tiên độ lệch giữa cường độ dự báo và cường độ nhập; xi măng chỉ là yếu tố phụ khi cùng độ lệch.")
            strict=st.checkbox("Bắt buộc gần dữ liệu train (P95 NN)",value=True)
            min_present=st.number_input("Vật liệu bắt buộc: lượng tối thiểu (kg/m³)",min_value=.1,max_value=100.,value=5.,step=1.,help="Ngưỡng nghiên cứu để tránh nghiệm vật liệu > 0 nhưng gần bằng 0; không phải quy phạm.")
        with o2:
            st.caption("Thuộc tính FA/GGBFS khi tối ưu: cần sử dụng giá trị đúng vật liệu thực tế.")
            f_opt={}
            for k in DEFAULT_FIXED:
                f_opt[k]=st.number_input(k,value=float(DEFAULT_FIXED[k]),key="opt_"+k)
            search_depth=st.selectbox("Mức tìm kiếm",["Chuyên sâu (ưu tiên bám sát)","Nhanh (tham khảo)"],index=0,
                                      help="Chuyên sâu tăng vòng lặp và quần thể, có thể tìm nghiệm gần mục tiêu hơn nhưng mất thời gian hơn.")
            seed=st.number_input("Seed tái lập tối ưu",min_value=0,max_value=1000000,value=42)
        st.caption("Nếu tối ưu chi phí, hãy nhập giá cho toàn bộ vật liệu có thể xuất hiện trong miền tìm kiếm, ở tab Dự toán.")
        go_opt=st.form_submit_button("TÌM CẤP PHỐI PHÙ HỢP",type="primary",disabled=model is None or train is None,use_container_width=True)
    price_raw={k:st.session_state.get("unitprice_"+k) for k in MATERIALS}
    prices_opt={k:(float(v) if v is not None else float("nan")) for k,v in price_raw.items()}
    opt_params={**common_payload(),"opt_mode":opt_mode,"strict":strict,"min_present":min_present,"seed":seed,"f_opt":f_opt,
                "prices": prices_opt if opt_mode=="Giảm chi phí vật liệu" else None,"search_depth":search_depth}
    opt_sig=request_id(opt_params)
    if go_opt:
        try:
            result=optimize(model,cols,train,age,f_opt,densities,target,wb_min=wb_min,wb_max=wb_max,
                           air=air_percent/100,volume_tol=vol_tol,steel_limit_percent=steel_limit,
                           require_fa=require_fa,require_ggbfs=require_ggbfs,require_quartz=require_quartz,
                           cement_max=cement_max,min_present_kg=min_present,margin=margin,use_distance=strict,
                           target_mode=target_mode,target_tolerance=target_tolerance,
                           objective_mode={"Giảm xi măng và sợi":"cement","Dự đoán sát cường độ nhập":"near_target","Giảm chi phí vật liệu":"cost"}[opt_mode],
                           prices=prices_opt,seed=seed,
                           maxiter=75 if search_depth.startswith("Chuyên sâu") else 38,
                           popsize=10 if search_depth.startswith("Chuyên sâu") else 7)
            result["fixed"]=f_opt
            result["enforce_distance"]=strict
            st.session_state["opt_result"]={"fingerprint":opt_sig,"data":result}
        except Exception as exc:
            st.session_state.pop("opt_result",None)
            st.error(f"Không thể đề xuất cấp phối: {exc}")
    render_if_fresh("opt_result",opt_sig,"Tối ưu DE · Phương án số, chưa được thử nghiệm")

with t2:
    st.subheader("Dự báo cường độ từ cấp phối nhập tay / dữ liệu gốc")
    preset_row=None
    if train is not None:
        eligible=train[np.isclose(train["Tuoi_duong_ho"],age)].reset_index(drop=True)
        choices=["Nhập thủ công"]+[f"TRAIN_{i+1:05d}" for i in range(len(eligible))]
        preset=st.selectbox("Chọn cấp phối từ train (không có cường độ đo trong bảng này)",choices)
        if preset!="Nhập thủ công":preset_row=eligible.iloc[int(preset.split("_")[1])-1]
    else:preset="Nhập thủ công"
    with st.form("pred_form"):
        cols_form=st.columns(3)
        mix={}
        for i,k in enumerate(MATERIALS):
            with cols_form[i%3]:
                mix[k]=st.number_input(LABELS.get(k,k)+" (kg/m³)",value=float(preset_row[k]) if preset_row is not None else None,min_value=0.,step=5.,key=f"input_{preset}_{k}",placeholder="Nhập giá trị")
        fixed_pred={}
        with st.expander("4 biến hóa lý bổ sung",expanded=True):
            fcols=st.columns(4)
            for i,k in enumerate(DEFAULT_FIXED):
                with fcols[i]:fixed_pred[k]=st.number_input(k,value=float(preset_row[k]) if preset_row is not None else float(DEFAULT_FIXED[k]),key=f"chem_{preset}_{k}")
        run_pred=st.form_submit_button("DỰ ĐOÁN VÀ KIỂM TRA",type="primary",disabled=model is None,use_container_width=True)
    input_sig=request_id({**common_payload(),"mix":mix,"fixed_pred":fixed_pred,"preset":preset})
    if run_pred:
        try:
            if any(x is None for x in mix.values()):raise AuditError("Hãy nhập đủ 9 vật liệu.")
            v=check_mix(mix)
            strength=predict(model,cols,v,age,fixed_pred)
            st.session_state["pred_result"]={"fingerprint":input_sig,"data":{"mix":v,"prediction":strength,"fixed":fixed_pred}}
        except Exception as exc:
            st.session_state.pop("pred_result",None)
            st.error(f"Không dự đoán được: {exc}")
    render_if_fresh("pred_result",input_sig,"Dự báo AI trực tiếp · "+preset)
    result=st.session_state.get("pred_result")
    if result and result["fingerprint"]==input_sig and model is not None:
        with st.expander("Biểu đồ thay đổi từng vật liệu ±5% (khảo sát một biến, không chứng minh nhân quả)"):
            base=result["data"]["mix"]
            fixed=result["data"]["fixed"]
            rows=[]
            for k in MATERIALS:
                for percent in [-5,5]:
                    v=base.copy();v[k]*=(1+percent/100)
                    try:y=predict(model,cols,v,age,fixed);rows.append({"Vật liệu":LABELS.get(k,k),"Biến đổi":f"{percent:+d}%", "Chênh lệch dự đoán (MPa)":y-result["data"]["prediction"]})
                    except Exception:pass
            if rows:
                df_sens=pd.DataFrame(rows)
                st.plotly_chart(px.bar(df_sens,x="Vật liệu",y="Chênh lệch dự đoán (MPa)",color="Biến đổi",barmode="group"),use_container_width=True)
            st.caption("Khảo sát chỉ thay một định lượng kg/m³, chưa hiệu chỉnh lại thể tích. Kết quả có thể vi phạm điều kiện cấp phối và không thể coi là ảnh hưởng nhân quả.")

with t3:
    st.subheader("Dữ liệu đầu vào, đánh giá ngoại suy và độ chính xác")
    if train is not None:
        a,b,c=st.columns(3)
        a.metric("Số dòng train",f"{len(train):,}")
        b.metric("Số mẫu đúng tuổi",int(np.isclose(train["Tuoi_duong_ho"],age).sum()))
        c.metric("Số mẫu có cả FA + GGBFS",int(((train["Tro_bay"]>0)&(train["Xil_lo_cao"]>0)).sum()))
        st.caption("Đây là **dữ liệu đặc trưng train**; nếu không có nhãn cường độ đo được, không thể từ riêng dữ liệu này tính R²/RMSE và không thể xác nhận nguồn thí nghiệm.")
        stats=train[MATERIALS].agg(["min","median","max"]).T.reset_index().rename(columns={"index":"Vật liệu"})
        st.dataframe(stats,use_container_width=True,hide_index=True)
        cnt=train["Tuoi_duong_ho"].value_counts().sort_index().reset_index()
        cnt.columns=["Tuổi (ngày)","Số mẫu"]
        st.plotly_chart(px.bar(cnt,x="Tuổi (ngày)",y="Số mẫu",title="Phân bố tuổi nén trong tập train"),use_container_width=True)
        st.dataframe(train.head(150),hide_index=True,use_container_width=True)
        st.download_button("⬇ Xuất bảng thống kê vật liệu",stats.to_csv(index=False).encode("utf-8-sig"),"UHPC_training_summary.csv","text/csv")
    if model is not None:
        with st.expander("Mức quan trọng đặc trưng (không phải quan hệ nhân quả)",expanded=False):
            fi=None;how=""
            try:
                booster=getattr(model,"booster_",None)
                if booster is not None:
                    fi=np.asarray(booster.feature_importance(importance_type="gain"),dtype=float);how="LightGBM gain"
                elif hasattr(model,"feature_importances_"):
                    fi=np.asarray(model.feature_importances_,dtype=float);how="impurity-based"
                if fi is not None and len(fi)==len(cols):
                    vals=pd.DataFrame({"Tên đặc trưng":cols,"Điểm":fi}).sort_values("Điểm")
                    st.plotly_chart(px.bar(vals,x="Điểm",y="Tên đặc trưng",orientation="h",title=how),use_container_width=True)
                    st.caption("Giá trị phụ thuộc thuật toán, các biến tương quan và dữ liệu huấn luyện; không kết luận vật liệu nào tạo ra cường độ cao hơn theo quan hệ nhân quả.")
                else:st.info("Mô hình chưa cung cấp feature importance tương thích.")
            except Exception as exc:st.warning("Không đọc được feature importance: "+str(exc))
    st.markdown("#### Kiểm định mô hình bằng tệp CSV có cường độ đo")
    st.caption("CSV phải có đủ 15 đặc trưng (tên giống trong notebook) và cột `Cuong_do_nen_MPa` hoặc tên tương đương. Chỉ tải dữ liệu do bạn có quyền sử dụng. Nguồn kiểm định cần **độc lập** với dữ liệu huấn luyện để đánh giá khách quan.")
    validation_window=st.number_input("Xem sai số gần cường độ nhập: nửa độ rộng vùng (MPa)",min_value=0.,max_value=100.,value=10.,step=2.,
                                      help="Ví dụ 120 ± 10 MPa: chỉ so sánh mẫu thực nghiệm đo từ 110 đến 130 MPa ở đúng tuổi nén đang chọn.")
    validation_upload=st.file_uploader("Tải CSV kiểm định thực nghiệm",type=["csv"],key="validation_csv")
    if validation_upload and model is not None:
        try:
            validation_df=pd.read_csv(validation_upload,encoding="utf-8-sig")
            met,res=evaluate_validation(model,cols,validation_df,train=train)
            v1,v2,v3=st.columns(3)
            v1.metric("MAE (MPa)",f"{met['MAE']:.2f}")
            v2.metric("RMSE (MPa)",f"{met['RMSE']:.2f}")
            v3.metric("R²",f"{met['R²']:.3f}")
            st.caption(f"Đánh giá trên {met['n']} hàng CSV tải lên; app KHÔNG tự xác nhận CSV này độc lập với train. Không suy diễn RMSE là biên an toàn của một cấp phối.")
            if met.get("exact_train_overlap",0):
                st.error(f"Phát hiện {met['exact_train_overlap']} hàng kiểm định trùng chính xác 15 đặc trưng với dữ liệu train. Không nên coi các chỉ số này là kiểm định độc lập.")
            fig=px.scatter(res,x="Đo thực nghiệm (MPa)",y="AI dự báo (MPa)",hover_data=["Tuổi nén (ngày)"])
            a=min(res.iloc[:,0].min(),res.iloc[:,1].min());b=max(res.iloc[:,0].max(),res.iloc[:,1].max())
            fig.add_shape(type="line",x0=a,y0=a,x1=b,y1=b,line=dict(dash="dash",color="gray"))
            st.plotly_chart(fig,use_container_width=True)
            near=res[np.isclose(res["Tuổi nén (ngày)"],age)]
            if len(near)>=3:
                st.caption(f"Nhóm đúng tuổi {age} ngày: n={len(near)}, MAE={np.mean(np.abs(near['Sai số (MPa)'])):.2f} MPa. Kiểm định riêng vùng gần cường độ bạn nhập nằm bên dưới.")
            region=validation_near_target(res,target,age,validation_window,target_tolerance)
            st.markdown("#### Kiểm định riêng quanh cường độ bạn nhập")
            if region is None:
                st.warning(f"CSV chưa có mẫu đo thực nghiệm tại {age} ngày trong khoảng {target-validation_window:g}–{target+validation_window:g} MPa.")
            else:
                t1m,t2m,t3m,t4m=st.columns(4)
                t1m.metric("Mẫu thực nghiệm cùng vùng",region["n"])
                t2m.metric("MAE vùng (MPa)",f"{region['mae_mpa']:.2f}")
                t3m.metric("RMSE vùng (MPa)",f"{region['rmse_mpa']:.2f}")
                t4m.metric("Sai lệch TB có dấu",f"{region['bias_mpa']:+.2f} MPa")
                st.caption(f"Trong {region['n']} mẫu đo ở {age} ngày gần {target:g} MPa: {region['within_tolerance_count']} mẫu có |AI − đo| ≤ {target_tolerance:g} MPa ({region['within_tolerance_percent']:.1f}%). Đây chỉ là thống kê trên CSV, không bảo đảm cấp phối mới chính xác trong dung sai.")
                if region["n"] < 10:st.warning("Vùng mục tiêu có ít hơn 10 mẫu; thống kê rất kém ổn định.")
            st.download_button("⬇ Bảng sai số kiểm định",res.to_csv(index=False).encode("utf-8-sig"),"UHPC_kiem_dinh_sai_so.csv","text/csv")
        except Exception as exc:st.error("Không thể kiểm định CSV: "+str(exc))
    else:
        st.info("Không có số liệu cường độ đo độc lập: chưa thể kết luận độ chính xác ứng dụng.")

with t4:
    st.subheader("Dự toán chi phí nguyên vật liệu")
    st.caption("Đơn giá VND/kg do bạn nhập, không tự lấy hoặc giả định giá thị trường. `0` là giá miễn phí thực sự; để trống tức chưa có giá.")
    bcols=st.columns(3)
    prices={}
    for i,k in enumerate(MATERIALS):
        with bcols[i%3]:prices[k]=st.number_input(LABELS.get(k,k)+" (VND/kg)",min_value=0.,value=None,step=100.,key="unitprice_"+k,placeholder="Chưa có giá")
    available=[]
    for key,label in [("opt_result","Kết quả thiết kế ngược"),("pred_result","Kết quả dự báo cấp phối")]:
        r=st.session_state.get(key)
        if r:available.append((key,label,r))
    if available:
        choice=st.selectbox("Nguồn cấp phối để tính tiền",[x[1] for x in available])
        selected=next(x for x in available if x[1]==choice)
        try:
            # Re-check result currency: never estimate from stale scenario values.
            fresh = (selected[0]=="opt_result" and selected[2]["fingerprint"]==opt_sig) or (selected[0]=="pred_result" and selected[2]["fingerprint"]==input_sig)
            if not fresh:st.warning("Dữ liệu cấp phối đã lỗi thời do thông số thay đổi. Tính lại ở tab trước.")
            else:
                costs=cost_table(selected[2]["data"]["mix"],prices,mode=batch_mode,amount=batch_size)
                total=costs["Thành tiền (VND)"].sum()
                st.metric("Chi phí vật liệu của mẻ",f"{total:,.0f} VND")
                st.dataframe(costs.style.format({"Khối lượng (kg)":"{:,.3f}","Đơn giá (VND/kg)":"{:,.0f}","Thành tiền (VND)":"{:,.0f}"}),hide_index=True,use_container_width=True)
                st.download_button("⬇ Chi phí CSV",costs.to_csv(index=False).encode("utf-8-sig"),"UHPC_chi_phi.csv","text/csv")
                st.caption("Không bao gồm vận chuyển, nhân công, máy trộn, hao hụt, thuế, thí nghiệm, bảo dưỡng và lợi nhuận.")
        except Exception as exc:st.warning(str(exc))
    else:st.info("Hãy tính một cấp phối ở tab Thiết kế ngược hoặc Dự báo trước.")

with t5:
    st.subheader("Quy trình và giới hạn nghiên cứu")
    st.markdown("""
**1. Nguồn dữ liệu:** mô hình được nạp từ `assets/uhpc_model.pkl`; tập huấn luyện đặc trưng từ `assets/training_features.csv`. Không tạo model giả, không tự huấn luyện và không sinh mẫu thực nghiệm mới.

**2. Tính đúng 15 đặc trưng:** tỷ lệ `Ty_le_W_B` theo notebook gốc bằng nước chia tổng năm thành phần bột (gồm cả bột thạch anh). Đây là W/P*, khác W/B truyền thống không tính bột thạch anh.

**3. Thiết kế ngược:** tối ưu tiến hóa vi phân trong khoảng P5–P95 từng lượng vật liệu. Chế độ mặc định ưu tiên giảm độ lệch |AI − cường độ nhập|; trong chế độ “sát, không thấp hơn” phải thỏa thêm AI ∈ [mục tiêu + dự phòng, mục tiêu + dự phòng + dung sai]. Một nghiệm thỏa đại số vẫn có thể không đạt khi thí nghiệm.

**4. Kiểm soát ngoại suy:** khoảng cách KNN trên 15 đặc trưng được chuẩn hóa, mốc P95 tính trên khoảng cách điểm train tới hàng train khác gần nhất. **Không** diễn giải P95 là xác suất 95% cường độ đạt.

**5. Vật liệu thực:** khối lượng riêng đang là đầu vào tham khảo; cần có phiếu vật liệu để hiệu chỉnh cát ẩm, nước tự do, độ linh động, đặc trưng sợi và độ hút nước. Mô hình không có biến rõ ràng mô tả đầy đủ chế độ dưỡng hộ, loại mẫu, độ chảy và nguồn cốt liệu.

**6. Kiểm định:** CSV có cường độ đo giúp tính sai số toàn bộ và vùng gần mục tiêu (cùng tuổi, đo thực nghiệm trong khoảng nhập). Không tự chứng minh dữ liệu độc lập; rò rỉ theo nguồn bài báo vẫn có thể tồn tại.

**7. Xuất kết quả:** CSV là bảng định lượng; JSON chứa sai lệch MPa, % và trạng thái khớp khoảng cường độ AI; **không phải chứng chỉ nghiệm thu** hay bằng chứng mẫu thực đạt mục tiêu.
""")
    st.markdown("**Đường dẫn dữ liệu phần mềm đang sử dụng**")
    st.code(f"MODEL: {MODEL_PATH.relative_to(ROOT)}\nTRAIN: {TRAIN_PATH.relative_to(ROOT)}",language="text")
    if metadata:
        st.markdown("**Thông tin kèm mô hình, nếu có**")
        st.json({k:str(v)[:1800] for k,v in metadata.items()})
    st.caption("Phiên bản nâng cấp độc lập; chưa cập nhật trực tiếp GitHub và Streamlit của bạn. Mục đích: nghiên cứu cấp phối, không thay nghiệm thu vật liệu.")

st.divider()
st.caption("UHPC AI Studio 2.2 · Đề xuất bằng AI, không tạo số liệu giả · Kiểm chứng thí nghiệm bắt buộc trước áp dụng")
