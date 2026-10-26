"""UHPC AI STUDIO 2.0: no invented reference mixtures, predictions or validation.

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
    load_training, optimize, physical, predict, request_id, verify_design, target_match,
)

ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [ROOT/"assets"/"uhpc_model.pkl", ROOT/"model"/"uhpc_model.pkl"]
TRAIN_CANDIDATES = [ROOT/"assets"/"training_features.csv", ROOT/"data"/"training_features.csv", ROOT/"data"/"uhpc_reference.csv"]
MODEL_PATH = next((x for x in MODEL_CANDIDATES if x.exists()),MODEL_CANDIDATES[0])
TRAIN_PATH = next((x for x in TRAIN_CANDIDATES if x.exists()),TRAIN_CANDIDATES[0])
# Mức dung sai dự báo của chế độ bám sát mục tiêu, KHÔNG phải sai số thí nghiệm.
TARGET_TOLERANCE_MPA = 0.01

st.set_page_config(page_title="UHPC AI Studio 2.0", page_icon="🧱",layout="wide",initial_sidebar_state="expanded")
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
    st.caption("PHÒNG THÍ NGHIỆM THIẾT KẾ CẤP PHỐI · 2.0")
    age=st.number_input("Tuổi nén (ngày)",min_value=1,max_value=365,value=28,step=1)
    target=st.number_input("Cường độ mục tiêu (MPa)",min_value=1.,max_value=300.,value=120.,step=5.)
    margin=st.number_input("Biên dự phòng số học (MPa)",min_value=0.,max_value=50.,value=0.,step=1.,help="Không phải khoảng tin cậy; khi có calibration độc lập, xem riêng phần kiểm định.")
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
    return dict(age=age,target=target,margin=margin,batch_size=batch_size,batch_mode=batch_mode,
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
                           require_quartz=require_quartz,cement_max=cement_max,target=target,predicted=prediction,margin=margin)
    proximity=None
    if train is not None:
        proximity=distance_check(feature_frame(pd.DataFrame([mix]),age,result["fixed"],cols),train)
        checks["15 đặc trưng trong min/max train"]=not bool(proximity.outside_features)
        if result.get("enforce_distance",False):checks["Gần train theo chỉ số P95"]=proximity.close
    if result.get("checks"):
        checks.update(result["checks"])
    all_pass=all(checks.values())
    align=target_match(prediction,target,margin=margin,tolerance=TARGET_TOLERANCE_MPA)
    near_mode=result.get("mode")=="near_target"
    st.subheader("Báo cáo cấp phối")
    st.caption(provenance+" · Kết quả AI, chưa kiểm chứng thực nghiệm độc lập.")
    s1,s2,s3,s4=st.columns(4)
    s1.metric("Cường độ AI dự báo",f"{prediction:.4f} MPa",f"{align['delta_mpa']:+.4f} MPa")
    s2.metric("W/B",f"{q['W_B']:.3f}")
    s3.metric("W/P* của mô hình",f"{q['W_P_notebook']:.3f}")
    s4.metric("Sợi thép",f"{q['steel_vol_percent']:.2f}% thể tích")
    st.caption(f"So với mức nhập {target:g} MPa: sai lệch tuyệt đối {align['absolute_delta_mpa']:.4f} MPa "
               f"({align['absolute_delta_percent']:.3f}%). "
               + (f"Dự báo sát mục tiêu trong khoảng {TARGET_TOLERANCE_MPA:g} MPa phía trên. " if align['near_from_above'] else
                  f"Dự báo chưa sát mục tiêu trong khoảng {TARGET_TOLERANCE_MPA:g} MPa phía trên. ")
               + "Đây là chênh lệch so với **mục tiêu nhập**, KHÔNG phải sai số AI so với cường độ thí nghiệm.")
    if all_pass:
        if near_mode:
            st.success(f"Các ràng buộc số học đạt và AI dự báo nằm trong khoảng mục tiêu + 0 đến +{TARGET_TOLERANCE_MPA:g} MPa. Không phải xác nhận nén thực nghiệm.")
        elif align['near_from_above']:
            st.success("Đạt các điều kiện số học và dự báo gần cường độ mục tiêu. Chưa kiểm chứng nén thực nghiệm.")
        else:
            st.warning("Đạt các điều kiện số học theo dự đoán AI, nhưng không bám sát cường độ nhập. Chưa kiểm chứng thực nghiệm.")
    else:
        st.error("Chưa đáp ứng đủ điều kiện thiết kế số hoặc chưa sát cường độ yêu cầu. Không được coi là cấp phối đã đạt yêu cầu.")
    if result.get("candidate_origin"):
        st.caption("Nguồn ứng viên: " + result["candidate_origin"])
    if near_mode and result.get("candidate_tier") in ("safe_but_not_matching","safe_above_not_matching"):
        st.warning("Thuật toán tìm thấy phương án đáp ứng các kiểm tra kỹ thuật nhưng dự báo chưa khớp khoảng cường độ yêu cầu; không làm tròn hoặc sửa số dự báo thành mục tiêu.")
    elif near_mode and result.get("candidate_tier")=="no_physically_safe_candidate":
        st.error("Thuật toán chưa tìm được phương án thỏa các ràng buộc kỹ thuật. Bảng số chỉ là ứng viên thử, KHÔNG đủ điều kiện đề xuất.")
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
    options = result.get("alternatives") or []
    if result.get("mode") == "near_target" and options:
        with st.expander("So sánh các cấp phối ứng viên gần mục tiêu (AI dự báo)"):
            compare_rows=[]
            for item in options:
                compare_rows.append({
                    "AI dự báo (MPa)":item["prediction_mpa"],
                    "Chênh mục tiêu (MPa)":item["difference_mpa"],
                    "Xi măng (kg/m³)":item["mix"]["Xi_mang"],
                    "Tro bay (kg/m³)":item["mix"]["Tro_bay"],
                    "GGBFS (kg/m³)":item["mix"]["Xil_lo_cao"],
                    "Bột thạch anh (kg/m³)":item["mix"]["Bot_thach_anh"],
                })
            st.dataframe(pd.DataFrame(compare_rows).style.format({
                "AI dự báo (MPa)":"{:.4f}", "Chênh mục tiêu (MPa)":"{:+.4f}",
                "Xi măng (kg/m³)":"{:.1f}","Tro bay (kg/m³)":"{:.1f}",
                "GGBFS (kg/m³)":"{:.1f}","Bột thạch anh (kg/m³)":"{:.1f}"}),
                use_container_width=True,hide_index=True)
            st.caption("Đây là các ứng viên được AI dự báo, qua kiểm tra kỹ thuật số trong lần tìm kiếm. Không phải kết quả nén thực nghiệm hoặc cam kết tìm ra tối ưu toàn cục.")
    out=table.copy()
    out.loc[len(out)] = ["TỔNG CỘNG",out["kg/m³"].sum(),out["Mẻ trộn (kg)"].sum(),100.]
    export={"origin":provenance,"age_days":int(age),"target_mpa":float(target),"predicted_mpa":prediction,"validated_in_lab":False,
            "pass_numerical_checks":bool(all_pass),"checks":checks,"physical_quantities":q,"mix_kg_m3":mix,
            "comparison_to_target":align,"matching_is_not_lab_accuracy":True,
            "alternatives":options,"model_features":list(cols),"proximity_p95_is_not_confidence":True}
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
    st.write("Thuật toán **Differential Evolution** chọn ứng viên trong dải phân vị P5–P95 của dữ liệu thật có cùng tuổi nén; kiểm tra lại ràng buộc sau tối ưu.")
    st.caption(f"Mặc định ưu tiên cường độ AI dự báo gần mức đã nhập nhưng không thấp hơn. Dung sai bám sát: {TARGET_TOLERANCE_MPA:g} MPa; nếu không tìm được, báo 'chưa khớp' thay vì sửa con số dự đoán.")
    with st.form("opt_form"):
        o1,o2=st.columns(2)
        with o1:
            opt_mode=st.selectbox("Mục tiêu tối ưu",["Giảm xi măng và sợi","Dự đoán gần mục tiêu","Giảm chi phí vật liệu"],index=1)
            strict=st.checkbox("Bắt buộc gần dữ liệu train (P95 NN)",value=True)
            min_present=st.number_input("Vật liệu bắt buộc: lượng tối thiểu (kg/m³)",min_value=.1,max_value=100.,value=5.,step=1.,help="Ngưỡng nghiên cứu để tránh nghiệm vật liệu > 0 nhưng gần bằng 0; không phải quy phạm.")
        with o2:
            st.caption("Thuộc tính FA/GGBFS khi tối ưu: cần sử dụng giá trị đúng vật liệu thực tế.")
            f_opt={}
            for k in DEFAULT_FIXED:
                f_opt[k]=st.number_input(k,value=float(DEFAULT_FIXED[k]),key="opt_"+k)
            seed=st.number_input("Seed tái lập tối ưu",min_value=0,max_value=1000000,value=42)
        st.caption("Nếu tối ưu chi phí, hãy nhập giá cho toàn bộ vật liệu có thể xuất hiện trong miền tìm kiếm, ở tab Dự toán.")
        go_opt=st.form_submit_button("TÌM CẤP PHỐI PHÙ HỢP",type="primary",disabled=model is None or train is None,use_container_width=True)
    price_raw={k:st.session_state.get("unitprice_"+k) for k in MATERIALS}
    prices_opt={k:(float(v) if v is not None else float("nan")) for k,v in price_raw.items()}
    opt_params={**common_payload(),"opt_mode":opt_mode,"strict":strict,"min_present":min_present,"seed":seed,"f_opt":f_opt,
                "prices": prices_opt if opt_mode=="Giảm chi phí vật liệu" else None}
    opt_sig=request_id(opt_params)
    if go_opt:
        try:
            result=optimize(model,cols,train,age,f_opt,densities,target,wb_min=wb_min,wb_max=wb_max,
                           air=air_percent/100,volume_tol=vol_tol,steel_limit_percent=steel_limit,
                           require_fa=require_fa,require_ggbfs=require_ggbfs,require_quartz=require_quartz,
                           cement_max=cement_max,min_present_kg=min_present,margin=margin,use_distance=strict,
                           objective_mode={"Giảm xi măng và sợi":"cement","Dự đoán gần mục tiêu":"near_target","Giảm chi phí vật liệu":"cost"}[opt_mode],
                           prices=prices_opt,seed=seed,target_tolerance=TARGET_TOLERANCE_MPA)
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
                st.caption(f"Nhóm {age} ngày: n={len(near)}, MAE={np.mean(np.abs(near['Sai số (MPa)'])):.2f} MPa. Kiểm định riêng vùng ≥{target:g} MPa được khuyến nghị nếu đủ số mẫu.")
            high=near[near["Đo thực nghiệm (MPa)"]>=target]
            if len(high)>=3:
                st.caption(f"Nhóm {age} ngày có cường độ thực nghiệm ≥{target:g} MPa: n={len(high)}, MAE={np.mean(np.abs(high['Sai số (MPa)'])):.2f} MPa.")
            neighborhood=near[(near["Đo thực nghiệm (MPa)"]-target).abs()<=10]
            if len(neighborhood)>=3:
                st.caption(f"Kiểm định gần cường độ bạn nhập ({target:g} ± 10 MPa, {age} ngày): "
                           f"n={len(neighborhood)}, MAE={np.mean(np.abs(neighborhood['Sai số (MPa)'])):.2f} MPa. "
                           "Chỉ đáng tin khi CSV độc lập với tập đã huấn luyện.")
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

**3. Thiết kế ngược:** tối ưu tiến hóa vi phân trong khoảng P5–P95 từng lượng vật liệu, lọc tuổi và tổ hợp được chọn. Một nghiệm thỏa các ràng buộc đại số vẫn có thể không đạt khi thí nghiệm.

**4. Kiểm soát ngoại suy:** khoảng cách KNN trên 15 đặc trưng được chuẩn hóa, mốc P95 tính trên khoảng cách điểm train tới hàng train khác gần nhất. **Không** diễn giải P95 là xác suất 95% cường độ đạt.

**5. Vật liệu thực:** khối lượng riêng đang là đầu vào tham khảo; cần có phiếu vật liệu để hiệu chỉnh cát ẩm, nước tự do, độ linh động, đặc trưng sợi và độ hút nước. Mô hình không có biến rõ ràng mô tả đầy đủ chế độ dưỡng hộ, loại mẫu, độ chảy và nguồn cốt liệu.

**6. Kiểm định:** số liệu CSV tải lên chỉ hỗ trợ kiểm tra sai số của tập được chọn, không tự kiểm tra độc lập hay trùng nguồn. Có thể rò rỉ dữ liệu nếu dùng các hàng từng huấn luyện để kiểm định.

**7. Xuất kết quả:** CSV là bảng định lượng; JSON ghi dự đoán AI và điều kiện số học, **không phải chứng chỉ nghiệm thu** hay chứng minh UHPC đạt 120 MPa.
""")
    st.markdown("**Đường dẫn dữ liệu phần mềm đang sử dụng**")
    st.code(f"MODEL: {MODEL_PATH.relative_to(ROOT)}\nTRAIN: {TRAIN_PATH.relative_to(ROOT)}",language="text")
    if metadata:
        st.markdown("**Thông tin kèm mô hình, nếu có**")
        st.json({k:str(v)[:1800] for k,v in metadata.items()})
    st.caption("Phiên bản nâng cấp độc lập; chưa cập nhật trực tiếp GitHub và Streamlit của bạn. Mục đích: nghiên cứu cấp phối, không thay nghiệm thu vật liệu.")

st.divider()
st.caption("UHPC AI Studio 2.0 · Đề xuất bằng AI, không tạo số liệu giả · Kiểm chứng thí nghiệm bắt buộc trước áp dụng")
