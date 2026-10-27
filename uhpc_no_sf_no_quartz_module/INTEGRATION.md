IMPORTANT: Both silica_fume and quartz_powder must have Bounds(0, 0). Do not remove their model features if the trained model expects them. Re-evaluate W/B and feature engineering for the zero-quartz case.

# UHPC No-Silica-Fume patch

This is a **safe additive module**, not a replacement for the GitHub application's existing trained-model interface. The repository contains `uhpc_model.pkl`, `uhpc_core.py`, and `uhpc_quality.py`. Integration must map canonical names to the existing model's exact feature schema and preprocessing. Do **not** retrain or drop the SF column just because SF is fixed to zero.

1. Add `uhpc_no_sf.py` beside `app.py`.
2. Inspect the existing `uhpc_core.py` prediction function and create a `predict(mix)` adapter which converts the canonical `mix` dict to the exact model features (including any derived W/B features and curing labels) before invoking the existing predictor. Do not use `model.predict()` on an unordered list.
3. Define `Bounds` for all nine material fields from observed SF=0 training samples or documented engineering limits. Fix `silica_fume` to `Bounds(0, 0)`. Avoid invented bounds.
4. Pass a `validate(mix)` callback enforcing verified water/binder, constituent, density, curing, and domain restrictions. Check that the trained model has support for SF=0; otherwise warn that predictions extrapolate.
5. Run `optimize_no_sf_no_quartz(predict, bounds, target_mpa=120, batch_kg=50, validate=validate)` and display results only if the return is non-None.
6. `batch_kg` scales the predicted mixture by the sum of modeled material masses; it does not imply an exact 50 kg/m3 density or correct for moisture, entrained air, or yield.

Example Streamlit controls:

```python
import streamlit as st
from uhpc_no_sf import optimize_no_sf, Bounds

st.title('UHPC AI Mix Design | Không Silica Fume')
target = st.number_input('Cường độ mục tiêu (MPa), 28 ngày', min_value=1.0, value=120.0)
batch = st.number_input('Khối lượng mẻ (kg)', min_value=0.1, value=50.0)
st.metric('Silica Fume (kg/m³)', '0')
# Supply actual trained-model `predict`, actual bounds, and actual validator.
# Never replace the callback with a mock or a hardcoded prediction.
if st.button('Tối ưu cấp phối'):
    if not all(name in globals() for name in ('predict', 'bounds', 'validate')):
        st.error('Chưa kết nối mô hình AI, giới hạn dữ liệu và kiểm tra hợp lệ.')
    else:
        result = optimize_no_sf_no_quartz(predict, bounds, target, batch, validate=validate)
        if result is None:
            st.warning('Không tìm thấy cấp phối thỏa mục tiêu trong miền tìm kiếm.')
        else:
            st.metric('Cường độ dự đoán (MPa)', f"{result['predicted_mpa']:.2f}")
            st.dataframe({'Vật liệu': list(result['mix_kg_m3']),
                          'kg/m³': list(result['mix_kg_m3'].values()),
                          'kg/mẻ': list(result['batch_kg'].values())})
            st.warning('Chỉ là dự đoán mô hình; cần kiểm chứng thực nghiệm.')
```

Note: a simple random search is used as a transparent baseline, not a claim of global optimality. The optimization objective is minimum predicted strength surplus above target; costs and workability are not optimized.
