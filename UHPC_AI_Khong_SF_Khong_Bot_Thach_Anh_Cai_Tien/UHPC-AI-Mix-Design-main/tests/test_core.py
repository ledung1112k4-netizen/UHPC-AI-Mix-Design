import numpy as np
import pandas as pd
import pytest

from uhpc_core import (
    DEFAULT_DENSITIES, DEFAULT_FIXED, FEATURES_15, MATERIALS,
    NOTEBOOK_TEST_MIX, DesignRules, batch_table, feature_frame,
    physical_quantities, predict, optimize_mix,
)


class UnitTestModel:
    """Chỉ dùng để kiểm thử phép gọi model, không được đóng gói cùng app."""
    n_features_in_ = 15
    feature_names_in_ = np.array(FEATURES_15)

    def predict(self, features):
        assert list(features.columns) == FEATURES_15
        return (features["Xi_mang"] * 0.20 + features["Tro_bay"] * 0.10).to_numpy()


def test_model_ratio_matches_original_notebook():
    features = feature_frame(pd.DataFrame([NOTEBOOK_TEST_MIX]), 28, DEFAULT_FIXED, FEATURES_15)
    total_powder = sum(NOTEBOOK_TEST_MIX[k] for k in MATERIALS[:5])
    assert features.iloc[0]["Ty_le_W_B"] == pytest.approx(NOTEBOOK_TEST_MIX["Nuoc"] / total_powder)
    quantities = physical_quantities(pd.DataFrame([NOTEBOOK_TEST_MIX]), DEFAULT_DENSITIES).iloc[0]
    assert quantities["W_B"] > quantities["W_powder"]


def test_batch_50kg_exact_sum():
    table, factor = batch_table(NOTEBOOK_TEST_MIX, "kg", 50.0)
    assert table["Khối lượng mẻ (kg)"].sum() == pytest.approx(50.0)
    assert all(table["Khối lượng mẻ (kg)"] >= 0)


def test_volume_batch():
    table, factor = batch_table(NOTEBOOK_TEST_MIX, "m3", 0.05)
    assert table["Khối lượng mẻ (kg)"].sum() == pytest.approx(sum(NOTEBOOK_TEST_MIX.values())*0.05)


def test_no_fake_prediction():
    with pytest.raises(ValueError):
        predict(UnitTestModel(), {**NOTEBOOK_TEST_MIX, "Nuoc": -10}, 28, DEFAULT_FIXED, FEATURES_15)


def test_fake_model_only_for_unit_test():
    strength = predict(UnitTestModel(), NOTEBOOK_TEST_MIX, 28, DEFAULT_FIXED, FEATURES_15)
    assert strength == pytest.approx(650 * .2 + 120 * .1)


def test_inverse_blocks_age_extrapolation():
    bounds = {k: (0.0, 1500.0) for k in FEATURES_15}
    bounds["Tuoi_duong_ho"] = (7.0, 21.0)
    with pytest.raises(ValueError, match="Tuoi_duong_ho"):
        optimize_mix(UnitTestModel(), FEATURES_15, bounds, 28.0, DEFAULT_FIXED,
                     DEFAULT_DENSITIES, DesignRules(), 120, 900)


def test_inverse_not_falsely_accepted_when_unreachable():
    bounds = {k: (1.0, 1000.0) for k in FEATURES_15}
    bounds["Ty_le_W_B"] = (0.05, 0.5)
    bounds["FlyAsh_SiO2"] = (60.0, 60.0)
    bounds["FlyAsh_Blaine"] = (3500.0, 3500.0)
    bounds["GGBFS_Grade"] = (100.0, 100.0)
    bounds["He_so_quy_doi_mau"] = (1.0, 1.0)
    out = optimize_mix(UnitTestModel(), FEATURES_15, bounds, 28.0, DEFAULT_FIXED,
        DEFAULT_DENSITIES, DesignRules(), target_mpa=10000.0,
        max_cement=900.0, mode="Nhanh")
    assert out["feasible"] is False
    assert out["checks"]["Cường độ AI ≥ mục tiêu + biên dự phòng"] is False


def test_invalid_rules_rejected():
    with pytest.raises(ValueError):
        DesignRules(wb_min=.3, wb_max=.2).validate()
