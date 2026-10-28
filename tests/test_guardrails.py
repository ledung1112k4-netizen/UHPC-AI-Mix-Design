"""Kiểm thử cấu hình sàng lọc và patch; không dùng dữ liệu/cường độ giả như dữ liệu khoa học."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import uhpc_engineering as eng
from ap_dung_sua_loi import update_core, update_app

MATERIALS = list(eng.MATERIAL_LIMITS)
VALID_180 = dict(
    Xi_mang=650.0, Tro_bay=100., Xil_lo_cao=100., Silica_fume=100.,
    Bot_thach_anh=0., Cat=1000., Nuoc=155., Phu_gia_sieu_deo=25., Soi_thep=100.,
)

class TestEngineering(unittest.TestCase):
    def test_180_mpa_cannot_have_60_cement(self):
        mix = {**VALID_180, 'Xi_mang': 60.}
        checks = eng.mix_checks(mix, 180.)
        self.assertFalse(checks['Xi măng đáp ứng ngưỡng sàng lọc nội bộ'])
        mask, penalty = eng.assess_population([[mix[k] for k in MATERIALS]], MATERIALS, 180.)
        self.assertFalse(bool(mask[0]))
        self.assertGreater(float(penalty[0]), 0.)
        self.assertTrue(any('Xi_mang' in s for s in eng.forward_issues(mix)))

    def test_dynamic_min_and_training_intersection(self):
        raw = [(0., 1500.) for _ in MATERIALS]
        limits = eng.tighten_bounds(raw, MATERIALS, 180.)
        self.assertEqual(limits[MATERIALS.index('Xi_mang')], (600., 1000.))
        self.assertEqual(limits[MATERIALS.index('Silica_fume')][0], 0.)
        self.assertEqual(limits[MATERIALS.index('Bot_thach_anh')][0], 0.)
        self.assertEqual(eng.tighten_bounds(raw, MATERIALS, 120.)[0][0], 400.)

    def test_fails_closed_when_training_cannot_support_high_strength_bounds(self):
        raw = [(0., 1500.) for _ in MATERIALS]
        raw[0] = (30., 590.)
        with self.assertRaisesRegex(ValueError, 'Không thể tối ưu'):
            eng.tighten_bounds(raw, MATERIALS, 180.)

    def test_passes_consistent_mix(self):
        mask, penalty = eng.assess_population([[VALID_180[k] for k in MATERIALS]], MATERIALS, 180.)
        self.assertTrue(bool(mask[0]))
        self.assertEqual(float(penalty[0]), 0.)
        self.assertTrue(all(eng.mix_checks(VALID_180, 180.).values()))

    def test_silica_and_quartz_zero_optional(self):
        mix = dict(VALID_180, Silica_fume=0., Bot_thach_anh=0., Xil_lo_cao=200.)
        mask, _ = eng.assess_population([[mix[k] for k in MATERIALS]], MATERIALS, 180.)
        self.assertTrue(bool(mask[0]))

    def test_forward_filters_low_cement(self):
        mix = dict(VALID_180, Xi_mang=60.)
        self.assertTrue(eng.forward_issues(mix))
        self.assertFalse(eng.forward_issues(VALID_180))

    def test_patched_source_core_has_guard_at_three_levels(self):
        baseline = Path('/mnt/data/uhpc_review/uhpc_core.py')
        if not baseline.is_file():
            self.skipTest('Mã tham chiếu bản cũ không có trong môi trường test')
        s = update_core(baseline.read_text(encoding='utf-8'))
        ast.parse(s)
        for text in ('uhpc_eng.tighten_bounds(bds', 'loss = np.asarray(loss',
                     'feasible = np.asarray(feasible', 'physical_checks.update(uhpc_eng.mix_checks'):
            self.assertIn(text, s)

    def test_patcher_rejects_unknown_code(self):
        with self.assertRaises(RuntimeError):
            update_core('def optimize_mix():\n    return True\n')
        with self.assertRaises(RuntimeError):
            update_app('import streamlit as st\n')

if __name__ == '__main__':
    unittest.main()
