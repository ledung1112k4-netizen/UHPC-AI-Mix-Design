import pandas as pd
from uhpc_core import DesignRules, material_search_bounds, DEFAULT_DENSITIES, FEATURES_15

def test_no_sf_quartz_bounds():
    bounds = {c: (0., 1000.) for c in FEATURES_15}
    rules = DesignRules(silica_mode='off', quartz_mode='off', similarity_policy='report_only')
    result = material_search_bounds(bounds, rules, 900., DEFAULT_DENSITIES['Soi_thep'])
    from uhpc_core import MATERIALS
    assert result[MATERIALS.index('Silica_fume')] == (0., 0.)
    assert result[MATERIALS.index('Bot_thach_anh')] == (0., 0.)
