import numpy as np
import pandas as pd
import pytest
from energy_forecast.dispatch import solve_dispatch
from energy_forecast.config import BatteryParams


def test_solve_dispatch_soc_feasibility():
    params = BatteryParams()

    # Load in parameters from Batteryparams
    c_max, d_max, s_max = params.c_max, params.d_max, params.s_max
    eta, t, k = params.eta, params.t, params.k
    s_start, s_end = params.s_start, params.s_end

    periods = np.arange(48)
    prices = pd.Series((20 + 30 * np.exp(-((periods - 14) ** 2) / 10) + 80 * np.exp(-((periods - 34) ** 2) / 10)))
    demand = prices / k

    result = solve_dispatch(demand, params)
    tol = 1e-4

    c = result["Charge rate"].values
    d = result["Discharge rate"].values
    s = result["SoC"].values

    # Start condition
    assert s[0] == pytest.approx(s_start + t*(eta*c[0] - d[0]),abs = tol), "Start condition not obeyed" 

    
    # Storage evolution equation
    # 1. Calculate expected RHS
    s_expected = s[0:-1] + t*(eta*c[1:] - d[1:])
    
    # 2. Assert elementwise relatio
    np.testing.assert_allclose(s[1:], s_expected, atol = tol, err_msg = "Storage evolution not obeyed")

    #  End condition
    assert s[-1] == pytest.approx(s_end, abs=tol), "End condition not obeyed"
    

    # Bounds for s, c, d
    s_clipped = np.clip(s, 0, s_max)
    np.testing.assert_allclose(s, s_clipped, atol = 1e-4, err_msg = "SoC out of bounds")

    c_clipped = np.clip(c, 0, c_max)
    np.testing.assert_allclose(c, c_clipped, atol = 1e-4, err_msg = "Charge rate out of bounds")

    d_clipped = np.clip(d, 0, d_max)
    np.testing.assert_allclose(d, d_clipped, atol = 1e-4, err_msg = "Discharge rate out of bounds")
