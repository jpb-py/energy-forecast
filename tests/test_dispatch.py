import numpy as np
import pandas as pd
import pytest

from energy_forecast.config import BatteryParams
from energy_forecast.dispatch import (
    realized_profit,
    run_multi_day_dispatch,
    solve_dispatch,
)


def _synthetic_day_demand(params: BatteryParams) -> pd.Series:
    periods = np.arange(48)
    prices = pd.Series(20 + 30 * np.exp(-((periods - 14) ** 2) / 10) + 80 * np.exp(-((periods - 34) ** 2) / 10))
    return prices / params.k


def test_solve_dispatch_soc_feasibility():
    params = BatteryParams()

    # Load in parameters from Batteryparams
    c_max, d_max, s_max = params.c_max, params.d_max, params.s_max
    eta, t, k = params.eta, params.t, params.k
    s_start, s_end = params.s_start, params.s_end

    periods = np.arange(48)
    prices = pd.Series(20 + 30 * np.exp(-((periods - 14) ** 2) / 10) + 80 * np.exp(-((periods - 34) ** 2) / 10))
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


def test_run_multi_day_dispatch_single_day_matches_solve_dispatch():
    """Regression check for the dispatch relaxation: solve_dispatch itself is
    unchanged, and run_multi_day_dispatch must reduce to it exactly for n=48."""
    params = BatteryParams()
    demand = _synthetic_day_demand(params)

    direct = solve_dispatch(demand, params)
    via_wrapper = run_multi_day_dispatch(demand, params)

    pd.testing.assert_frame_equal(direct, via_wrapper)


def test_run_multi_day_dispatch_two_days_are_solved_independently():
    # s_start != s_end so a continuation bug (carrying SoC across midnight)
    # would be visible as a mismatch between day 1 and day 2's solutions.
    params = BatteryParams(s_start=0.3, s_end=0.7)
    one_day = _synthetic_day_demand(params)
    demand = pd.concat([one_day, one_day], ignore_index=True)

    result = run_multi_day_dispatch(demand, params)

    assert len(result) == 96
    day_one = result.iloc[:48].reset_index(drop=True)
    day_two = result.iloc[48:].reset_index(drop=True)
    pd.testing.assert_frame_equal(day_one, day_two, atol=1e-4)


def test_run_multi_day_dispatch_total_profit_is_sum_of_daily_profits():
    params = BatteryParams()
    one_day = _synthetic_day_demand(params)
    demand = pd.concat([one_day, one_day, one_day], ignore_index=True)

    combined = run_multi_day_dispatch(demand, params)
    single_day = solve_dispatch(one_day, params)

    assert combined["profit per period"].sum() == pytest.approx(
        3 * single_day["profit per period"].sum(), abs=1e-4
    )


def test_run_multi_day_dispatch_rejects_non_multiple_of_48():
    params = BatteryParams()
    demand = pd.Series(np.arange(50, dtype=float))

    with pytest.raises(AssertionError):
        run_multi_day_dispatch(demand, params)


def test_realized_profit_matches_schedules_own_profit_when_demand_is_unchanged():
    # Regression case: re-pricing a schedule against the exact demand it was
    # optimised against must reduce to its own reported 'profit per period'.
    params = BatteryParams()
    demand = _synthetic_day_demand(params)
    schedule = solve_dispatch(demand, params)

    realized = realized_profit(schedule, demand, params)

    np.testing.assert_allclose(realized.to_numpy(), schedule["profit per period"].to_numpy(), atol=1e-8)


def test_realized_profit_reprices_against_different_actual_demand():
    params = BatteryParams()
    demand = _synthetic_day_demand(params)
    schedule = solve_dispatch(demand, params)

    actual_demand = pd.Series(500.0, index=demand.index)  # flat, unrelated to the forecast used

    realized = realized_profit(schedule, actual_demand, params)

    expected = params.t * (actual_demand * params.k) * (schedule["Discharge rate"] - schedule["Charge rate"])
    np.testing.assert_allclose(realized.to_numpy(), expected.to_numpy(), atol=1e-8)
    # a flat actual price removes the arbitrage spread the schedule was built around,
    # so realized profit should come in below what the optimiser believed it would earn
    assert realized.sum() < schedule["profit per period"].sum()
