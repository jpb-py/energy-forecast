import pandas as pd
import cvxpy as cp
from energy_forecast.config import BatteryParams

def solve_dispatch(demand, params: BatteryParams) -> pd.DataFrame:
    # Function takes in demand and batter parameters and outputs charge/discharge/SoC/cost per period
    
    # Load in parameters from Batteryparams
    c_max, d_max, s_max = params.c_max, params.d_max, params.s_max
    eta, t, k = params.eta, params.t, params.k
    s_start, s_end = params.s_start, params.s_end

    prices = demand*k # Assume price is a scalar multiple of demand for speed and simplicity
    ## Take n as length of prices series 
    n = len(prices)

    # Confirm of length 48
    assert(n) == 48

    # Define variables for optimisation
    s = cp.Variable(n) # SoC [MWh]
    d = cp.Variable(n) # Discharge [MW]
    c = cp.Variable(n) # Charge [MW]

    # Constraints
    constraints = [0 <= c, c <= c_max,
                   0 <= d, d <= d_max,
                   0 <= s, s <= s_max,
                   s[0] == s_start + t*(eta*(c[0] - d[0])),
                   s[n-1] == s_end]
    
    # Add time linking constraints
    constraints.append(
        s[1:n] == s[0:n-1] + t*(eta*c[1:n] - d[1:n])
    )

    # Objectives
    obj = cp.Maximize(t*prices.values@(d-c))

    # Form & solve problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver = cp.HIGHS)

    # Create outputs
    df_output = pd.DataFrame({
        'Charge rate': c.value,
        'Discharge rate': d.value,
        'SoC': s.value,
        'cost per period': t*prices*(d.value-c.value)
    }, index = prices.index)


    return df_output