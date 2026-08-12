from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryParams:
    c_max: float = 0.5
    d_max: float = 0.5
    s_max: float = 1.0
    eta: float = 0.9
    t: float = 0.5
    k: float = 0.003
    s_start: float = 0.5
    s_end: float = 0.5


@dataclass(frozen=True)
class HierarchicalSigmaPriors:
    n_hour_buckets: int = 24
    alpha_mu: float = 6.03  # log(414): pooled train-residual std, MW
    alpha_sigma: float = 0.5
    tau_sigma: float = 0.5
    interval_low: float = 0.025
    interval_high: float = 0.975
    draws: int = 1000
    tune: int = 1000
    chains: int = 4