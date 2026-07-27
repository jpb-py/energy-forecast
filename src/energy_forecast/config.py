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