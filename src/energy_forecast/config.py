from dataclasses import dataclass

@dataclass(frozen=True)
class BatteryParams:
    c_max: 0.2
    d_max: 0.2
    s_max: 1
    eta: 0.9
    t: 0.5
    k: 0.003
    s_start: 0.5
    s_end: 0.5