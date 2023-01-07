from collections import namedtuple
from dataclasses import dataclass

SimulationParams = namedtuple(
    'SimulationParams', 'nqva, nqv, nq, nv, nu, nee, nsim, ntime'
)


@dataclass
class ModelParams:
    nq: int = 0
    nv: int = 0
    nu: int = 0
    nx: int = 0
    nxd: int = 0


@dataclass
class SimulationParams:
    modeel_params: ModelParams = (0, 0, 0, 0, 0)
    nqva: int = 0
    nqv: int = 0
    nq: int = 0
    nv: int = 0
    nu: int = 0
    nsim: int = 0
    ntime: int = 0

