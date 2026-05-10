from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional, Union

# ==========================================
# Environment Models
# ==========================================

@dataclass
class SSPLayer:
    """Sound Speed Profile layer point"""
    depth: float
    c: float
    density: float = 1.0
    attenuation: float = 0.0

@dataclass
class SSPModel:
    """Sound Speed Profile definition"""
    data: List[SSPLayer]
    type: Literal["c-linear", "spline", "n2-linear"] = "spline"

@dataclass
class BoundaryCondition:
    """Boundary condition (e.g., for bottom or surface)"""
    type: Literal["vacuum", "rigid", "acousto-elastic"]
    depth: Optional[float] = None
    c_p: Optional[float] = None
    c_s: Optional[float] = None
    density: Optional[float] = None
    attenuation_p: Optional[float] = 0.0
    attenuation_s: Optional[float] = 0.0

@dataclass
class EnvironmentModel:
    """Complete acoustic environment"""
    ssp: SSPModel
    bottom: BoundaryCondition
    surface: BoundaryCondition = field(default_factory=lambda: BoundaryCondition(type="vacuum", depth=0.0))
    absorption_model: Optional[Literal["thorp"]] = None

# ==========================================
# Geometry Models
# ==========================================

@dataclass
class SourceModel:
    depths: List[float]

@dataclass
class ReceiverModel:
    ranges: List[float]
    depths: List[float]

@dataclass
class GeometryModel:
    """Simulation geometry (Source & Receivers)"""
    source: SourceModel
    receivers: ReceiverModel

# ==========================================
# Solver & Config Models
# ==========================================

@dataclass
class BellhopConfig:
    """Bellhop specific solver configurations"""
    angles: Tuple[float, float]
    num_beams: int = 0
    step_size: float = 0.0
    type: str = "bellhop"

@dataclass
class KrakenConfig:
    """Kraken normal modes specific solver configurations"""
    phase_speed_limits: Tuple[float, float] = (0.0, 1e6)
    mesh_multiplier: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    n_mesh_points: int = 2000
    type: str = "normal_modes"

@dataclass
class SimulationConfig:
    """Root simulation configuration model"""
    project: str
    frequency: float
    environment: EnvironmentModel
    geometry: GeometryModel
    solver: Union[BellhopConfig, KrakenConfig]
