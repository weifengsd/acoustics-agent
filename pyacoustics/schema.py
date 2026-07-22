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
    type: Literal["c-linear", "spline", "n2-linear", "quad-2d", "hexahedral-3d"] = "spline"
    ranges: Optional[List[float]] = None
    c_matrix: Optional[List[List[float]]] = None
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None
    z: Optional[List[float]] = None
    c: Optional[List[List[List[float]]]] = None

@dataclass
class Bathymetry3DModel:
    """3D Grid Bathymetry (Depth as a function of X and Y)"""
    x: List[float]
    y: List[float]
    z: List[List[float]]

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
    rms_roughness_m: Optional[float] = 0.0
    bathymetry: Optional[List[Tuple[float, float]]] = None
    bathymetry_file: Optional[str] = None
    bathymetry_interpolation: Optional[Literal["L", "C"]] = None
    bathymetry_geoacoustics: Optional[List[Tuple[float, float, float, float, float]]] = None
    altimetry: Optional[List[Tuple[float, float]]] = None
    altimetry_file: Optional[str] = None
    altimetry_interpolation: Optional[Literal["L", "C"]] = None
    altimetry_geoacoustics: Optional[List[Tuple[float, float, float, float, float]]] = None
    reflection_table: Optional[List[Tuple[float, float, float]]] = None
    reflection_file: Optional[str] = None
    bathymetry_3d: Optional[Bathymetry3DModel] = None

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
    x: Optional[List[float]] = None
    y: Optional[List[float]] = None

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
    beam_type: Literal[
        "default",
        "geometric_hat",
        "geometric_gaussian",
        "simple_gaussian",
        "geometric_hat_ray_centered",
        "geometric_gaussian_ray_centered",
    ] = "default"
    beam_displacement: Literal["none", "seongil"] = "none"
    source_type: Literal["point", "line"] = "point"
    receiver_grid: Literal["rectilinear", "irregular"] = "rectilinear"
    type: str = "bellhop"

@dataclass
class KrakenConfig:
    """Kraken normal modes specific solver configurations"""
    phase_speed_limits: Tuple[float, float] = (0.0, 1e6)
    mesh_multiplier: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    n_mesh_points: int = 2000
    max_modes: int = 9999
    acoustic_media_model: Literal["auto", "single"] = "auto"
    field_option: str = "RA"
    field_profile_ranges_km: Optional[List[float]] = None
    field_receiver_range_offsets_m: Optional[List[float]] = None
    field_receiver_thetas_deg: Optional[List[float]] = None
    field_node_x_km: Optional[List[float]] = None
    field_node_y_km: Optional[List[float]] = None
    field_node_env_files: Optional[List[str]] = None
    field_elements: Optional[List[Tuple[int, int, int]]] = None
    field_alpha_scan: Optional[Tuple[float, float, int]] = None
    field_step: Optional[Tuple[float, int]] = None
    field_epmult: Optional[float] = None
    type: str = "normal_modes"

@dataclass
class Bellhop3DConfig:
    """Bellhop3D specific solver configurations"""
    elevation_angles: Tuple[float, float]
    azimuth_angles: Tuple[float, float]
    num_elevation_beams: int = 0
    num_azimuth_beams: int = 0
    step_size: float = 0.0
    box_x: Optional[float] = None
    box_y: Optional[float] = None
    box_z: Optional[float] = None
    type: str = "bellhop3d"

@dataclass
class SimulationConfig:
    """Root simulation configuration model"""
    project: str
    frequency: float
    environment: EnvironmentModel
    geometry: GeometryModel
    solver: Union[BellhopConfig, KrakenConfig, Bellhop3DConfig]
