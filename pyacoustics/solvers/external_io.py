import numpy as np
import struct
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ATPRTLog:
    """Parsed summary of an Acoustics Toolbox .prt text log."""

    path: Path
    text: str
    lines: list[str]
    warnings: list[str]
    errors: list[str]
    warning_count: int = 0
    error_count: int = 0
    fatal_error_count: int = 0
    title: str | None = None
    program: str | None = None
    case_title: str | None = None
    frequency_hz: float | None = None
    nmedia: int | None = None
    bottom_depth_m: float | None = None
    rms_roughness_m: float | None = None
    source_depth_count: int | None = None
    source_depth_min_m: float | None = None
    source_depth_max_m: float | None = None
    receiver_depth_count: int | None = None
    receiver_range_count: int | None = None
    receiver_depth_min_m: float | None = None
    receiver_depth_max_m: float | None = None
    receiver_range_min_km: float | None = None
    receiver_range_max_km: float | None = None
    beam_count: int | None = None
    beam_bearing_count: int | None = None
    beam_angle_min_deg: float | None = None
    beam_angle_max_deg: float | None = None
    beam_bearing_angle_min_deg: float | None = None
    beam_bearing_angle_max_deg: float | None = None
    step_length_m: float | None = None
    box_x_m: float | None = None
    box_y_m: float | None = None
    box_depth_m: float | None = None
    box_range_km: float | None = None
    max_arrivals: int | None = None
    cpu_time_s: float | None = None
    run_type_description: str | None = None
    beam_description: str | None = None
    source_description: str | None = None
    receiver_grid_description: str | None = None
    source_beam_description: str | None = None
    mode_addition_description: str | None = None
    calculation_description: str | None = None
    field_option: str | None = None
    field_alpha_min_deg: float | None = None
    field_alpha_max_deg: float | None = None
    field_alpha_count: int | None = None
    field_step_m: float | None = None
    field_step_count: int | None = None
    source_x_count: int | None = None
    source_y_count: int | None = None
    source_x_min_km: float | None = None
    source_x_max_km: float | None = None
    source_y_min_km: float | None = None
    source_y_max_km: float | None = None
    source_coordinate_x_m: float | None = None
    source_coordinate_y_m: float | None = None
    source_coordinate_z_m: float | None = None
    receiver_theta_count: int | None = None
    receiver_theta_min_deg: float | None = None
    receiver_theta_max_deg: float | None = None
    node_count: int | None = None
    element_count: int | None = None
    profile_range_count: int | None = None
    profile_range_min_km: float | None = None
    profile_range_max_km: float | None = None
    receiver_range_offset_count: int | None = None
    receiver_range_offset_min_m: float | None = None
    receiver_range_offset_max_m: float | None = None
    completed_successfully: bool = False
    completed_without_errors: bool = False
    bathymetry_x_count: int | None = None
    bathymetry_y_count: int | None = None
    bathymetry_x_min_km: float | None = None
    bathymetry_x_max_km: float | None = None
    bathymetry_y_min_km: float | None = None
    bathymetry_y_max_km: float | None = None
    beam_shift_description: str | None = None
    c_low_mps: float | None = None
    c_high_mps: float | None = None
    rmax_km: float | None = None
    attenuation_units: str | None = None
    mesh_point_count: int | None = None
    mesh_multiplier_cpu_s: dict[int, float] | None = None
    max_root_finder_restarts: int | None = None
    beam_halfwidth1: float | None = None
    beam_halfwidth2: float | None = None
    eps_multiplier: float | None = None
    mode_count: int | None = None
    mode_wavenumber_min_1pm: float | None = None
    mode_wavenumber_max_1pm: float | None = None
    mode_attenuation_min_1pm: float | None = None
    mode_attenuation_max_1pm: float | None = None
    mode_phase_speed_min_mps: float | None = None
    mode_phase_speed_max_mps: float | None = None
    mode_group_speed_min_mps: float | None = None
    mode_group_speed_max_mps: float | None = None
    parameters: dict[str, float | str] | None = None


@dataclass(frozen=True)
class ATBoundaryProfile:
    """2D AT boundary profile used by .ati and .bty files."""

    interpolation: str
    ranges_m: np.ndarray
    values_m: np.ndarray
    geoacoustics: np.ndarray | None = None


@dataclass(frozen=True)
class ATBathymetry3D:
    """3D AT Bellhop3D gridded bathymetry used by ``.bty`` files."""

    interpolation: str
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray


@dataclass(frozen=True)
class ATBathymetry3DProvinces:
    """Long-format Bellhop3D ``.bty`` with bottom province/material indices."""

    interpolation: str
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    province_indices: np.ndarray
    province_properties: np.ndarray
    province_property_kind: str


@dataclass(frozen=True)
class ATRangeDependentSSP:
    """2D AT Bellhop range-dependent SSP sidecar used by ``.ssp`` files."""

    ranges_m: np.ndarray
    c_matrix: np.ndarray


@dataclass(frozen=True)
class ATReflectionTable:
    """Reflection coefficient table used by AT .irc and .brc files."""

    angles_deg: np.ndarray
    amplitudes: np.ndarray
    phases_deg: np.ndarray

    @property
    def coefficients(self) -> np.ndarray:
        return self.amplitudes * np.exp(1j * np.deg2rad(self.phases_deg))


@dataclass(frozen=True)
class ATSourceBeamPattern:
    """Source beam pattern table used by AT ``.sbp`` files."""

    angles_deg: np.ndarray
    levels_db: np.ndarray


@dataclass(frozen=True)
class ATEnvSummary:
    """Lightweight metadata parsed from an AT ``.env`` file."""

    title: str
    frequency_hz: float
    nmedia: int
    ssp_option: str | None
    top_option: str | None
    bottom_option: str | None
    model_family: str
    run_type: str | None = None
    source_depth_count: int | None = None
    receiver_depth_count: int | None = None
    receiver_range_count: int | None = None
    source_x_count: int | None = None
    source_y_count: int | None = None
    receiver_bearing_count: int | None = None
    beam_count: int | None = None
    beam_bearing_count: int | None = None
    box_range_km: float | None = None
    box_y_km: float | None = None
    box_z_m: float | None = None


@dataclass(frozen=True)
class ATRAMInSummary:
    """Lightweight metadata parsed from an AT RAM ``ram.in`` input file."""

    title: str
    frequency_hz: float
    source_depth_m: float
    receiver_depth_m: float
    rmax_m: float
    dr_m: float
    receiver_range_decimation: int
    zmax_m: float
    dz_m: float
    depth_decimation: int
    plot_depth_m: float
    reference_sound_speed_mps: float
    range_dependent_profile_count: int
    sediment_layer_count: int
    source_range_m: float
    bathymetry_point_count: int
    ssp_point_count: int
    bottom_sound_speed_point_count: int
    density_point_count: int
    attenuation_point_count: int
    profile_range_count: int
    bathymetry_range_min_m: float | None = None
    bathymetry_range_max_m: float | None = None
    bathymetry_depth_min_m: float | None = None
    bathymetry_depth_max_m: float | None = None
    profile_range_min_m: float | None = None
    profile_range_max_m: float | None = None
    sound_speed_min_mps: float | None = None
    sound_speed_max_mps: float | None = None
    bottom_sound_speed_min_mps: float | None = None
    bottom_sound_speed_max_mps: float | None = None
    density_min_gpcm3: float | None = None
    density_max_gpcm3: float | None = None
    attenuation_min_db_per_wavelength: float | None = None
    attenuation_max_db_per_wavelength: float | None = None


@dataclass(frozen=True)
class ATMatlabScriptSummary:
    """Lightweight inventory metadata for an AT Matlab/Octave ``.m`` script."""

    path: Path
    line_count: int
    code_line_count: int
    comment_line_count: int
    function_count: int
    function_names: tuple[str, ...]
    at_command_counts: dict[str, int]
    external_command_counts: dict[str, int]
    referenced_extensions: tuple[str, ...]


@dataclass(frozen=True)
class ATFortranSourceSummary:
    """Lightweight inventory metadata for an AT Fortran source file."""

    path: Path
    line_count: int
    code_line_count: int
    comment_line_count: int
    program_names: tuple[str, ...]
    module_names: tuple[str, ...]
    subroutine_names: tuple[str, ...]
    function_names: tuple[str, ...]
    used_modules: tuple[str, ...]


@dataclass(frozen=True)
class ATRayRecord:
    """One ray from an AT ``.ray`` file."""

    launch_angle_deg: float
    ranges_m: np.ndarray
    depths_m: np.ndarray
    amplitudes: np.ndarray
    num_top_bounces: int
    num_bottom_bounces: int
    coordinate_system: str = "rz"
    y_m: np.ndarray | None = None
    launch_azimuth_deg: float | None = None


@dataclass(frozen=True)
class ATShadeData:
    """Structured AT ``.shd`` pressure data."""

    pressure: np.ndarray
    receiver_depths_m: np.ndarray
    receiver_ranges_m: np.ndarray
    source_depths_m: np.ndarray
    bearings_deg: np.ndarray
    source_x_m: np.ndarray | None = None
    source_y_m: np.ndarray | None = None
    pressure_full: np.ndarray | None = None
    pressure_frequency_full: np.ndarray | None = None
    frequency_vector_hz: np.ndarray | None = None
    plot_type: str | None = None
    attenuation: float | None = None
    title: str | None = None
    frequency_hz: float | None = None
    selected_frequency_index: int = 0
    selected_source_x_index: int = 0
    selected_source_y_index: int = 0


@dataclass(frozen=True)
class ATFieldFLP:
    """Parsed summary of an AT Field/Field3D ``.flp`` input file."""

    path: Path
    title: str
    option: str
    max_modes: int
    is_3d: bool
    profile_range_count: int | None = None
    profile_ranges_km: np.ndarray | None = None
    receiver_range_count: int | None = None
    receiver_ranges_km: np.ndarray | None = None
    source_depth_count: int | None = None
    source_depths_m: np.ndarray | None = None
    receiver_depth_count: int | None = None
    receiver_depths_m: np.ndarray | None = None
    receiver_range_offset_count: int | None = None
    receiver_range_offsets_m: np.ndarray | None = None
    source_x_count: int | None = None
    source_x_km: np.ndarray | None = None
    source_y_count: int | None = None
    source_y_km: np.ndarray | None = None
    receiver_theta_count: int | None = None
    receiver_thetas_deg: np.ndarray | None = None
    node_count: int | None = None
    node_x_km: np.ndarray | None = None
    node_y_km: np.ndarray | None = None
    node_env_files: list[str] | None = None
    element_count: int | None = None
    elements: np.ndarray | None = None
    alpha_min_deg: float | None = None
    alpha_max_deg: float | None = None
    alpha_count: int | None = None
    step_m: float | None = None
    step_count: int | None = None
    epmult: float | None = None


def _strip_at_comment(line: str) -> str:
    return line.split("!", 1)[0].strip()


def _clean_at_token(token: str) -> str:
    return token.strip().strip("'\"")

def _at_data_lines(path):
    lines = []
    for line in Path(path).read_text().splitlines():
        clean = _strip_at_comment(line).strip()
        if clean:
            lines.append(clean)
    return lines

def _at_quoted_or_token(line):
    match = re.search(r"['\"]([^'\"]*)['\"]", line)
    if match:
        return match.group(1).strip()
    return _clean_at_token(line.replace(",", " ").split()[0])

def _at_numeric_values(line):
    clean = _strip_at_comment(line).replace(",", " ").replace("/", " ")
    return np.array(
        [
            float(token.replace("D", "E").replace("d", "e"))
            for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?", clean)
        ],
        dtype=np.float64,
    )

def _at_count(line):
    values = _at_numeric_values(line)
    if values.size == 0:
        raise ValueError("AT numeric count line is empty")
    return int(values[0])

def _at_values_for_count(line, count):
    values = _at_numeric_values(line)
    if values.size == count:
        return values
    if values.size == 2 and count > 2:
        return np.linspace(values[0], values[1], count)
    if values.size == 1 and count == 1:
        return values
    if values.size == 1 and count > 1:
        return np.full(count, values[0], dtype=np.float64)
    return values[:count]

def _at_values_array_for_count(values, count):
    values = np.asarray(values, dtype=np.float64)
    if values.size == count:
        return values
    if values.size == 2 and count > 2:
        return np.linspace(values[0], values[1], count)
    if values.size == 1 and count == 1:
        return values
    if values.size == 1 and count > 1:
        return np.full(count, values[0], dtype=np.float64)
    return values[:count]

def _field_count_values(lines, idx):
    values = _at_numeric_values(lines[idx])
    if values.size == 0:
        raise ValueError("AT .flp count line is empty")
    count = int(values[0])
    if values.size > 1:
        return count, _at_values_array_for_count(values[1:], count), idx + 1
    idx += 1
    collected = []
    while idx < len(lines):
        collected.extend(_at_numeric_values(lines[idx]).tolist())
        idx += 1
        if len(collected) >= count or (count > 2 and len(collected) >= 2):
            break
    return count, _at_values_array_for_count(np.asarray(collected, dtype=np.float64), count), idx

def _field_grid_values(lines, idx):
    values = _at_numeric_values(lines[idx])
    if values.size >= 3:
        count = int(values[2])
        return count, _at_values_array_for_count(values[:2], count), idx + 1
    return _field_count_values(lines, idx)

def _is_field3d_option(value):
    upper = value.upper()
    return upper.startswith("ST") or upper.startswith("GBT")


def _numeric_tokens(path):
    tokens = []
    for line in Path(path).read_text().splitlines():
        line = _strip_at_comment(line).replace(",", " ").replace("/", " ")
        for token in line.split():
            tokens.append(token.replace("D", "E").replace("d", "e"))
    return tokens

def _value_or_default(value, default):
    return default if value is None else value

def _format_at_values(values):
    return " ".join(f"{float(value):.9g}" for value in values) + " /"

def _format_at_int_values(values):
    return ", ".join(str(int(value)) for value in values)

def _is_even_endpoint_grid(values, count):
    values = np.asarray(values, dtype=np.float64)
    if count <= 2 or values.size != count:
        return False
    expected = np.linspace(values[0], values[-1], count)
    return bool(np.allclose(values, expected, rtol=1e-10, atol=1e-12))

def _field_grid_count_and_values(values, *, scale=1.0, interpolated_count=1001):
    values = np.asarray(values, dtype=np.float64) * scale
    if values.size == 0:
        raise ValueError("AT field grid requires at least one value")
    if values.size == 2:
        return interpolated_count, [float(values[0]), float(values[-1])]
    return int(values.size), values.tolist()

def generate_at_field_flp(config, flp_file_path):
    max_modes = int(_value_or_default(getattr(config.solver, "max_modes", None), 9999))
    field_option = str(_value_or_default(getattr(config.solver, "field_option", None), "RA")).strip().strip("'\"")
    if not field_option:
        raise ValueError("AT Field .flp option must be non-empty")
    if _is_field3d_option(field_option):
        source_x = getattr(config.geometry.source, "x", None)
        source_y = getattr(config.geometry.source, "y", None)
        source_x_km = np.asarray(source_x if source_x is not None else [0.0], dtype=np.float64) / 1000.0
        source_y_km = np.asarray(source_y if source_y is not None else [0.0], dtype=np.float64) / 1000.0
        receiver_thetas = getattr(config.solver, "field_receiver_thetas_deg", None) or [0.0]
        node_x = np.asarray(getattr(config.solver, "field_node_x_km", None) or [], dtype=np.float64)
        node_y = np.asarray(getattr(config.solver, "field_node_y_km", None) or [], dtype=np.float64)
        node_env_files = list(getattr(config.solver, "field_node_env_files", None) or [])
        if node_x.size == 0 or node_y.size == 0 or not node_env_files:
            raise ValueError("AT Field3D .flp generation requires node x/y coordinates and environment files")
        if node_x.size != node_y.size or node_x.size != len(node_env_files):
            raise ValueError("AT Field3D node x/y coordinates and environment files must have the same length")

        rr_count, rr_values = _field_grid_count_and_values(
            config.geometry.receivers.ranges,
            scale=1.0 / 1000.0,
        )
        rd_count, rd_values = _field_grid_count_and_values(config.geometry.receivers.depths)
        theta_count, theta_values = _field_grid_count_and_values(receiver_thetas)

        out = f"'{field_option}'\n{max_modes}\n"
        out += f"{source_x_km.size}\n{_format_at_values(source_x_km)}\n"
        out += f"{source_y_km.size}\n{_format_at_values(source_y_km)}\n"
        out += f"{len(config.geometry.source.depths)}\n{_format_at_values(config.geometry.source.depths)}\n"
        out += f"{rd_count}\n{_format_at_values(rd_values)}\n"
        out += f"{rr_count}\n{_format_at_values(rr_values)}\n"
        out += f"{theta_count}\n{_format_at_values(theta_values)}\n"
        out += f"{node_x.size}\n"
        for x_value, y_value, env_name in zip(node_x, node_y, node_env_files):
            out += f"{float(x_value):.9g}, {float(y_value):.9g}, '{env_name}'\n"

        elements = getattr(config.solver, "field_elements", None)
        if elements is not None:
            elements_arr = np.asarray(elements, dtype=np.int64)
            if elements_arr.ndim != 2 or elements_arr.shape[1] != 3:
                raise ValueError("AT Field3D elements must be triples of node indices")
            out += f"{elements_arr.shape[0]}\n"
            for row in elements_arr:
                out += _format_at_int_values(row) + "\n"
        alpha_scan = getattr(config.solver, "field_alpha_scan", None)
        if alpha_scan is not None:
            out += _format_at_values(alpha_scan) + "\n"
        field_step = getattr(config.solver, "field_step", None)
        if field_step is not None:
            out += _format_at_values(field_step) + "\n"
        epmult = getattr(config.solver, "field_epmult", None)
        if epmult is not None:
            out += _format_at_values([epmult]) + "\n"
        with open(flp_file_path, 'w') as f: f.write(out)
        return

    profile_ranges = getattr(config.solver, "field_profile_ranges_km", None)
    if profile_ranges is None:
        out = f"'/'\n'{field_option}'\n{max_modes}\n1\n"
        out += "0.0 /\n"
    else:
        profile_values = np.asarray(profile_ranges, dtype=np.float64)
        if profile_values.size == 0:
            raise ValueError("AT Field profile ranges require at least one value")
        out = f"'/'\n'{field_option}'\n{max_modes}\n{profile_values.size}\n"
        out += _format_at_values(profile_values) + "\n"
    rr_count, rr_values = _field_grid_count_and_values(
        config.geometry.receivers.ranges,
        scale=1.0 / 1000.0,
    )
    out += f"{rr_count}\n"
    out += _format_at_values(rr_values) + "\n"
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += _format_at_values(src_depths) + "\n"
    rd_count, rd_values = _field_grid_count_and_values(config.geometry.receivers.depths)
    out += f"{rd_count}\n"
    out += _format_at_values(rd_values) + "\n"
    offsets = getattr(config.solver, "field_receiver_range_offsets_m", None)
    if offsets is None:
        offset_count = rd_count
        offset_values = np.zeros(rd_count, dtype=np.float64)
    else:
        offset_values_arr = np.asarray(offsets, dtype=np.float64)
        if offset_values_arr.size == 0:
            raise ValueError("AT Field receiver range offsets require at least one value")
        receiver_depth_count = len(config.geometry.receivers.depths)
        if offset_values_arr.size == rd_count:
            offset_count = rd_count
            offset_values = offset_values_arr
        elif offset_values_arr.size == receiver_depth_count:
            offset_count = rd_count
            offset_values = offset_values_arr
        else:
            raise ValueError(
                "AT Field receiver range offsets must match receiver depth count "
                "or the expanded Field receiver-depth grid count"
            )
    out += f"{offset_count}\n"
    out += _format_at_values(offset_values) + "\n"
    with open(flp_file_path, 'w') as f: f.write(out)

def write_at_field_flp(flp, flp_file_path):
    """Write a parsed :class:`ATFieldFLP` back to AT ``.flp`` text."""
    path = Path(flp_file_path)
    if (
        not flp.is_3d
        and flp.profile_range_count is None
        and flp.source_depth_count is None
        and flp.receiver_depth_count is None
        and flp.receiver_range_offset_count is None
    ):
        receiver_ranges = np.asarray(flp.receiver_ranges_km, dtype=np.float64)
        receiver_range_count = int(flp.receiver_range_count or receiver_ranges.size)
        if _is_even_endpoint_grid(receiver_ranges, receiver_range_count):
            lines = [
                f"'{flp.option}'",
                f"{float(receiver_ranges[0]):.9g} {float(receiver_ranges[-1]):.9g} {receiver_range_count}",
            ]
            path.write_text("\n".join(lines) + "\n")
            return
        lines = [
            f"'{flp.option}'",
            f"{receiver_range_count}",
            _format_at_values(receiver_ranges),
        ]
        path.write_text("\n".join(lines) + "\n")
        return
    lines = [
        f"'{flp.title}'",
        f"'{flp.option}'",
        f"{int(flp.max_modes)}",
    ]
    if not flp.is_3d:
        profile_ranges = np.asarray(flp.profile_ranges_km, dtype=np.float64)
        receiver_ranges = np.asarray(flp.receiver_ranges_km, dtype=np.float64)
        source_depths = np.asarray(flp.source_depths_m, dtype=np.float64)
        receiver_depths = np.asarray(flp.receiver_depths_m, dtype=np.float64)
        offsets = np.asarray(flp.receiver_range_offsets_m, dtype=np.float64)
        lines.extend(
            [
                f"{int(flp.profile_range_count or profile_ranges.size)}",
                _format_at_values(profile_ranges),
                f"{int(flp.receiver_range_count or receiver_ranges.size)}",
                _format_at_values(receiver_ranges),
                f"{int(flp.source_depth_count or source_depths.size)}",
                _format_at_values(source_depths),
                f"{int(flp.receiver_depth_count or receiver_depths.size)}",
                _format_at_values(receiver_depths),
                f"{int(flp.receiver_range_offset_count or offsets.size)}",
                _format_at_values(offsets),
            ]
        )
    else:
        source_x = np.asarray(flp.source_x_km, dtype=np.float64)
        source_y = np.asarray(flp.source_y_km, dtype=np.float64)
        source_depths = np.asarray(flp.source_depths_m, dtype=np.float64)
        receiver_depths = np.asarray(flp.receiver_depths_m, dtype=np.float64)
        receiver_ranges = np.asarray(flp.receiver_ranges_km, dtype=np.float64)
        thetas = np.asarray(flp.receiver_thetas_deg, dtype=np.float64)
        node_x = np.asarray(flp.node_x_km, dtype=np.float64)
        node_y = np.asarray(flp.node_y_km, dtype=np.float64)
        node_env_files = flp.node_env_files or []
        elements = np.asarray(flp.elements if flp.elements is not None else np.empty((0, 3)), dtype=np.int64)
        lines.extend(
            [
                f"{int(flp.source_x_count or source_x.size)}",
                _format_at_values(source_x),
                f"{int(flp.source_y_count or source_y.size)}",
                _format_at_values(source_y),
                f"{int(flp.source_depth_count or source_depths.size)}",
                _format_at_values(source_depths),
                f"{int(flp.receiver_depth_count or receiver_depths.size)}",
                _format_at_values(receiver_depths),
                f"{int(flp.receiver_range_count or receiver_ranges.size)}",
                _format_at_values(receiver_ranges),
                f"{int(flp.receiver_theta_count or thetas.size)}",
                _format_at_values(thetas),
                f"{int(flp.node_count or node_x.size)}",
            ]
        )
        for x_value, y_value, env_name in zip(node_x, node_y, node_env_files):
            lines.append(f"{float(x_value):.9g}, {float(y_value):.9g}, '{env_name}'")
        if flp.element_count is not None or elements.size:
            lines.append(f"{int(flp.element_count or elements.shape[0])}")
            for row in elements:
                lines.append(_format_at_int_values(row[:3]))
        if flp.alpha_count is not None:
            lines.append(_format_at_values([flp.alpha_min_deg, flp.alpha_max_deg, flp.alpha_count]))
        if flp.step_count is not None:
            lines.append(_format_at_values([flp.step_m, flp.step_count]))
        if flp.epmult is not None:
            lines.append(_format_at_values([flp.epmult]))
    path.write_text("\n".join(lines) + "\n")

def read_at_field_flp(flp_file_path):
    """Read common AT Field/Field3D ``.flp`` grids.

    Two-dimensional Field files use profile range, receiver range, source
    depth, receiver depth, and array-tilt range-offset blocks. Field3D files
    add source x/y grids, receiver theta, and a node-to-environment table.
    Endpoint grids are expanded to the declared count to mirror AT's read
    semantics.
    """
    path = Path(flp_file_path)
    lines = _at_data_lines(path)
    if len(lines) < 2:
        raise ValueError("AT .flp file is too short")
    idx = 0
    first_token = _at_quoted_or_token(lines[idx])
    if len(lines) == 2:
        receiver_range_values = _at_numeric_values(lines[1])
        if receiver_range_values.size < 3:
            raise ValueError("AT compact .flp range line requires rmin, rmax, and count")
        receiver_range_count = int(receiver_range_values[2])
        return ATFieldFLP(
            path=path,
            title="",
            option=first_token,
            max_modes=0,
            is_3d=False,
            receiver_range_count=receiver_range_count,
            receiver_ranges_km=_at_values_array_for_count(receiver_range_values[:2], receiver_range_count),
        )
    if len(lines) < 3:
        raise ValueError("AT .flp file is too short")
    if len(lines) == 3:
        receiver_range_count = _at_count(lines[1])
        return ATFieldFLP(
            path=path,
            title="",
            option=first_token,
            max_modes=0,
            is_3d=False,
            receiver_range_count=receiver_range_count,
            receiver_ranges_km=_at_values_for_count(lines[2], receiver_range_count),
        )
    if len(lines) < 5:
        raise ValueError("AT .flp file is too short")
    if _is_field3d_option(first_token):
        title = ""
        option = first_token
        idx += 1
    else:
        title = first_token; idx += 1
        option = _at_quoted_or_token(lines[idx]); idx += 1
    max_modes = _at_count(lines[idx]); idx += 1
    option_upper = option.upper()
    is_3d = _is_field3d_option(option_upper)

    if not is_3d:
        profile_count = _at_count(lines[idx]); idx += 1
        profile_ranges = _at_values_for_count(lines[idx], profile_count); idx += 1
        receiver_range_count = _at_count(lines[idx]); idx += 1
        receiver_ranges = _at_values_for_count(lines[idx], receiver_range_count); idx += 1
        source_depth_count = _at_count(lines[idx]); idx += 1
        source_depths = _at_values_for_count(lines[idx], source_depth_count); idx += 1
        receiver_depth_count = _at_count(lines[idx]); idx += 1
        receiver_depths = _at_values_for_count(lines[idx], receiver_depth_count); idx += 1
        offset_count = _at_count(lines[idx]); idx += 1
        offsets = _at_values_for_count(lines[idx], offset_count) if idx < len(lines) else np.zeros(offset_count)
        return ATFieldFLP(
            path=path,
            title=title,
            option=option,
            max_modes=max_modes,
            is_3d=False,
            profile_range_count=profile_count,
            profile_ranges_km=profile_ranges,
            receiver_range_count=receiver_range_count,
            receiver_ranges_km=receiver_ranges,
            source_depth_count=source_depth_count,
            source_depths_m=source_depths,
            receiver_depth_count=receiver_depth_count,
            receiver_depths_m=receiver_depths,
            receiver_range_offset_count=offset_count,
            receiver_range_offsets_m=offsets,
        )

    source_xy_values = _at_numeric_values(lines[idx])
    if source_xy_values.size >= 2:
        source_x_count = 1
        source_y_count = 1
        source_x = np.array([source_xy_values[0]], dtype=np.float64)
        source_y = np.array([source_xy_values[1]], dtype=np.float64)
        idx += 1
    else:
        source_x_count = _at_count(lines[idx]); idx += 1
        source_x = _at_values_for_count(lines[idx], source_x_count); idx += 1
        source_y_count = _at_count(lines[idx]); idx += 1
        source_y = _at_values_for_count(lines[idx], source_y_count); idx += 1
    source_depth_count, source_depths, idx = _field_count_values(lines, idx)
    receiver_depth_count, receiver_depths, idx = _field_count_values(lines, idx)
    receiver_range_count, receiver_ranges, idx = _field_grid_values(lines, idx)
    theta_count, thetas, idx = _field_grid_values(lines, idx)
    node_count = _at_count(lines[idx]); idx += 1
    node_x = []
    node_y = []
    node_env_files = []
    for line in lines[idx: idx + node_count]:
        values = _at_numeric_values(line)
        if values.size >= 2:
            node_x.append(float(values[0]))
            node_y.append(float(values[1]))
        node_env_files.append(_at_quoted_or_token(line))
    idx += node_count
    element_count = None
    elements = None
    alpha_min = None
    alpha_max = None
    alpha_count = None
    step_m = None
    step_count = None
    epmult = None
    if idx < len(lines):
        element_count = _at_count(lines[idx]); idx += 1
        element_rows = []
        for line in lines[idx: idx + element_count]:
            values = _at_numeric_values(line).astype(np.int64)
            if values.size >= 3:
                element_rows.append(values[:3])
        idx += element_count
        elements = np.vstack(element_rows) if element_rows else np.empty((0, 3), dtype=np.int64)
    if idx < len(lines):
        alpha_values = _at_numeric_values(lines[idx]); idx += 1
        if alpha_values.size >= 3:
            alpha_min = float(alpha_values[0])
            alpha_max = float(alpha_values[1])
            alpha_count = int(alpha_values[2])
    if idx < len(lines):
        step_values = _at_numeric_values(lines[idx]); idx += 1
        if step_values.size >= 2:
            step_m = float(step_values[0])
            step_count = int(step_values[1])
    if idx < len(lines):
        ep_values = _at_numeric_values(lines[idx])
        if ep_values.size >= 1:
            epmult = float(ep_values[0])
    return ATFieldFLP(
        path=path,
        title=title,
        option=option,
        max_modes=max_modes,
        is_3d=True,
        receiver_range_count=receiver_range_count,
        receiver_ranges_km=receiver_ranges,
        source_depth_count=source_depth_count,
        source_depths_m=source_depths,
        receiver_depth_count=receiver_depth_count,
        receiver_depths_m=receiver_depths,
        source_x_count=source_x_count,
        source_x_km=source_x,
        source_y_count=source_y_count,
        source_y_km=source_y,
        receiver_theta_count=theta_count,
        receiver_thetas_deg=thetas,
        node_count=node_count,
        node_x_km=np.asarray(node_x, dtype=np.float64),
        node_y_km=np.asarray(node_y, dtype=np.float64),
        node_env_files=node_env_files,
        element_count=element_count,
        elements=elements,
        alpha_min_deg=alpha_min,
        alpha_max_deg=alpha_max,
        alpha_count=alpha_count,
        step_m=step_m,
        step_count=step_count,
        epmult=epmult,
    )

def generate_at_kraken_env(config, env_file_path):
    surface = config.environment.surface
    top_reflection = getattr(surface, "reflection_table", None)
    if top_reflection is not None:
        top_opt = "F"
        table = np.asarray(top_reflection, dtype=np.float64)
        write_at_reflection_table(
            Path(env_file_path).with_suffix(".trc"),
            ATReflectionTable(table[:, 0], table[:, 1], table[:, 2]),
        )
    elif surface.type == "rigid":
        top_opt = "R"
    else:
        top_opt = "V"
    ssp_type = getattr(config.environment.ssp, "type", "c-linear")
    ssp_opt = {"spline": "S", "n2-linear": "N"}.get(ssp_type, "C")
    out = f"'{config.project}'\n{config.frequency}\n1\n'{ssp_opt}{top_opt}W'\n"
    z_bottom = config.environment.bottom.depth or config.environment.ssp.data[-1].depth
    n_mesh_points = int(_value_or_default(getattr(config.solver, "n_mesh_points", None), 0))
    out += f"{n_mesh_points} 0.0 {z_bottom}\n"
    for p in config.environment.ssp.data:
        rho = p.density if hasattr(p, 'density') and p.density else 1.0
        alpha = p.attenuation if hasattr(p, 'attenuation') and p.attenuation else 0.0
        out += f"  {p.depth} {p.c} 0.0 {rho} {alpha} 0.0 /\n"
    bot_bc = config.environment.bottom.type
    if getattr(config.environment.bottom, "reflection_table", None) is not None:
        bot_opt = "F"
        table = np.asarray(config.environment.bottom.reflection_table, dtype=np.float64)
        write_at_reflection_table(
            Path(env_file_path).with_suffix(".brc"),
            ATReflectionTable(table[:, 0], table[:, 1], table[:, 2]),
        )
    else:
        bot_opt = 'V' if bot_bc == "vacuum" else ('R' if bot_bc == "rigid" else 'A')
    bottom_roughness = _value_or_default(getattr(config.environment.bottom, "rms_roughness_m", None), 0.0)
    out += f"'{bot_opt}' {bottom_roughness}\n"
    if bot_opt == "A":
        bot_c = _value_or_default(getattr(config.environment.bottom, 'c_p', None), 1600.0)
        bot_c_s = _value_or_default(getattr(config.environment.bottom, 'c_s', None), 0.0)
        bot_rho = _value_or_default(getattr(config.environment.bottom, 'density', None), 1.8)
        bot_alpha = _value_or_default(getattr(config.environment.bottom, 'attenuation_p', None), 0.0)
        bot_alpha_s = _value_or_default(getattr(config.environment.bottom, 'attenuation_s', None), 0.0)
        out += f"{z_bottom} {bot_c} {bot_c_s} {bot_rho} {bot_alpha} {bot_alpha_s} /\n"
    c_low, c_high = 1450.0, 1650.0
    if hasattr(config.solver, 'phase_speed_limits'):
        c_low, c_high = config.solver.phase_speed_limits
        if c_low <= 0: c_low = 1450.0
        if c_high <= 0: c_high = 1800.0
    receiver_ranges = np.asarray(getattr(config.geometry.receivers, "ranges", []) or [0.0], dtype=np.float64)
    r_max_km = float(np.max(receiver_ranges)) / 1000.0 if receiver_ranges.size else 0.0
    out += f"{c_low} {c_high}\n{r_max_km}\n"
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    rd_vec = config.geometry.receivers.depths
    rd_count, rd_values = _field_grid_count_and_values(rd_vec)
    out += f"{rd_count}\n"
    out += _format_at_values(rd_values) + "\n"
    with open(env_file_path, 'w') as f: f.write(out)

def _format_bellhop_run_type(config, run_type):
    run_type_text = str(run_type).strip().strip("'\"")
    beam_type = getattr(config.solver, "beam_type", "default")
    source_type = getattr(config.solver, "source_type", "point")
    receiver_grid = getattr(config.solver, "receiver_grid", "rectilinear")
    beam_displacement = getattr(config.solver, "beam_displacement", "none")
    beam_type_map = {
        "geometric_hat": "G",
        "geometric_gaussian": "B",
        "simple_gaussian": "S",
        "geometric_hat_ray_centered": "g",
        "geometric_gaussian_ray_centered": "b",
    }
    if (
        beam_type == "default"
        and source_type != "line"
        and receiver_grid != "irregular"
        and beam_displacement != "seongil"
    ):
        return run_type_text
    chars = list(run_type_text.ljust(7))
    if beam_type != "default":
        if beam_type not in beam_type_map:
            raise ValueError(f"Unsupported Bellhop beam_type: {beam_type}")
        chars[1] = beam_type_map[beam_type]
    if source_type == "line":
        chars[3] = "X"
    if receiver_grid == "irregular":
        chars[4] = "I"
    chars[6] = "S"
    if beam_displacement != "seongil":
        chars[6] = " "
    return "".join(chars)

def generate_at_bellhop_env(config, env_file_path, run_type="R"):
    env_file_path = Path(env_file_path)
    out = f"'{config.project}'\n{config.frequency}\n1\n"
    ssp_type = config.environment.ssp.type
    if ssp_type == "spline":
        opt1 = "S"
    elif ssp_type == "quad-2d":
        opt1 = "Q"
    else:
        opt1 = "C"
    top_altimetry = getattr(config.environment.surface, "altimetry", None)
    bottom_bathymetry = getattr(config.environment.bottom, "bathymetry", None)
    surface = config.environment.surface
    top_reflection = getattr(surface, "reflection_table", None)
    bottom_reflection = getattr(config.environment.bottom, "reflection_table", None)
    if top_reflection is not None:
        top_bc = "F"
    elif surface.type == "rigid":
        top_bc = "R"
    elif surface.type == "acousto-elastic":
        top_bc = "A"
    else:
        top_bc = "V"
    top_profile_opt = " *" if top_altimetry else ""
    out += f"'{opt1}{top_bc}W{top_profile_opt}'\n"
    if top_bc == "A":
        surf_z = _value_or_default(getattr(surface, "depth", None), 0.0)
        surf_c = _value_or_default(getattr(surface, "c_p", None), 1600.0)
        surf_c_s = _value_or_default(getattr(surface, "c_s", None), 0.0)
        surf_rho = _value_or_default(getattr(surface, "density", None), 1.0)
        surf_alpha = _value_or_default(getattr(surface, "attenuation_p", None), 0.0)
        surf_alpha_s = _value_or_default(getattr(surface, "attenuation_s", None), 0.0)
        out += f"{surf_z} {surf_c} {surf_c_s} {surf_rho} {surf_alpha} {surf_alpha_s} /\n"
    depths = [d.depth for d in config.environment.ssp.data]
    cs = [d.c for d in config.environment.ssp.data]
    z_bottom = config.environment.bottom.depth
    out += f"{len(depths)} 0.0 {z_bottom}\n"
    for d, c in zip(depths, cs): out += f"  {d} {c} /\n"
    bottom_profile_opt = "~" if bottom_bathymetry else ""
    bottom = config.environment.bottom
    if bottom_reflection is not None:
        bot_opt = "F"
    elif bottom.type == "vacuum":
        bot_opt = "V"
    elif bottom.type == "rigid":
        bot_opt = "R"
    else:
        bot_opt = "A"
    bottom_roughness = _value_or_default(getattr(bottom, "rms_roughness_m", None), 0.0)
    out += f"'{bot_opt}{bottom_profile_opt}' {bottom_roughness}\n"
    bot_c = _value_or_default(getattr(bottom, 'c_p', None), 1600.0)
    bot_c_s = _value_or_default(getattr(bottom, 'c_s', None), 0.0)
    bot_rho = _value_or_default(getattr(bottom, 'density', None), 1.8)
    bot_alpha = _value_or_default(getattr(bottom, 'attenuation_p', None), 0.5)
    bot_alpha_s = _value_or_default(getattr(bottom, 'attenuation_s', None), 0.0)
    if bot_opt == "A":
        out += f"{z_bottom} {bot_c} {bot_c_s} {bot_rho} {bot_alpha} {bot_alpha_s} /\n"
    src_depths = config.geometry.source.depths
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    rd_vec = config.geometry.receivers.depths
    receiver_grid = getattr(config.solver, "receiver_grid", "rectilinear")
    if receiver_grid == "irregular" and len(rd_vec) != len(config.geometry.receivers.ranges):
        raise ValueError("Bellhop irregular receiver grid requires matching receiver depth and range counts")
    out += f"{len(rd_vec)}\n"
    out += " ".join(map(str, rd_vec)) + " /\n"
    rr_vec = np.array(config.geometry.receivers.ranges) / 1000.0
    out += f"{len(rr_vec)}\n"
    out += " ".join(map(str, rr_vec)) + " /\n"
    out += f"'{_format_bellhop_run_type(config, run_type)}'\n"
    num_beams = config.solver.num_beams if config.solver.num_beams > 0 else 400
    out += f"{num_beams}\n"
    angles = config.solver.angles
    out += f"{angles[0]} {angles[1]} /\n"
    step = config.solver.step_size if config.solver.step_size > 0 else 0.0
    out += f"{step} {z_bottom*1.2} {max(rr_vec)*1.1}\n"
    with open(env_file_path, 'w') as f: f.write(out)

    if top_altimetry:
        ati_data = np.asarray(top_altimetry, dtype=np.float64)
        interpolation = getattr(config.environment.surface, "altimetry_interpolation", None) or "L"
        geoacoustics = getattr(config.environment.surface, "altimetry_geoacoustics", None)
        geoacoustics_arr = None if geoacoustics is None else np.asarray(geoacoustics, dtype=np.float64)
        write_at_boundary_profile(
            env_file_path.with_suffix(".ati"),
            ATBoundaryProfile(interpolation, ati_data[:, 0], ati_data[:, 1], geoacoustics_arr),
        )
    if bottom_bathymetry:
        bty_data = np.asarray(bottom_bathymetry, dtype=np.float64)
        interpolation = getattr(config.environment.bottom, "bathymetry_interpolation", None) or "L"
        geoacoustics = getattr(config.environment.bottom, "bathymetry_geoacoustics", None)
        geoacoustics_arr = None if geoacoustics is None else np.asarray(geoacoustics, dtype=np.float64)
        write_at_boundary_profile(
            env_file_path.with_suffix(".bty"),
            ATBoundaryProfile(interpolation, bty_data[:, 0], bty_data[:, 1], geoacoustics_arr),
        )
    if top_reflection is not None:
        table = np.asarray(top_reflection, dtype=np.float64)
        write_at_reflection_table(
            env_file_path.with_suffix(".irc"),
            ATReflectionTable(table[:, 0], table[:, 1], table[:, 2]),
        )
    if bottom_reflection is not None:
        table = np.asarray(bottom_reflection, dtype=np.float64)
        write_at_reflection_table(
            env_file_path.with_suffix(".brc"),
            ATReflectionTable(table[:, 0], table[:, 1], table[:, 2]),
        )
    if ssp_type == "quad-2d":
        ssp_ranges = getattr(config.environment.ssp, "ranges", None)
        ssp_matrix = getattr(config.environment.ssp, "c_matrix", None)
        if ssp_ranges is None or ssp_matrix is None:
            raise ValueError("Bellhop quad-2d SSP requires ssp.ranges and ssp.c_matrix")
        if len(ssp_matrix) != len(config.environment.ssp.data):
            raise ValueError("Bellhop quad-2d c_matrix row count must match SSP depth points")
        write_at_ssp2d(env_file_path.with_suffix(".ssp"), ssp_ranges, ssp_matrix)

def generate_at_bellhop3d_env(config, env_file_path, run_type="C"):
    """Write a Bellhop3D AT ``.env`` file from the native config model.

    Source ``x``/``y`` coordinates are stored in meters in ``SourceModel`` and
    converted to the kilometers expected by AT Bellhop3D.
    """
    env_file_path = Path(env_file_path)
    opt1 = 'S' if config.environment.ssp.type == 'spline' else 'C'
    top_altimetry = getattr(config.environment.surface, "altimetry", None)
    bottom_bathymetry = getattr(config.environment.bottom, "bathymetry", None)
    bottom_bathymetry_3d = getattr(config.environment.bottom, "bathymetry_3d", None)
    surface = config.environment.surface
    top_reflection = getattr(surface, "reflection_table", None)
    bottom_reflection = getattr(config.environment.bottom, "reflection_table", None)
    if top_reflection is not None:
        top_bc = "F"
    elif surface.type == "rigid":
        top_bc = "R"
    elif surface.type == "acousto-elastic":
        top_bc = "A"
    else:
        top_bc = "V"
    top_profile_opt = " *" if top_altimetry else ""
    out = f"'{config.project}'\n{config.frequency}\n1\n"
    out += f"'{opt1}{top_bc}W{top_profile_opt}'\n"
    if top_bc == "A":
        surf_z = _value_or_default(getattr(surface, "depth", None), 0.0)
        surf_c = _value_or_default(getattr(surface, "c_p", None), 1600.0)
        surf_c_s = _value_or_default(getattr(surface, "c_s", None), 0.0)
        surf_rho = _value_or_default(getattr(surface, "density", None), 1.0)
        surf_alpha = _value_or_default(getattr(surface, "attenuation_p", None), 0.0)
        surf_alpha_s = _value_or_default(getattr(surface, "attenuation_s", None), 0.0)
        out += f"{surf_z} {surf_c} {surf_c_s} {surf_rho} {surf_alpha} {surf_alpha_s} /\n"
    depths = [d.depth for d in config.environment.ssp.data]
    cs = [d.c for d in config.environment.ssp.data]
    z_bottom = config.environment.bottom.depth or depths[-1]
    out += f"{len(depths)} 0.0 {z_bottom}\n"
    for d, c in zip(depths, cs):
        out += f"  {d} {c} /\n"
    bottom_profile_opt = "~" if bottom_bathymetry or bottom_bathymetry_3d else ""
    bottom = config.environment.bottom
    if bottom_reflection is not None:
        bot_opt = "F"
    elif bottom.type == "vacuum":
        bot_opt = "V"
    elif bottom.type == "rigid":
        bot_opt = "R"
    else:
        bot_opt = "A"
    bottom_roughness = _value_or_default(getattr(bottom, "rms_roughness_m", None), 0.0)
    out += f"'{bot_opt}{bottom_profile_opt}' {bottom_roughness}\n"
    if bot_opt == "A":
        bot_c = _value_or_default(getattr(bottom, 'c_p', None), 1600.0)
        bot_c_s = _value_or_default(getattr(bottom, 'c_s', None), 0.0)
        bot_rho = _value_or_default(getattr(bottom, 'density', None), 1.8)
        bot_alpha = _value_or_default(getattr(bottom, 'attenuation_p', None), 0.5)
        bot_alpha_s = _value_or_default(getattr(bottom, 'attenuation_s', None), 0.0)
        out += f"{z_bottom} {bot_c} {bot_c_s} {bot_rho} {bot_alpha} {bot_alpha_s} /\n"
    source_x = np.asarray(config.geometry.source.x or [0.0], dtype=np.float64) / 1000.0
    source_y = np.asarray(config.geometry.source.y or [0.0], dtype=np.float64) / 1000.0
    src_depths = config.geometry.source.depths
    out += f"{len(source_x)}\n"
    out += " ".join(map(str, source_x)) + " /\n"
    out += f"{len(source_y)}\n"
    out += " ".join(map(str, source_y)) + " /\n"
    out += f"{len(src_depths)}\n"
    out += " ".join(map(str, src_depths)) + " /\n"
    rd_vec = config.geometry.receivers.depths
    out += f"{len(rd_vec)}\n"
    out += " ".join(map(str, rd_vec)) + " /\n"
    rr_vec = np.array(config.geometry.receivers.ranges, dtype=np.float64) / 1000.0
    out += f"{len(rr_vec)}\n"
    out += " ".join(map(str, rr_vec)) + " /\n"
    bearings = getattr(config.solver, "receiver_bearings", None) or [0.0]
    out += f"{len(bearings)}\n"
    out += " ".join(map(str, bearings)) + " /\n"
    run_option = run_type if str(run_type).startswith("'") else f"'{run_type}B   3'"
    out += f"{run_option}\n"
    num_elevation = config.solver.num_elevation_beams if config.solver.num_elevation_beams > 0 else 400
    num_azimuth = config.solver.num_azimuth_beams if config.solver.num_azimuth_beams > 0 else max(len(bearings), 1)
    out += f"{num_elevation}\n"
    out += f"{config.solver.elevation_angles[0]} {config.solver.elevation_angles[1]} /\n"
    out += f"{num_azimuth}\n"
    out += f"{config.solver.azimuth_angles[0]} {config.solver.azimuth_angles[1]} /\n"
    step = config.solver.step_size if config.solver.step_size > 0 else 0.0
    box_x = config.solver.box_x if config.solver.box_x is not None else (float(np.max(rr_vec)) * 1.1 if rr_vec.size else 1.0)
    box_y = config.solver.box_y if config.solver.box_y is not None else box_x
    box_z = config.solver.box_z if config.solver.box_z is not None else z_bottom * 1.2
    out += f"{step} {box_x} {box_y} {box_z}\n"
    with open(env_file_path, 'w') as f:
        f.write(out)

    if top_altimetry:
        ati_data = np.asarray(top_altimetry, dtype=np.float64)
        interpolation = getattr(config.environment.surface, "altimetry_interpolation", None) or "L"
        geoacoustics = getattr(config.environment.surface, "altimetry_geoacoustics", None)
        geoacoustics_arr = None if geoacoustics is None else np.asarray(geoacoustics, dtype=np.float64)
        write_at_boundary_profile(
            env_file_path.with_suffix(".ati"),
            ATBoundaryProfile(interpolation, ati_data[:, 0], ati_data[:, 1], geoacoustics_arr),
        )
    if bottom_bathymetry_3d:
        write_at_bathymetry_3d(env_file_path.with_suffix(".bty"), bottom_bathymetry_3d)
    elif bottom_bathymetry:
        bty_data = np.asarray(bottom_bathymetry, dtype=np.float64)
        interpolation = getattr(config.environment.bottom, "bathymetry_interpolation", None) or "L"
        geoacoustics = getattr(config.environment.bottom, "bathymetry_geoacoustics", None)
        geoacoustics_arr = None if geoacoustics is None else np.asarray(geoacoustics, dtype=np.float64)
        write_at_boundary_profile(
            env_file_path.with_suffix(".bty"),
            ATBoundaryProfile(interpolation, bty_data[:, 0], bty_data[:, 1], geoacoustics_arr),
        )
    if top_reflection is not None:
        table = np.asarray(top_reflection, dtype=np.float64)
        write_at_reflection_table(
            env_file_path.with_suffix(".irc"),
            ATReflectionTable(table[:, 0], table[:, 1], table[:, 2]),
        )
    if bottom_reflection is not None:
        table = np.asarray(bottom_reflection, dtype=np.float64)
        write_at_reflection_table(
            env_file_path.with_suffix(".brc"),
            ATReflectionTable(table[:, 0], table[:, 1], table[:, 2]),
        )

def read_at_shd_full(filename, *, frequency_hz=None, source_x_m=None, source_y_m=None):
    """Read an AT binary ``.shd`` file with source/bearing dimensions.

    The returned ``pressure`` shape is ``(bearing, source_depth,
    receiver_depth, range)`` for the selected frequency and source x/y
    position. Selection uses the nearest available AT grid point and defaults
    to the first frequency/source position. Full 3D source grids are preserved
    in ``pressure_full`` for the selected frequency with
    shape ``(source_x, source_y, bearing, source_depth, receiver_depth,
    range)``. Full broadband grids are preserved in ``pressure_frequency_full``
    with a leading frequency axis.
    """
    with open(filename, 'rb') as f:
        l_raw = f.read(4)
        if not l_raw:
            return None
        rec_size = struct.unpack('i', l_raw)[0] * 4
        def read_rec(n):
            f.seek((n-1)*rec_size); return f.read(rec_size)
        def valid_frequency(value):
            return value is not None and np.isfinite(value) and 1.0e-6 <= value <= 1.0e9
        def frequency_from_record(record):
            value = struct.unpack("<d", record[:8])[0] if len(record) >= 8 else None
            if valid_frequency(value):
                return float(value), "f8"
            value = struct.unpack("<f", record[:4])[0] if len(record) >= 4 else None
            if valid_frequency(value):
                return float(value), "f4"
            return None, None
        rec1 = read_rec(1)
        title = rec1[4:84].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip() or None
        rec2 = read_rec(2)
        frequency = struct.unpack("<f", rec2[:4])[0] if len(rec2) >= 4 else None
        plot_type = rec2[:10].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip() or None
        at_plot_type = bool(
            plot_type
            and (
                plot_type in {"rectilin", "irregular", "Green"}
                or plot_type.startswith("TL")
            )
        )
        rec3 = read_rec(3)
        rec4 = read_rec(4)
        rec4_frequency, rec4_frequency_dtype = frequency_from_record(rec4)
        native_pyacoustics_layout = (
            not at_plot_type
            and rec4_frequency is None
            and
            frequency is not None
            and np.isfinite(frequency)
            and 0.0 < frequency <= 1.0e9
        )
        attenuation = None
        if not native_pyacoustics_layout:
            frequency = rec4_frequency
        dims = struct.unpack('7i', rec3[:28])
        nfreq, ntheta, nsx, nsy, nsd, nrz, nrr = dims
        nfreq = max(int(nfreq), 1)
        ntheta = max(int(ntheta), 1)
        nsx = max(int(nsx), 1)
        nsy = max(int(nsy), 1)
        nsd = max(int(nsd), 1)
        nrz = int(nrz)
        nrr = int(nrr)
        if native_pyacoustics_layout:
            freq_vec = np.array([frequency], dtype=np.float64)
            bearings = np.frombuffer(read_rec(5)[:4*ntheta], dtype=np.float32).astype(np.float64)
            source_x = np.frombuffer(read_rec(6)[:4*nsx], dtype=np.float32).astype(np.float64)
            source_y = np.frombuffer(read_rec(7)[:4*nsy], dtype=np.float32).astype(np.float64)
        else:
            if len(rec3) >= 44:
                rec3_frequency = struct.unpack("<d", rec3[28:36])[0]
                if valid_frequency(rec3_frequency):
                    frequency = rec3_frequency
                attenuation = struct.unpack("<d", rec3[36:44])[0]
            if rec4_frequency_dtype == "f4":
                freq_vec = np.frombuffer(rec4[:4*nfreq], dtype=np.float32).astype(np.float64)
            else:
                freq_vec = np.frombuffer(rec4[:8*nfreq], dtype=np.float64).astype(np.float64)
            if freq_vec.size and valid_frequency(float(freq_vec[0])):
                frequency = float(freq_vec[0])
            bearings = np.frombuffer(read_rec(5)[:8*ntheta], dtype=np.float64)
            if plot_type and plot_type.startswith("TL"):
                source_x_endpoints = np.frombuffer(read_rec(6)[:16], dtype=np.float64)
                source_x = np.linspace(source_x_endpoints[0], source_x_endpoints[-1], nsx)
            else:
                source_x = np.frombuffer(read_rec(6)[:8*nsx], dtype=np.float64)
            if plot_type and plot_type.startswith("TL"):
                source_y_endpoints = np.frombuffer(read_rec(7)[:16], dtype=np.float64)
                source_y = np.linspace(source_y_endpoints[0], source_y_endpoints[-1], nsy)
            else:
                source_y = np.frombuffer(read_rec(7)[:8*nsy], dtype=np.float64)
        source_depths = np.frombuffer(read_rec(8)[:4*nsd], dtype=np.float32).astype(np.float64)
        rz = np.frombuffer(read_rec(9)[:4*nrz], dtype=np.float32)
        rr = np.frombuffer(read_rec(10)[:8*nrr], dtype=np.float64)
        nrcvrs_per_range = 1 if plot_type == "irregular" else nrz
        dtype = np.complex128 if native_pyacoustics_layout else np.complex64
        item = 16 if native_pyacoustics_layout else 8
        P = np.zeros((nfreq, nsx, nsy, ntheta, nsd, nrcvrs_per_range, nrr), dtype=dtype)
        rec = 11
        for ifreq in range(nfreq):
            for isx in range(nsx):
                for isy in range(nsy):
                    for itheta in range(ntheta):
                        for isd in range(nsd):
                            for ird in range(nrcvrs_per_range):
                                raw = read_rec(rec)[: item * nrr]
                                if native_pyacoustics_layout:
                                    P[ifreq, isx, isy, itheta, isd, ird, :] = np.frombuffer(raw, dtype=dtype)
                                else:
                                    values = np.frombuffer(raw, dtype="<f4", count=2*nrr)
                                    P[ifreq, isx, isy, itheta, isd, ird, :] = values[0::2] + 1j * values[1::2]
                                rec += 1
        selected_frequency_index = 0
        if frequency_hz is not None and freq_vec.size:
            selected_frequency_index = int(np.nanargmin(np.abs(freq_vec - float(frequency_hz))))
        selected_source_x_index = 0
        if source_x_m is not None and source_x.size:
            selected_source_x_index = int(np.nanargmin(np.abs(source_x - float(source_x_m))))
        selected_source_y_index = 0
        if source_y_m is not None and source_y.size:
            selected_source_y_index = int(np.nanargmin(np.abs(source_y - float(source_y_m))))
        selected_frequency = (
            float(freq_vec[selected_frequency_index])
            if freq_vec.size
            else frequency
        )
        return ATShadeData(
            pressure=P[selected_frequency_index, selected_source_x_index, selected_source_y_index],
            receiver_depths_m=rz,
            receiver_ranges_m=rr,
            source_depths_m=source_depths,
            bearings_deg=bearings,
            source_x_m=source_x,
            source_y_m=source_y,
            pressure_full=P[selected_frequency_index],
            pressure_frequency_full=P,
            frequency_vector_hz=freq_vec,
            plot_type=plot_type,
            attenuation=attenuation,
            title=title,
            frequency_hz=selected_frequency,
            selected_frequency_index=selected_frequency_index,
            selected_source_x_index=selected_source_x_index,
            selected_source_y_index=selected_source_y_index,
        )


def read_at_shd(filename):
    shade = read_at_shd_full(filename)
    if shade is None:
        return None, None, None
    return shade.pressure[0, 0], shade.receiver_depths_m, shade.receiver_ranges_m


def write_at_shd(
    filename,
    pressure,
    depths_m,
    ranges_m,
    *,
    title="PyAcoustics native shade",
    frequency=0.0,
    source_depths_m=None,
    bearings_deg=None,
    source_x_m=None,
    source_y_m=None,
    plot_type="rectilin",
    attenuation=0.0,
    frequencies_hz=None,
):
    """Write a minimal AT-compatible binary ``.shd`` pressure file.

    The layout follows AT ``RWSHDFile.WriteHeader`` direct-access records:
    record 3 stores the receiver grid dimensions plus nominal frequency and
    attenuation, records 4-10 store frequency/source/receiver grids, and
    subsequent records store one single-precision complex pressure row per
    receiver depth.
    Pressure may be 2D ``(depth, range)``, 4D ``(bearing, source_depth,
    depth, range)``, 6D ``(source_x, source_y, bearing, source_depth,
    depth, range)``, or 7D with a leading frequency axis.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    pressure = np.asarray(pressure, dtype=np.complex128)
    depths_m = np.asarray(depths_m, dtype=np.float32)
    ranges_m = np.asarray(ranges_m, dtype=np.float64)
    source_depths_m = np.array([0.0], dtype=np.float32) if source_depths_m is None else np.asarray(source_depths_m, dtype=np.float32)
    bearings_deg = np.array([0.0], dtype=np.float64) if bearings_deg is None else np.asarray(bearings_deg, dtype=np.float64)
    source_x_m = np.array([0.0], dtype=np.float64) if source_x_m is None else np.asarray(source_x_m, dtype=np.float64)
    source_y_m = np.array([0.0], dtype=np.float64) if source_y_m is None else np.asarray(source_y_m, dtype=np.float64)
    frequencies_hz = (
        np.array([float(frequency)], dtype=np.float64)
        if frequencies_hz is None
        else np.asarray(frequencies_hz, dtype=np.float64)
    )
    plot_type_name = str(plot_type).strip()
    is_irregular = plot_type_name == "irregular"
    is_tl_compressed = plot_type_name.startswith("TL")
    if is_irregular and depths_m.size != ranges_m.size:
        raise ValueError("irregular .shd output requires one receiver depth per receiver range")
    pressure_depth_count = 1 if is_irregular else depths_m.size
    nominal_frequency = float(frequency)
    if (not np.isfinite(nominal_frequency) or nominal_frequency <= 0.0) and frequencies_hz.size:
        nominal_frequency = float(frequencies_hz[0])
    if pressure.ndim == 2:
        pressure_7d = pressure[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
    elif pressure.ndim == 4:
        pressure_7d = pressure[np.newaxis, np.newaxis, np.newaxis, :, :, :, :]
    elif pressure.ndim == 6:
        pressure_7d = pressure[np.newaxis, :, :, :, :, :, :]
    elif pressure.ndim == 7:
        pressure_7d = pressure
    else:
        raise ValueError(
            "pressure must be shaped (depth, range), "
            "(bearing, source_depth, depth, range), or "
            "(source_x, source_y, bearing, source_depth, depth, range), or "
            "(frequency, source_x, source_y, bearing, source_depth, depth, range)"
        )
    if pressure_7d.shape != (
        frequencies_hz.size,
        source_x_m.size,
        source_y_m.size,
        bearings_deg.size,
        source_depths_m.size,
        pressure_depth_count,
        ranges_m.size,
    ):
        raise ValueError("pressure shape must match depths_m and ranges_m")

    nfreq, nsx, nsy, ntheta, nsd, nrz, nrr = pressure_7d.shape
    nrz_header = int(depths_m.size)
    source_x_record_bytes = 16 if is_tl_compressed else 8 * nsx
    source_y_record_bytes = 16 if is_tl_compressed else 8 * nsy
    rec_size = max(
        164,
        44,
        8 * nfreq,
        8 * ntheta,
        source_x_record_bytes,
        source_y_record_bytes,
        4 * nsd,
        4 * nrz_header,
        8 * nrr,
    )
    rec_size = int(np.ceil(rec_size / 4.0) * 4)
    lrecl = rec_size // 4

    def pack_record(payload=b""):
        if len(payload) > rec_size:
            raise ValueError("record payload exceeds .shd record size")
        return payload + b"\x00" * (rec_size - len(payload))

    safe_title = str(title).encode("ascii", errors="ignore")[:80].ljust(80, b" ")
    safe_plot_type = str(plot_type).encode("ascii", errors="ignore")[:10].ljust(10, b" ")
    records = [bytearray(rec_size) for _ in range(10 + nfreq * nsx * nsy * ntheta * nsd * nrz)]
    records[0][:4] = struct.pack("<i", lrecl)
    records[0][4:84] = safe_title
    records[1][:10] = safe_plot_type
    records[2][:44] = struct.pack(
        "<7idd",
        int(nfreq),
        int(ntheta),
        int(nsx),
        int(nsy),
        int(nsd),
        nrz_header,
        int(nrr),
        nominal_frequency,
        float(attenuation),
    )
    records[3][: 8 * nfreq] = frequencies_hz.astype("<f8").tobytes()
    records[4][: 8 * ntheta] = bearings_deg.astype("<f8").tobytes()
    if is_tl_compressed:
        records[5][:16] = np.array([source_x_m[0], source_x_m[-1]], dtype="<f8").tobytes()
        records[6][:16] = np.array([source_y_m[0], source_y_m[-1]], dtype="<f8").tobytes()
    else:
        records[5][: 8 * nsx] = source_x_m.astype("<f8").tobytes()
        records[6][: 8 * nsy] = source_y_m.astype("<f8").tobytes()
    records[7][: 4 * nsd] = source_depths_m.astype("<f4").tobytes()
    records[8][: 4 * nrz_header] = depths_m.astype("<f4").tobytes()
    records[9][: 8 * nrr] = ranges_m.astype("<f8").tobytes()
    rec = 10
    for ifreq in range(nfreq):
        for isx in range(nsx):
            for isy in range(nsy):
                for itheta in range(ntheta):
                    for isd in range(nsd):
                        for ird in range(nrz):
                            records[rec][: 8 * nrr] = pressure_7d[ifreq, isx, isy, itheta, isd, ird].astype("<c8").tobytes()
                            rec += 1

    with path.open("wb") as f:
        for record in records:
            f.write(pack_record(bytes(record)))

def read_at_ray_records(filename):
    """Read AT ``.ray`` records with launch angle and bounce metadata."""
    rays = []
    lines = _at_data_lines(filename)
    if not lines:
        return []
    idx = 3
    if idx >= len(lines):
        return []
    ray_counts = _at_numeric_values(lines[idx])
    if ray_counts.size == 0:
        return []
    num_rays = int(ray_counts[0])
    idx += 1
    coordinate_system = "rz"
    while idx < len(lines):
        if "'rz'" in lines[idx] or "'xy'" in lines[idx]:
            coordinate_system = _at_quoted_or_token(lines[idx])
            break
        if "'xyz'" in lines[idx]:
            coordinate_system = "xyz"
            break
        idx += 1
    idx += 1
    for _ in range(num_rays):
        if idx >= len(lines):
            break
        launch_tokens = _at_numeric_values(lines[idx])
        if launch_tokens.size == 0:
            break
        if coordinate_system == "xyz" and launch_tokens.size == 1:
            # AT Bellhop3D writes one launch angle per alpha block in radians;
            # beta is implicit and can be recovered from the first ray segment.
            launch_angle = float(np.degrees(launch_tokens[0]))
            launch_azimuth = None
        else:
            launch_angle = float(launch_tokens[0])
            launch_azimuth = float(launch_tokens[1]) if coordinate_system == "xyz" and launch_tokens.size > 1 else None
        idx += 1
        if idx >= len(lines):
            break
        metadata = _at_numeric_values(lines[idx])
        if metadata.size == 0:
            break
        npts = int(metadata[0])
        num_top_bounces = int(metadata[1]) if metadata.size > 1 else 0
        num_bottom_bounces = int(metadata[2]) if metadata.size > 2 else 0
        idx += 1
        c1, c2, c3 = [], [], []
        for _ in range(npts):
            if idx >= len(lines):
                break
            parts = _at_numeric_values(lines[idx])
            if parts.size >= 3:
                c1.append(float(parts[0]))
                c2.append(float(parts[1]))
                c3.append(float(parts[2]))
            elif parts.size >= 2:
                c1.append(float(parts[0]))
                c2.append(float(parts[1]))
                c3.append(1.0)
            idx += 1
        if c1:
            c1_arr = np.array(c1)
            c2_arr = np.array(c2)
            c3_arr = np.array(c3)
            if coordinate_system == "xyz":
                ranges_m = c1_arr
                depths_m = c3_arr
                amplitudes = np.ones_like(c1_arr, dtype=float)
                y_m = c2_arr
                if launch_azimuth is None and c1_arr.size >= 2:
                    dx = float(c1_arr[1] - c1_arr[0])
                    dy = float(c2_arr[1] - c2_arr[0])
                    if np.hypot(dx, dy) > np.finfo(float).eps:
                        launch_azimuth = float(np.degrees(np.arctan2(dy, dx)))
            else:
                ranges_m = c1_arr
                depths_m = c2_arr
                amplitudes = c3_arr
                y_m = None
            rays.append(
                ATRayRecord(
                    launch_angle_deg=launch_angle,
                    ranges_m=ranges_m,
                    depths_m=depths_m,
                    amplitudes=amplitudes,
                    num_top_bounces=num_top_bounces,
                    num_bottom_bounces=num_bottom_bounces,
                    coordinate_system=coordinate_system,
                    y_m=y_m,
                    launch_azimuth_deg=launch_azimuth,
                )
            )
    return rays

def read_at_ray(filename):
    """Read AT ``.ray`` coordinates using the legacy tuple return shape."""
    return [
        (ray.ranges_m, ray.y_m, ray.depths_m)
        if ray.coordinate_system == "xyz" and ray.y_m is not None
        else (ray.ranges_m, ray.depths_m, ray.amplitudes)
        for ray in read_at_ray_records(filename)
    ]


def write_at_ray_records(
    filename,
    records: list[ATRayRecord],
    *,
    title: str = "BELLHOP",
    frequency: float = 0.0,
    top_depth: float = 0.0,
    bottom_depth: float = 0.0,
) -> None:
    """Write AT-compatible ``.ray`` records.

    ``ATRayRecord`` defaults to 2D ``rz`` coordinates. Set
    ``coordinate_system="xyz"`` and provide ``y_m`` to write a Bellhop3D-style
    ray file.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_title = title.replace("'", "")
    coordinate_system = "xyz" if any(record.coordinate_system == "xyz" or record.y_m is not None for record in records) else "rz"
    with path.open("w") as f:
        f.write(f"'{safe_title}'\n")
        f.write(f"{frequency:.9g}\n")
        f.write("1 1 1\n")
        f.write(f"{len(records)} 1\n")
        f.write(f"{top_depth:.9g}\n")
        f.write(f"{bottom_depth:.9g}\n")
        f.write(f"'{coordinate_system}'\n")
        for record in records:
            npts = len(record.ranges_m)
            if len(record.depths_m) != npts:
                raise ValueError("ATRayRecord ranges_m and depths_m must have the same length")
            record_coordinate_system = "xyz" if record.coordinate_system == "xyz" or record.y_m is not None else "rz"
            if record_coordinate_system != coordinate_system:
                raise ValueError("All ATRayRecord entries must use the same coordinate system")
            if coordinate_system == "xyz":
                if record.y_m is None or len(record.y_m) != npts:
                    raise ValueError("3D ATRayRecord entries require y_m with the same length as ranges_m")
                launch_azimuth = 0.0 if record.launch_azimuth_deg is None else record.launch_azimuth_deg
                f.write(f"{record.launch_angle_deg:.9g} {launch_azimuth:.9g}\n")
            else:
                f.write(f"{record.launch_angle_deg:.9g}\n")
            f.write(f"{npts:d} {record.num_top_bounces:d} {record.num_bottom_bounces:d}\n")
            if coordinate_system == "xyz":
                for x, y, z in zip(record.ranges_m, record.y_m, record.depths_m):
                    f.write(f"{float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
            else:
                for r, z in zip(record.ranges_m, record.depths_m):
                    f.write(f"{float(r):.9g} {float(z):.9g}\n")


def read_at_ssp3d(filename):
    """
    Reads the Bellhop3D .ssp file used by the H/hexahedral option.
    Returns x/y in meters, z in meters, and c_grid shaped as (nx, ny, nz).
    """
    values = []
    for line in Path(filename).read_text().splitlines():
        values.extend(_at_numeric_values(line))

    idx = 0
    nx = int(values[idx]); idx += 1
    x_arr = np.array(values[idx:idx + nx], dtype=np.float64) * 1000.0
    idx += nx

    ny = int(values[idx]); idx += 1
    y_arr = np.array(values[idx:idx + ny], dtype=np.float64) * 1000.0
    idx += ny

    nz = int(values[idx]); idx += 1
    z_arr = np.array(values[idx:idx + nz], dtype=np.float64)
    idx += nz

    raw = np.array(values[idx:idx + nx * ny * nz], dtype=np.float64)
    if raw.size != nx * ny * nz:
        raise ValueError(f"Expected {nx * ny * nz} sound-speed samples in {filename}, found {raw.size}")

    c_grid = np.zeros((nx, ny, nz), dtype=np.float64)
    offset = 0
    for iz in range(nz):
        for iy in range(ny):
            c_grid[:, iy, iz] = raw[offset:offset + nx]
            offset += nx

    return x_arr, y_arr, z_arr, c_grid

def read_at_ssp2d(filename, *, range_unit="km"):
    """Read a Bellhop 2D range-dependent ``.ssp`` sidecar.

    AT stores the range count, range vector, then a depth-major sound-speed
    matrix. The depth grid is defined by the matching ``.env`` file, so this
    reader preserves only the range axis and matrix available in the sidecar.
    """

    lines = _at_data_lines(filename)
    if len(lines) < 3:
        raise ValueError(f"AT 2D SSP file is too short: {filename}")
    count_values = _at_numeric_values(lines[0])
    if count_values.size != 1:
        raise ValueError(f"Expected one range count in AT 2D SSP file: {filename}")
    n_ranges = int(count_values[0])
    if n_ranges < 1:
        raise ValueError(f"AT 2D SSP range count must be positive: {filename}")

    ranges = _at_numeric_values(lines[1])
    if ranges.size != n_ranges:
        raise ValueError(f"Expected {n_ranges} ranges in AT 2D SSP file: {filename}")
    if range_unit == "km":
        ranges = ranges * 1000.0
    elif range_unit != "m":
        raise ValueError("range_unit must be either 'km' or 'm'")

    rows = [_at_numeric_values(line) for line in lines[2:]]
    if not rows:
        raise ValueError(f"AT 2D SSP file has no sound-speed rows: {filename}")
    bad_rows = [index for index, row in enumerate(rows) if row.size != n_ranges]
    if bad_rows:
        first = bad_rows[0] + 3
        raise ValueError(f"Expected {n_ranges} sound-speed values on line {first} in AT 2D SSP file: {filename}")
    c_matrix = np.vstack(rows).astype(np.float64)
    return ATRangeDependentSSP(ranges_m=np.asarray(ranges, dtype=np.float64), c_matrix=c_matrix)

def read_at_arr(filename):
    """Read an AT ASCII .arr file into the postprocessing ArrivalData model."""
    from pyacoustics.postprocessing.delay_and_sum import read_arrivals_ascii

    return read_arrivals_ascii(filename)

def _env_int_at(lines, idx):
    if idx >= len(lines):
        return None
    values = _at_numeric_values(lines[idx])
    if values.size == 0:
        return None
    return int(values[0])

def _env_float_at(lines, idx, pos=0):
    if idx >= len(lines):
        return None
    values = _at_numeric_values(lines[idx])
    if values.size <= pos:
        return None
    return float(values[pos])

def _at_title_line(line):
    if re.search(r"['\"]", line):
        return _at_quoted_or_token(line)
    return _strip_at_comment(line).strip()

def _ram_in_blocks_after_bathymetry(lines):
    idx = 5
    bathymetry = []
    while idx < len(lines):
        values = _at_numeric_values(lines[idx])
        if values.size >= 2 and values[0] == -1 and values[1] == -1:
            idx += 1
            break
        if values.size >= 2:
            bathymetry.append(values[:2])
        idx += 1

    blocks = []
    current = []
    while idx < len(lines):
        values = _at_numeric_values(lines[idx])
        if values.size >= 2 and values[0] == -1 and values[1] == -1:
            blocks.append(current)
            current = []
        else:
            current.append(values)
        idx += 1
    if current:
        blocks.append(current)
    return np.asarray(bathymetry, dtype=np.float64), blocks

def _ram_block_value_range(blocks, *, offset, profile_count, column, skip_profile_range=False):
    values = []
    for profile_idx in range(profile_count):
        block_idx = offset + 4 * profile_idx
        if block_idx >= len(blocks):
            break
        block = blocks[block_idx]
        start = 1 if skip_profile_range and profile_idx > 0 else 0
        for row in block[start:]:
            if row.size > column:
                values.append(float(row[column]))
    if not values:
        return 0, None, None
    arr = np.asarray(values, dtype=np.float64)
    return int(arr.size), float(arr.min()), float(arr.max())

def read_at_ram_in_summary(filename):
    """Read stable high-level metadata from a RAM ``ram.in`` input file.

    RAM inputs are block-oriented: after the bathymetry section, each range
    profile contributes water SSP, bottom sound-speed, density and attenuation
    blocks. This parser preserves the file-level metadata and block sizes used
    for AT fixture compatibility checks without attempting to run the PE model.
    """

    lines = _at_data_lines(filename)
    if len(lines) < 10:
        raise ValueError(f"AT RAM input is too short: {filename}")

    title = _at_title_line(lines[0])
    frequency = _at_numeric_values(lines[1])
    range_grid = _at_numeric_values(lines[2])
    depth_grid = _at_numeric_values(lines[3])
    profile_setup = _at_numeric_values(lines[4])
    if frequency.size < 3 or range_grid.size < 3 or depth_grid.size < 4 or profile_setup.size < 4:
        raise ValueError(f"AT RAM input has incomplete header metadata: {filename}")

    bathymetry, blocks = _ram_in_blocks_after_bathymetry(lines)
    if bathymetry.size == 0:
        raise ValueError(f"AT RAM input has no bathymetry points: {filename}")
    profile_count = max(1, len(blocks) // 4)

    profile_ranges = []
    for profile_idx in range(1, profile_count):
        block_idx = 4 * profile_idx
        if block_idx < len(blocks) and blocks[block_idx] and blocks[block_idx][0].size:
            profile_ranges.append(float(blocks[block_idx][0][0]))
    ssp_count, c_min, c_max = _ram_block_value_range(
        blocks,
        offset=0,
        profile_count=profile_count,
        column=1,
        skip_profile_range=True,
    )
    bottom_count, bottom_min, bottom_max = _ram_block_value_range(blocks, offset=1, profile_count=profile_count, column=1)
    density_count, density_min, density_max = _ram_block_value_range(blocks, offset=2, profile_count=profile_count, column=1)
    attenuation_count, attenuation_min, attenuation_max = _ram_block_value_range(
        blocks,
        offset=3,
        profile_count=profile_count,
        column=1,
    )

    profile_range_arr = np.asarray(profile_ranges, dtype=np.float64)
    return ATRAMInSummary(
        title=title,
        frequency_hz=float(frequency[0]),
        source_depth_m=float(frequency[1]),
        receiver_depth_m=float(frequency[2]),
        rmax_m=float(range_grid[0]),
        dr_m=float(range_grid[1]),
        receiver_range_decimation=int(range_grid[2]),
        zmax_m=float(depth_grid[0]),
        dz_m=float(depth_grid[1]),
        depth_decimation=int(depth_grid[2]),
        plot_depth_m=float(depth_grid[3]),
        reference_sound_speed_mps=float(profile_setup[0]),
        range_dependent_profile_count=profile_count,
        sediment_layer_count=int(profile_setup[2]),
        source_range_m=float(profile_setup[3]),
        bathymetry_point_count=int(bathymetry.shape[0]),
        ssp_point_count=ssp_count,
        bottom_sound_speed_point_count=bottom_count,
        density_point_count=density_count,
        attenuation_point_count=attenuation_count,
        profile_range_count=int(profile_range_arr.size),
        bathymetry_range_min_m=float(bathymetry[:, 0].min()),
        bathymetry_range_max_m=float(bathymetry[:, 0].max()),
        bathymetry_depth_min_m=float(bathymetry[:, 1].min()),
        bathymetry_depth_max_m=float(bathymetry[:, 1].max()),
        profile_range_min_m=float(profile_range_arr.min()) if profile_range_arr.size else None,
        profile_range_max_m=float(profile_range_arr.max()) if profile_range_arr.size else None,
        sound_speed_min_mps=c_min,
        sound_speed_max_mps=c_max,
        bottom_sound_speed_min_mps=bottom_min,
        bottom_sound_speed_max_mps=bottom_max,
        density_min_gpcm3=density_min,
        density_max_gpcm3=density_max,
        attenuation_min_db_per_wavelength=attenuation_min,
        attenuation_max_db_per_wavelength=attenuation_max,
    )

_AT_MATLAB_COMMANDS = (
    "bellhop",
    "bellhopM",
    "bellhop3d",
    "bounce",
    "field",
    "field3d",
    "field_noise",
    "krakel",
    "kraken",
    "krakenM",
    "krakenc",
    "kraken_rd",
    "modal_noise_diag",
    "modal_noise_full",
    "simplePE",
    "scooter",
    "scooterM",
    "sparc",
    "spectral_noise",
    "plotarr",
    "plotati",
    "plotbdry3d",
    "plotbty",
    "plotenv",
    "plotgrn",
    "plotmovie",
    "plotray",
    "plotray3d",
    "plotshd",
    "plotshdpol",
    "plotssp",
    "plotts",
    "read_arrivals_asc",
    "read_arrivals_bin",
    "read_modes",
    "read_ram_tlgrid",
    "read_shd",
    "read_shd_bin",
    "read_shd_asc",
    "read_ts",
    "write_fieldflp",
    "write_shd",
    "delayandsum",
    "stack",
    "make_bdry",
    "makeshd",
    "makebty",
    "makeenv",
    "writebdry3d",
)

_AT_MATLAB_COMMAND_BY_LOWER = {command.lower(): command for command in _AT_MATLAB_COMMANDS}


def _strip_matlab_comment(line: str) -> str:
    in_single = False
    idx = 0
    while idx < len(line):
        char = line[idx]
        if char == "'":
            if in_single and idx + 1 < len(line) and line[idx + 1] == "'":
                idx += 2
                continue
            in_single = not in_single
        elif char == "%" and not in_single:
            return line[:idx]
        idx += 1
    return line


def _canonical_at_matlab_command(name: str) -> str | None:
    return _AT_MATLAB_COMMAND_BY_LOWER.get(name.lower().removesuffix(".exe"))


def _matlab_external_command_counts(code_lines: list[str]) -> dict[str, int]:
    executable_variables: dict[str, str] = {}
    counts: dict[str, int] = {}

    for code in code_lines:
        assignment = re.search(
            r"(?i)\b([A-Za-z]\w*)\s*=\s*which\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
            code,
        )
        if assignment:
            command = _canonical_at_matlab_command(Path(assignment.group(2)).name)
            if command is not None:
                executable_variables[assignment.group(1)] = command

    command_pattern = "|".join(re.escape(command) for command in sorted(_AT_MATLAB_COMMANDS, key=len, reverse=True))
    bang_pattern = re.compile(rf"(?i)!\s*\"?\s*({command_pattern})(?:\.exe)?\b")
    system_pattern = re.compile(
        rf"(?i)\b(?:system|dos|unix)\s*\(\s*['\"]\s*({command_pattern})(?:\.exe)?\b"
    )

    for code in code_lines:
        for pattern in (bang_pattern, system_pattern):
            for match in pattern.finditer(code):
                command = _canonical_at_matlab_command(match.group(1))
                if command is not None:
                    counts[command] = counts.get(command, 0) + 1

        if re.search(r"(?i)\beval\s*\(", code):
            for variable, command in executable_variables.items():
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(variable)}(?![A-Za-z0-9_])", code):
                    counts[command] = counts.get(command, 0) + 1

    return counts


def read_at_matlab_script_summary(filename):
    """Read command inventory metadata from an AT Matlab/Octave ``.m`` script."""

    path = Path(filename)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    code_lines = []
    comment_line_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%"):
            comment_line_count += 1
        code = _strip_matlab_comment(line).strip()
        if code:
            code_lines.append(code)

    code_text = "\n".join(code_lines)
    function_names = tuple(
        match.group(1) or match.group(2)
        for match in re.finditer(
            r"(?im)^\s*function\s+(?:\[[^\]]*\]\s*=\s*|[A-Za-z]\w*\s*=\s*)?([A-Za-z]\w*)|^\s*function\s+([A-Za-z]\w*)",
            code_text,
        )
    )
    command_counts = {
        command: len(re.findall(rf"(?<![A-Za-z0-9_]){re.escape(command)}(?![A-Za-z0-9_])", code_text, flags=re.IGNORECASE))
        for command in _AT_MATLAB_COMMANDS
    }
    command_counts = {command: count for command, count in command_counts.items() if count}
    external_command_counts = _matlab_external_command_counts(code_lines)
    referenced_extensions = tuple(
        sorted(
            {
                f".{match.group(1).lower()}"
                for match in re.finditer(
                    r"['\"][^'\"]+\.([A-Za-z0-9_]+)['\"]",
                    code_text,
                )
            }
        )
    )
    return ATMatlabScriptSummary(
        path=path,
        line_count=len(lines),
        code_line_count=len(code_lines),
        comment_line_count=comment_line_count,
        function_count=len(function_names),
        function_names=function_names,
        at_command_counts=command_counts,
        external_command_counts=external_command_counts,
        referenced_extensions=referenced_extensions,
    )


def _strip_fortran_comment(line: str) -> str:
    in_single = False
    in_double = False
    idx = 0
    while idx < len(line):
        char = line[idx]
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "!" and not in_single and not in_double:
            return line[:idx]
        idx += 1
    return line


def _fortran_names(pattern: str, code_text: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in re.finditer(pattern, code_text, flags=re.IGNORECASE | re.MULTILINE))


def read_at_fortran_source_summary(filename):
    """Read static inventory metadata from an AT Fortran source file."""

    path = Path(filename)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    code_lines = []
    comment_line_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!"):
            comment_line_count += 1
        code = _strip_fortran_comment(line).strip()
        if code:
            code_lines.append(code)

    code_text = "\n".join(code_lines)
    module_names = _fortran_names(r"^\s*module\s+(?!procedure\b)([A-Za-z]\w*)\b", code_text)
    used_modules = tuple(
        sorted(
            {
                match.group(1)
                for match in re.finditer(
                    r"^\s*use(?:\s*,\s*(?:non_)?intrinsic\s*)?(?:::)?\s*([A-Za-z]\w*)\b",
                    code_text,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
            },
            key=str.lower,
        )
    )
    return ATFortranSourceSummary(
        path=path,
        line_count=len(lines),
        code_line_count=len(code_lines),
        comment_line_count=comment_line_count,
        program_names=_fortran_names(r"^\s*program\s+([A-Za-z]\w*)\b", code_text),
        module_names=module_names,
        subroutine_names=_fortran_names(r"^\s*(?:recursive\s+)?subroutine\s+([A-Za-z]\w*)\b", code_text),
        function_names=_fortran_names(
            r"^\s*(?:(?:recursive|pure|elemental)\s+)*(?:(?:integer|real|complex|logical|character|double\s+precision|type\s*\([^)]*\))\s+)?function\s+([A-Za-z]\w*)\b",
            code_text,
        ),
        used_modules=used_modules,
    )


def _read_compact_bellhop3d_env_summary(filename, lines):
    frequency_values = _at_numeric_values(lines[2])
    if frequency_values.size == 0:
        raise ValueError(f"AT .env frequency line is empty: {filename}")
    beam_values = _at_numeric_values(lines[3]) if len(lines) > 3 else np.array([])
    beam_bearing_values = _at_numeric_values(lines[4]) if len(lines) > 4 else np.array([])
    receiver_bearing_values = _at_numeric_values(lines[9]) if len(lines) > 9 else np.array([])
    receiver_range_values = _at_numeric_values(lines[10]) if len(lines) > 10 else np.array([])
    return ATEnvSummary(
        title=_at_title_line(lines[0]),
        frequency_hz=float(frequency_values[0]),
        nmedia=_env_int_at(lines, 5) or 0,
        ssp_option=None,
        top_option=None,
        bottom_option=None,
        model_family="bellhop3d",
        run_type=_at_quoted_or_token(lines[11]) if len(lines) > 11 else None,
        source_depth_count=1,
        receiver_depth_count=1,
        receiver_range_count=int(receiver_range_values[2]) if receiver_range_values.size > 2 else None,
        receiver_bearing_count=int(receiver_bearing_values[2]) if receiver_bearing_values.size > 2 else None,
        beam_count=int(beam_values[2]) if beam_values.size > 2 else None,
        beam_bearing_count=int(beam_bearing_values[2]) if beam_bearing_values.size > 2 else None,
        box_range_km=float(receiver_range_values[1]) if receiver_range_values.size > 1 else None,
    )

def read_at_env_summary(filename):
    """Read stable high-level metadata from an AT ``.env`` file.

    This deliberately avoids full environment reconstruction. AT ``.env``
    grammar differs across Bellhop, Bellhop3D, Kraken and Field workflows, but
    the title/frequency/options/grid-count metadata is stable enough to use as
    a file-format compatibility gate.
    """

    lines = _at_data_lines(filename)
    if len(lines) < 5:
        raise ValueError(f"AT .env file is too short: {filename}")

    if len(lines) >= 12 and _at_numeric_values(lines[1]).size == 0 and _at_numeric_values(lines[2]).size > 0:
        return _read_compact_bellhop3d_env_summary(filename, lines)

    title = _at_title_line(lines[0])
    frequency_values = _at_numeric_values(lines[1])
    if frequency_values.size == 0:
        raise ValueError(f"AT .env frequency line is empty: {filename}")
    frequency_hz = float(frequency_values[0])
    nmedia = _at_count(lines[2])
    top_token = _at_quoted_or_token(lines[3])
    ssp_option = top_token[0] if top_token else None
    top_option = top_token[1] if len(top_token) > 1 else None

    bottom_idx = None
    for idx in range(4, len(lines)):
        if re.search(r"['\"]", lines[idx]):
            bottom_idx = idx
            break
    if bottom_idx is None:
        return ATEnvSummary(
            title=title,
            frequency_hz=frequency_hz,
            nmedia=nmedia,
            ssp_option=ssp_option,
            top_option=top_option,
            bottom_option=None,
            model_family="kraken",
        )

    bottom_token = _at_quoted_or_token(lines[bottom_idx])
    bottom_option = bottom_token[0] if bottom_token else None
    quoted_after_bottom = [
        (idx, _at_quoted_or_token(lines[idx]))
        for idx in range(bottom_idx + 1, len(lines))
        if re.search(r"['\"]", lines[idx])
    ]
    run_idx = quoted_after_bottom[-1][0] if quoted_after_bottom else None
    run_type = quoted_after_bottom[-1][1] if quoted_after_bottom else None

    grid_idx = bottom_idx + 1
    if bottom_option == "A" and grid_idx < len(lines) and not re.search(r"['\"]", lines[grid_idx]):
        grid_idx += 1

    if run_type is None:
        return ATEnvSummary(
            title=title,
            frequency_hz=frequency_hz,
            nmedia=nmedia,
            ssp_option=ssp_option,
            top_option=top_option,
            bottom_option=bottom_option,
            model_family="kraken",
            source_depth_count=_env_int_at(lines, grid_idx + 2),
            receiver_depth_count=_env_int_at(lines, grid_idx + 4),
            box_range_km=_env_float_at(lines, grid_idx + 1),
        )

    if run_type.rstrip().endswith("3"):
        box_values = _at_numeric_values(lines[run_idx + 5]) if run_idx + 5 < len(lines) else np.array([])
        return ATEnvSummary(
            title=title,
            frequency_hz=frequency_hz,
            nmedia=nmedia,
            ssp_option=ssp_option,
            top_option=top_option,
            bottom_option=bottom_option,
            model_family="bellhop3d",
            run_type=run_type,
            source_x_count=_env_int_at(lines, grid_idx),
            source_y_count=_env_int_at(lines, grid_idx + 2),
            source_depth_count=_env_int_at(lines, grid_idx + 4),
            receiver_depth_count=_env_int_at(lines, grid_idx + 6),
            receiver_range_count=_env_int_at(lines, grid_idx + 8),
            receiver_bearing_count=_env_int_at(lines, grid_idx + 10),
            beam_count=_env_int_at(lines, run_idx + 1),
            beam_bearing_count=_env_int_at(lines, run_idx + 3),
            box_range_km=float(box_values[1]) if box_values.size > 1 else None,
            box_y_km=float(box_values[2]) if box_values.size > 2 else None,
            box_z_m=float(box_values[3]) if box_values.size > 3 else None,
        )

    box_values = _at_numeric_values(lines[run_idx + 3]) if run_idx + 3 < len(lines) else np.array([])
    return ATEnvSummary(
        title=title,
        frequency_hz=frequency_hz,
        nmedia=nmedia,
        ssp_option=ssp_option,
        top_option=top_option,
        bottom_option=bottom_option,
        model_family="bellhop",
        run_type=run_type,
        source_depth_count=_env_int_at(lines, grid_idx),
        receiver_depth_count=_env_int_at(lines, grid_idx + 2),
        receiver_range_count=_env_int_at(lines, grid_idx + 4),
        beam_count=_env_int_at(lines, run_idx + 1),
        box_z_m=float(box_values[1]) if box_values.size > 1 else None,
        box_range_km=float(box_values[2]) if box_values.size > 2 else None,
    )

def read_at_source_beam_pattern(filename):
    """Read an AT source beam pattern ``.sbp`` file."""

    lines = _at_data_lines(filename)
    if len(lines) < 2:
        raise ValueError(f"AT source beam pattern file is too short: {filename}")
    count_values = _at_numeric_values(lines[0])
    if count_values.size != 1:
        raise ValueError(f"Expected one sample count in AT source beam pattern file: {filename}")
    n_rows = int(count_values[0])
    if n_rows < 1:
        raise ValueError(f"AT source beam pattern sample count must be positive: {filename}")
    rows = [_at_numeric_values(line) for line in lines[1:]]
    if len(rows) != n_rows or any(row.size != 2 for row in rows):
        raise ValueError(f"Expected {n_rows} angle/level rows in AT source beam pattern file: {filename}")
    raw = np.vstack(rows).astype(np.float64)
    return ATSourceBeamPattern(angles_deg=raw[:, 0], levels_db=raw[:, 1])


def write_at_source_beam_pattern(filename, pattern):
    """Write an AT source beam pattern ``.sbp`` file."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = np.asarray(pattern.angles_deg, dtype=np.float64)
    levels = np.asarray(pattern.levels_db, dtype=np.float64)
    if angles.shape != levels.shape:
        raise ValueError("source beam pattern angles_deg and levels_db must have the same shape")
    if angles.ndim != 1 or angles.size < 1:
        raise ValueError("source beam pattern requires at least one sample")
    with path.open("w") as handle:
        handle.write(f"{angles.size}\n")
        for angle, level in zip(angles, levels):
            handle.write(f"{angle:.9g} {level:.9g}\n")


def write_at_arr(filename, arrivals, **kwargs):
    """Write an AT ASCII .arr file from the postprocessing ArrivalData model."""
    from pyacoustics.postprocessing.delay_and_sum import write_arrivals_ascii

    return write_arrivals_ascii(filename, arrivals, **kwargs)

def _parse_prt_scalar(value: str) -> float | str:
    value = value.strip()
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?", value)
    if not match:
        return value
    number = match.group(0).replace("D", "E").replace("d", "e")
    try:
        return float(number)
    except ValueError:
        return value

def _parse_prt_key_values(lines: list[str]) -> dict[str, float | str]:
    params: dict[str, float | str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        left, right = stripped.split("=", 1)
        key = re.sub(r"\s+", " ", left.strip())
        if not key:
            continue
        params[key] = _parse_prt_scalar(right)
    return params

def _first_prt_match(text: str, pattern: str, cast=float):
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    value = match.group(1).replace("D", "E").replace("d", "e")
    try:
        return cast(value)
    except ValueError:
        return None

def _is_prt_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"_", "-"}

def _first_prt_match_any(text: str, patterns: list[str], cast=float):
    for pattern in patterns:
        value = _first_prt_match(text, pattern, cast)
        if value is not None:
            return value
    return None

def _first_prt_pair(text: str, pattern: str):
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None, None
    values = []
    for group in match.groups()[:2]:
        try:
            values.append(float(group.replace("D", "E").replace("d", "e")))
        except ValueError:
            values.append(None)
    return tuple(values)

def _first_prt_triplet(text: str, pattern: str):
    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return None, None, None
    values = []
    for group in match.groups()[:3]:
        try:
            values.append(float(group.replace("D", "E").replace("d", "e")))
        except ValueError:
            values.append(None)
    return tuple(values)

def _count_kraken_modes(lines: list[str]) -> int | None:
    rows = _kraken_mode_table_rows(lines)
    return len(rows) or None

def _kraken_mode_table_rows(lines: list[str]) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    in_table = False
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    mode_line = re.compile(
        rf"^\s*\d+\s+({number})\s+({number})\s+({number})\s+({number})(?:\s|$)"
    )
    for line in lines:
        lower = line.lower()
        if "phase speed" in lower and "group speed" in lower:
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("("):
            continue
        if _is_prt_separator(line) or stripped.lower().startswith("end of"):
            break
        match = mode_line.match(line)
        if match:
            rows.append(
                tuple(float(value.replace("D", "E").replace("d", "e")) for value in match.groups())
            )
    return rows

def _kraken_mode_table_ranges(lines: list[str]) -> tuple[float | None, ...]:
    rows = _kraken_mode_table_rows(lines)
    if not rows:
        return (None,) * 8
    columns = list(zip(*rows))
    ranges: list[float] = []
    for column in columns:
        ranges.extend([min(column), max(column)])
    return tuple(ranges)

def _kraken_mesh_multiplier_cpu_table(lines: list[str]) -> dict[int, float] | None:
    """Extract Kraken/KRAKENC mesh multiplier CPU timing rows from .prt logs."""
    rows: dict[int, float] = {}
    in_table = False
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    row_pattern = re.compile(rf"^\s*(\d+)\s+({number})\s*s?\s*$", re.IGNORECASE)
    for line in lines:
        lower = line.lower()
        if "mesh multiplier" in lower and "cpu" in lower:
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if "max." in lower and "root" in lower:
            continue
        if _is_prt_separator(line) or "phase speed" in lower or stripped.lower().startswith("end of"):
            break
        match = row_pattern.match(line)
        if match:
            rows[int(match.group(1))] = float(match.group(2).replace("D", "E").replace("d", "e"))
    return rows or None

def _prt_numeric_block_after(lines: list[str], header_pattern: str) -> list[float]:
    values: list[float] = []
    in_block = False
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    for line in lines:
        if not in_block:
            if re.search(header_pattern, line, re.IGNORECASE):
                in_block = True
            continue
        stripped = line.strip()
        if not stripped:
            if values:
                break
            continue
        if values and not re.match(r"(?:\.\.\.|[+-]?(?:\d|\.\d))", stripped):
            break
        if _is_prt_separator(line) or re.match(r"Number\s+of\b", stripped, re.IGNORECASE):
            break
        values.extend(float(match.replace("D", "E").replace("d", "e")) for match in re.findall(number, line))
    return values

def _prt_block_min_max(lines: list[str], header_pattern: str) -> tuple[float | None, float | None]:
    values = _prt_numeric_block_after(lines, header_pattern)
    if not values:
        return None, None
    return min(values), max(values)

def read_at_prt(filename):
    """Read an AT ``.prt`` log and extract common run metadata."""
    path = Path(filename)
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    warnings = []
    errors = []
    fatal_error_count = 0
    for line in lines:
        normalized = line.lower()
        if "warning" in normalized:
            warnings.append(line.strip())
        if "error" in normalized or "fatal" in normalized:
            if "0 error" not in normalized and "no error" not in normalized:
                errors.append(line.strip())
                if "fatal" in normalized:
                    fatal_error_count += 1

    title = None
    for line in lines:
        stripped = line.strip()
        if stripped and not _is_prt_separator(stripped):
            title = stripped
            break

    nonempty = [line.strip() for line in lines if line.strip() and not _is_prt_separator(line)]
    program = nonempty[0] if nonempty else None
    case_title = None
    if len(nonempty) > 1 and not re.match(r"(nominal\s+)?frequency\b", nonempty[1], re.IGNORECASE):
        case_title = nonempty[1]

    run_type_description = None
    beam_description = None
    source_description = None
    receiver_grid_description = None
    source_beam_description = None
    mode_addition_description = None
    calculation_description = None
    beam_shift_description = None
    for line in nonempty:
        lower = line.lower()
        if run_type_description is None and (
            " tl calculation" in lower or "arrivals calculation" in lower or "eigenray" in lower
        ):
            run_type_description = line
        elif beam_description is None and " beams" in lower:
            beam_description = line
        elif source_description is None and "source" in lower and not re.search(
            r"\b[xyz]-coordinates\b", lower
        ):
            source_description = line
        elif receiver_grid_description is None and "receiver grid" in lower:
            receiver_grid_description = line
        elif source_beam_description is None and "source" in lower and (
            "omnidirectional" in lower or "directional" in lower
        ):
            source_beam_description = line
        elif mode_addition_description is None and "mode addition" in lower:
            mode_addition_description = line
        elif calculation_description is None and (
            "range-independent calculation" in lower
            or "range-dependent calculation" in lower
            or lower == "3d calculation"
        ):
            calculation_description = line
        elif beam_shift_description is None and "beam shift" in lower:
            beam_shift_description = line

    max_arrivals = _first_prt_match(text, r"Maximum\s+#\s+of\s+arrivals\s*=\s*([+-]?\d+)", int)
    if max_arrivals is None:
        max_arrivals = _first_prt_match(text, r"MaxNArr\s*=\s*([+-]?\d+)", int)
    c_low_mps, c_high_mps = _first_prt_pair(
        text,
        r"cLow\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s*m/s\s+cHigh\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
    )
    attenuation_units_match = re.search(r"Attenuation\s+units\s*:\s*(.+)", text, re.IGNORECASE)
    attenuation_units = attenuation_units_match.group(1).strip() if attenuation_units_match else None
    field_option_match = re.search(r"Option\s*=\s*(\S+)", text, re.IGNORECASE)
    field_option = field_option_match.group(1).strip() if field_option_match else None
    field_alpha_min_deg, field_alpha_max_deg, field_alpha_count_raw = _first_prt_triplet(
        text,
        r"^\s*(?:Field\s+)?Alpha(?:\s+scan)?\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s+"
        r"([+-]?\d+)",
    )
    field_step_m, field_step_count_raw = _first_prt_pair(
        text,
        r"^\s*(?:Field\s+)?Step\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"
        r"(?:\s*m)?\s+(?:N(?:STEPS|Steps)\s*=\s*)?([+-]?\d+)",
    )
    field_alpha_count = int(field_alpha_count_raw) if field_alpha_count_raw is not None else None
    field_step_count = int(field_step_count_raw) if field_step_count_raw is not None else None
    source_depth_min_m, source_depth_max_m = _prt_block_min_max(
        lines, r"Source\s+z-coordinates,\s*Sz\s*\(m\)"
    )
    source_x_min_km, source_x_max_km = _prt_block_min_max(
        lines, r"Source\s+x-coordinates,\s*Sx\s*\(km\)"
    )
    source_y_min_km, source_y_max_km = _prt_block_min_max(
        lines, r"Source\s+y-coordinates,\s*Sy\s*\(km\)"
    )
    source_coordinate_x_m, source_coordinate_y_m, source_coordinate_z_m = _first_prt_triplet(
        text,
        r"^\s*xs\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
    )
    receiver_depth_min_m, receiver_depth_max_m = _prt_block_min_max(
        lines, r"Receiver\s+z-coordinates,\s*Rz\s*\(m\)"
    )
    receiver_range_min_km, receiver_range_max_km = _prt_block_min_max(
        lines, r"Receiver\s+r-coordinates,\s*Rr\s*\(km\)"
    )
    beam_angle_min_deg, beam_angle_max_deg = _prt_block_min_max(
        lines, r"Beam\s+take-off\s+angles\s*\(degrees\)"
    )
    beam_bearing_angle_min_deg, beam_bearing_angle_max_deg = _prt_block_min_max(
        lines, r"Number\s+of\s+beams\s+in\s+bearing\b"
    )
    receiver_theta_min_deg, receiver_theta_max_deg = _prt_block_min_max(
        lines, r"Receiver\s+theta-coordinates,\s*theta\s*\(degrees\)"
    )
    profile_range_min_km, profile_range_max_km = _prt_block_min_max(
        lines, r"Profile\s+ranges,\s*RProf\s*\(km\)"
    )
    receiver_range_offset_min_m, receiver_range_offset_max_m = _prt_block_min_max(
        lines, r"receiver\s+range\s+offsets\s*\(array\s+tilt\),\s*Rro\s*\(m\)"
    )
    bathymetry_x_min_km, bathymetry_x_max_km = _prt_block_min_max(
        lines, r"Number\s+of\s+bathymetry\s+points\s+in\s+x\b"
    )
    bathymetry_y_min_km, bathymetry_y_max_km = _prt_block_min_max(
        lines, r"Number\s+of\s+bathymetry\s+points\s+in\s+y\b"
    )
    (
        mode_wavenumber_min_1pm,
        mode_wavenumber_max_1pm,
        mode_attenuation_min_1pm,
        mode_attenuation_max_1pm,
        mode_phase_speed_min_mps,
        mode_phase_speed_max_mps,
        mode_group_speed_min_mps,
        mode_group_speed_max_mps,
    ) = _kraken_mode_table_ranges(lines)
    mode_count = _first_prt_match(text, r"Number\s+of\s+modes\s*=\s*([+-]?\d+)", int) or _count_kraken_modes(lines)
    completed_successfully = bool(re.search(r"completed\s+successfully", text, re.IGNORECASE))
    program_upper = (program or "").upper()
    kraken_completed = (
        (program_upper.startswith("KRAKEN") or program_upper.startswith("KRAKENC"))
        and mode_count is not None
        and bool(re.search(r"End\s+of\s+environmental\s+file", text, re.IGNORECASE))
    )
    completed_successfully = completed_successfully or kraken_completed

    return ATPRTLog(
        path=path,
        text=text,
        lines=lines,
        warnings=warnings,
        errors=errors,
        warning_count=len(warnings),
        error_count=len(errors),
        fatal_error_count=fatal_error_count,
        title=title,
        program=program,
        case_title=case_title,
        frequency_hz=_first_prt_match_any(
            text,
            [
                r"frequency\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
                r"Nominal\s+Frequency\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
            ],
        ),
        nmedia=_first_prt_match(text, r"NMedia\s*=\s*([+-]?\d+)", int),
        bottom_depth_m=_first_prt_match(text, r"Depth\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s*m"),
        rms_roughness_m=_first_prt_match(text, r"RMS\s+roughness\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"),
        source_depth_count=_first_prt_match(text, r"Number\s+of\s+Source\s+z-coordinates,\s*Sz\s*=\s*([+-]?\d+)", int),
        source_depth_min_m=source_depth_min_m,
        source_depth_max_m=source_depth_max_m,
        receiver_depth_count=_first_prt_match(text, r"Number\s+of\s+Receiver\s+z-coordinates,\s*Rz\s*=\s*([+-]?\d+)", int),
        receiver_range_count=_first_prt_match(text, r"Number\s+of\s+Receiver\s+r-coordinates,\s*Rr\s*=\s*([+-]?\d+)", int),
        receiver_depth_min_m=receiver_depth_min_m,
        receiver_depth_max_m=receiver_depth_max_m,
        receiver_range_min_km=receiver_range_min_km,
        receiver_range_max_km=receiver_range_max_km,
        beam_count=_first_prt_match(text, r"Number\s+of\s+beams\s+in\s+elevation\s*=\s*([+-]?\d+)", int),
        beam_bearing_count=_first_prt_match(text, r"Number\s+of\s+beams\s+in\s+bearing\s*=\s*([+-]?\d+)", int),
        beam_angle_min_deg=beam_angle_min_deg,
        beam_angle_max_deg=beam_angle_max_deg,
        beam_bearing_angle_min_deg=beam_bearing_angle_min_deg,
        beam_bearing_angle_max_deg=beam_bearing_angle_max_deg,
        step_length_m=_first_prt_match(text, r"Step\s+length,\s*deltas\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"),
        box_x_m=_first_prt_match(text, r"Box%x\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"),
        box_y_m=_first_prt_match(text, r"Box%y\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"),
        box_depth_m=_first_prt_match(text, r"Box%z\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"),
        box_range_km=_first_prt_match(text, r"Box%r\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)"),
        max_arrivals=max_arrivals,
        cpu_time_s=_first_prt_match_any(
            text,
            [
                r"CPU\s+Time\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
                r"^\s*\d+\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)s\s*$",
            ],
        ),
        run_type_description=run_type_description,
        beam_description=beam_description,
        source_description=source_description,
        receiver_grid_description=receiver_grid_description,
        source_beam_description=source_beam_description,
        mode_addition_description=mode_addition_description,
        calculation_description=calculation_description,
        field_option=field_option,
        field_alpha_min_deg=field_alpha_min_deg,
        field_alpha_max_deg=field_alpha_max_deg,
        field_alpha_count=field_alpha_count,
        field_step_m=field_step_m,
        field_step_count=field_step_count,
        source_x_count=_first_prt_match(text, r"Number\s+of\s+Source\s+x-coordinates,\s*Sx\s*=\s*([+-]?\d+)", int),
        source_y_count=_first_prt_match(text, r"Number\s+of\s+Source\s+y-coordinates,\s*Sy\s*=\s*([+-]?\d+)", int),
        source_x_min_km=source_x_min_km,
        source_x_max_km=source_x_max_km,
        source_y_min_km=source_y_min_km,
        source_y_max_km=source_y_max_km,
        source_coordinate_x_m=source_coordinate_x_m,
        source_coordinate_y_m=source_coordinate_y_m,
        source_coordinate_z_m=source_coordinate_z_m,
        receiver_theta_count=_first_prt_match_any(
            text,
            [
                r"Number\s+of\s+Receiver\s+theta-coordinates,\s*theta\s*=\s*([+-]?\d+)",
                r"Number\s+of\s+Receiver\s+\S+-coordinates,\s*theta\s*=\s*([+-]?\d+)",
            ],
            int,
        ),
        receiver_theta_min_deg=receiver_theta_min_deg,
        receiver_theta_max_deg=receiver_theta_max_deg,
        node_count=_first_prt_match(text, r"NNodes\s*=\s*([+-]?\d+)", int),
        element_count=_first_prt_match(text, r"NElts\s*=\s*([+-]?\d+)", int),
        profile_range_count=_first_prt_match(text, r"Number\s+of\s+Profile\s+ranges,\s*RProf\s*=\s*([+-]?\d+)", int),
        profile_range_min_km=profile_range_min_km,
        profile_range_max_km=profile_range_max_km,
        receiver_range_offset_count=_first_prt_match(
            text,
            r"Number\s+of\s+receiver\s+range\s+offsets\s*\(array\s+tilt\),\s*Rro\s*=\s*([+-]?\d+)",
            int,
        ),
        receiver_range_offset_min_m=receiver_range_offset_min_m,
        receiver_range_offset_max_m=receiver_range_offset_max_m,
        completed_successfully=completed_successfully,
        completed_without_errors=completed_successfully and not errors,
        bathymetry_x_count=_first_prt_match(text, r"Number\s+of\s+bathymetry\s+points\s+in\s+x\s+([+-]?\d+)", int),
        bathymetry_y_count=_first_prt_match(text, r"Number\s+of\s+bathymetry\s+points\s+in\s+y\s+([+-]?\d+)", int),
        bathymetry_x_min_km=bathymetry_x_min_km,
        bathymetry_x_max_km=bathymetry_x_max_km,
        bathymetry_y_min_km=bathymetry_y_min_km,
        bathymetry_y_max_km=bathymetry_y_max_km,
        beam_shift_description=beam_shift_description,
        c_low_mps=c_low_mps,
        c_high_mps=c_high_mps,
        rmax_km=_first_prt_match(text, r"RMax\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s*km"),
        attenuation_units=attenuation_units,
        mesh_point_count=_first_prt_match(text, r"#\s*mesh\s+points\s*=\s*([+-]?\d+)\s+automatically calculated", int),
        mesh_multiplier_cpu_s=_kraken_mesh_multiplier_cpu_table(lines),
        max_root_finder_restarts=_first_prt_match(text, r"MaxTries\s*=\s*([+-]?\d+)", int),
        beam_halfwidth1=_first_prt_match(
            text,
            r"HalfWidth1\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
        ),
        beam_halfwidth2=_first_prt_match(
            text,
            r"HalfWidth2\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
        ),
        eps_multiplier=_first_prt_match(
            text,
            r"EpsMult\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)",
        ),
        mode_count=mode_count,
        mode_wavenumber_min_1pm=mode_wavenumber_min_1pm,
        mode_wavenumber_max_1pm=mode_wavenumber_max_1pm,
        mode_attenuation_min_1pm=mode_attenuation_min_1pm,
        mode_attenuation_max_1pm=mode_attenuation_max_1pm,
        mode_phase_speed_min_mps=mode_phase_speed_min_mps,
        mode_phase_speed_max_mps=mode_phase_speed_max_mps,
        mode_group_speed_min_mps=mode_group_speed_min_mps,
        mode_group_speed_max_mps=mode_group_speed_max_mps,
        parameters=_parse_prt_key_values(lines),
    )

def read_at_boundary_profile(filename, *, value_name="depth", range_unit="km"):
    """Read a simple 2D AT boundary profile such as ``.ati`` or ``.bty``.

    The supported ASCII layout is the common AT form:
    interpolation flag, number of points, then ``range value`` pairs. Ranges
    are converted to meters when ``range_unit='km'``.
    """
    tokens = [_clean_at_token(tok) for tok in _numeric_tokens(filename)]
    if not tokens:
        raise ValueError(f"Empty AT boundary profile: {filename}")

    idx = 0
    interpolation = "L"
    try:
        n_points = int(float(tokens[idx]))
    except ValueError:
        interpolation = tokens[idx]
        idx += 1
        if idx >= len(tokens):
            raise ValueError(f"Missing point count in AT boundary profile: {filename}") from None
        n_points = int(float(tokens[idx]))
    else:
        if idx + 1 < len(tokens):
            try:
                float(tokens[idx + 1])
            except ValueError:
                interpolation = tokens[idx + 1]
                idx += 1
    idx += 1

    if n_points < 0:
        raise ValueError(f"AT boundary profile point count must be non-negative: {filename}")
    interpolation = interpolation.upper()
    point_format = "L" if len(interpolation) > 1 and interpolation[1] == "L" else "S"
    interpolation = interpolation[0]

    values_per_point = 7 if point_format == "L" else 2
    expected_values = values_per_point * n_points
    raw = tokens[idx:idx + expected_values]
    if len(raw) != expected_values:
        raise ValueError(
            f"Expected {n_points} {value_name} rows with {values_per_point} values in {filename}, "
            f"found {len(raw) // values_per_point}"
        )

    values = np.array([float(v) for v in raw], dtype=np.float64).reshape(n_points, values_per_point)
    ranges = values[:, 0]
    if range_unit == "km":
        ranges = ranges * 1000.0
    elif range_unit != "m":
        raise ValueError("range_unit must be either 'km' or 'm'")

    return ATBoundaryProfile(
        interpolation=interpolation,
        ranges_m=ranges,
        values_m=values[:, 1],
        geoacoustics=values[:, 2:] if point_format == "L" else None,
    )

def write_at_boundary_profile(filename, profile, *, range_unit="km"):
    """Write a simple 2D AT boundary profile."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    ranges = np.asarray(profile.ranges_m, dtype=np.float64)
    values = np.asarray(profile.values_m, dtype=np.float64)
    if ranges.shape != values.shape:
        raise ValueError("profile ranges_m and values_m must have the same shape")
    geoacoustics = None if profile.geoacoustics is None else np.asarray(profile.geoacoustics, dtype=np.float64)
    if geoacoustics is not None and geoacoustics.shape != (ranges.size, 5):
        raise ValueError("profile geoacoustics must have shape (n_points, 5)")
    if range_unit == "km":
        ranges_out = ranges / 1000.0
    elif range_unit == "m":
        ranges_out = ranges
    else:
        raise ValueError("range_unit must be either 'km' or 'm'")

    with path.open("w") as f:
        interpolation = profile.interpolation.upper()[0]
        suffix = "L" if geoacoustics is not None else ""
        f.write(f"'{interpolation}{suffix}'\n")
        f.write(f"{ranges.size}\n")
        for ii, (range_value, boundary_value) in enumerate(zip(ranges_out, values)):
            if geoacoustics is None:
                f.write(f"{range_value:.9g} {boundary_value:.9g}\n")
            else:
                media = " ".join(f"{value:.9g}" for value in geoacoustics[ii])
                f.write(f"{range_value:.9g} {boundary_value:.9g} {media}\n")


def write_at_ssp2d(filename, ranges_m, c_matrix):
    """Write a Bellhop ``Q`` range-dependent SSP matrix file."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    ranges = np.asarray(ranges_m, dtype=np.float64)
    speeds = np.asarray(c_matrix, dtype=np.float64)
    if ranges.ndim != 1 or ranges.size < 2:
        raise ValueError("Bellhop quad-2d SSP requires at least two range profiles")
    if speeds.ndim != 2:
        raise ValueError("Bellhop quad-2d c_matrix must be depth-major 2D data")
    if speeds.shape[1] != ranges.size:
        raise ValueError("Bellhop quad-2d c_matrix column count must match ranges")

    with path.open("w") as f:
        f.write(f"{ranges.size}\n")
        f.write(" ".join(f"{value:.9g}" for value in ranges / 1000.0) + " /\n")
        for row in speeds:
            f.write(" ".join(f"{value:.9g}" for value in row) + " /\n")


def write_at_bathymetry_3d(filename, bathymetry_3d, *, coordinate_unit="km"):
    """Write an AT Bellhop3D gridded bathymetry ``.bty`` file."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_source = bathymetry_3d.x_m if hasattr(bathymetry_3d, "x_m") else bathymetry_3d.x
    y_source = bathymetry_3d.y_m if hasattr(bathymetry_3d, "y_m") else bathymetry_3d.y
    z_source = bathymetry_3d.z_m if hasattr(bathymetry_3d, "z_m") else bathymetry_3d.z
    x = np.asarray(x_source, dtype=np.float64)
    y = np.asarray(y_source, dtype=np.float64)
    z = np.asarray(z_source, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
        raise ValueError("Bellhop3D bathymetry_3d requires at least two x and y coordinates")
    if z.shape != (y.size, x.size):
        raise ValueError("Bellhop3D bathymetry_3d z matrix must have shape (len(y), len(x))")
    if coordinate_unit == "km":
        x_out = x / 1000.0
        y_out = y / 1000.0
    elif coordinate_unit == "m":
        x_out = x
        y_out = y
    else:
        raise ValueError("coordinate_unit must be either 'km' or 'm'")
    interpolation = str(getattr(bathymetry_3d, "interpolation", "R")).upper()

    with path.open("w") as f:
        f.write(f"'{interpolation}'\n")
        f.write(f"{x.size}\n")
        f.write(" ".join(f"{value:.9g}" for value in x_out) + " /\n")
        f.write(f"{y.size}\n")
        f.write(" ".join(f"{value:.9g}" for value in y_out) + " /\n")
        for row in z:
            f.write(" ".join(f"{value:.9g}" for value in row) + "\n")


def read_at_bathymetry_3d(filename, *, coordinate_unit="km"):
    """Read an AT Bellhop3D gridded bathymetry ``.bty`` file."""

    lines = _at_data_lines(filename)
    if len(lines) < 5:
        raise ValueError(f"AT 3D bathymetry file is too short: {filename}")

    interpolation = _at_quoted_or_token(lines[0]).upper()
    nx_values = _at_numeric_values(lines[1])
    if nx_values.size != 1:
        raise ValueError(f"Expected one x-coordinate count in AT 3D bathymetry: {filename}")
    nx = int(nx_values[0])
    if nx < 2:
        raise ValueError(f"AT 3D bathymetry requires at least two x coordinates: {filename}")
    x = _at_numeric_values(lines[2])
    if x.size == 2 and nx > 2:
        x = np.linspace(float(x[0]), float(x[1]), nx, dtype=np.float64)
    if x.size != nx:
        raise ValueError(f"Expected {nx} x coordinates in AT 3D bathymetry: {filename}")

    ny_values = _at_numeric_values(lines[3])
    if ny_values.size != 1:
        raise ValueError(f"Expected one y-coordinate count in AT 3D bathymetry: {filename}")
    ny = int(ny_values[0])
    if ny < 2:
        raise ValueError(f"AT 3D bathymetry requires at least two y coordinates: {filename}")
    y = _at_numeric_values(lines[4])
    if y.size == 2 and ny > 2:
        y = np.linspace(float(y[0]), float(y[1]), ny, dtype=np.float64)
    if y.size != ny:
        raise ValueError(f"Expected {ny} y coordinates in AT 3D bathymetry: {filename}")

    z_values: list[float] = []
    for line in lines[5:]:
        z_values.extend(float(value) for value in _at_numeric_values(line))
    if len(z_values) != nx * ny:
        raise ValueError(
            f"Expected {nx * ny} z samples in AT 3D bathymetry: {filename}, found {len(z_values)}"
        )
    z = np.asarray(z_values, dtype=np.float64).reshape(ny, nx)

    if coordinate_unit == "km":
        x = x * 1000.0
        y = y * 1000.0
    elif coordinate_unit != "m":
        raise ValueError("coordinate_unit must be either 'km' or 'm'")

    return ATBathymetry3D(
        interpolation=interpolation,
        x_m=np.asarray(x, dtype=np.float64),
        y_m=np.asarray(y, dtype=np.float64),
        z_m=z,
    )


def read_at_bathymetry_3d_provinces(filename, *, coordinate_unit="km"):
    """Read Bellhop3D long-format ``.bty`` province/material grids."""

    lines = _at_data_lines(filename)
    if len(lines) < 8:
        raise ValueError(f"AT 3D province bathymetry file is too short: {filename}")

    interpolation = _at_quoted_or_token(lines[0]).upper()
    if len(interpolation) < 2 or interpolation[1] != "L":
        raise ValueError(f"AT 3D province bathymetry requires long-format interpolation: {filename}")
    nx_values = _at_numeric_values(lines[1])
    if nx_values.size != 1:
        raise ValueError(f"Expected one x-coordinate count in AT 3D province bathymetry: {filename}")
    nx = int(nx_values[0])
    if nx < 2:
        raise ValueError(f"AT 3D province bathymetry requires at least two x coordinates: {filename}")
    x = _at_numeric_values(lines[2])
    if x.size == 2 and nx > 2:
        x = np.linspace(float(x[0]), float(x[1]), nx, dtype=np.float64)
    if x.size != nx:
        raise ValueError(f"Expected {nx} x coordinates in AT 3D province bathymetry: {filename}")

    ny_values = _at_numeric_values(lines[3])
    if ny_values.size != 1:
        raise ValueError(f"Expected one y-coordinate count in AT 3D province bathymetry: {filename}")
    ny = int(ny_values[0])
    if ny < 2:
        raise ValueError(f"AT 3D province bathymetry requires at least two y coordinates: {filename}")
    y = _at_numeric_values(lines[4])
    if y.size == 2 and ny > 2:
        y = np.linspace(float(y[0]), float(y[1]), ny, dtype=np.float64)
    if y.size != ny:
        raise ValueError(f"Expected {ny} y coordinates in AT 3D province bathymetry: {filename}")

    cursor = 5
    depth_rows = []
    for row_index in range(ny):
        if cursor >= len(lines):
            raise ValueError(f"Missing depth row {row_index + 1} in AT 3D province bathymetry: {filename}")
        row = _at_numeric_values(lines[cursor])
        cursor += 1
        if row.size != nx:
            raise ValueError(f"Expected {nx} depth values on row {row_index + 1} in {filename}")
        depth_rows.append(row)
    z = np.vstack(depth_rows).astype(np.float64)

    province_rows = []
    for row_index in range(ny):
        if cursor >= len(lines):
            raise ValueError(f"Missing province row {row_index + 1} in AT 3D province bathymetry: {filename}")
        row = _at_numeric_values(lines[cursor])
        cursor += 1
        if row.size != nx:
            raise ValueError(f"Expected {nx} province values on row {row_index + 1} in {filename}")
        province_rows.append(row.astype(np.int64))
    province_indices = np.vstack(province_rows).astype(np.int64)

    if cursor >= len(lines):
        raise ValueError(f"Missing province count in AT 3D province bathymetry: {filename}")
    province_count_values = _at_numeric_values(lines[cursor])
    cursor += 1
    if province_count_values.size != 1:
        raise ValueError(f"Expected one province count in AT 3D province bathymetry: {filename}")
    province_count = int(province_count_values[0])
    if province_count < 1:
        raise ValueError(f"AT 3D province bathymetry province count must be positive: {filename}")

    property_rows = []
    for province_index in range(province_count):
        if cursor >= len(lines):
            raise ValueError(f"Missing province property row {province_index + 1} in {filename}")
        row = _at_numeric_values(lines[cursor])
        cursor += 1
        if row.size == 0:
            raise ValueError(f"Province property row {province_index + 1} is empty in {filename}")
        property_rows.append(row)
    property_widths = {row.size for row in property_rows}
    if len(property_widths) != 1:
        raise ValueError(f"AT 3D province property rows have inconsistent widths: {filename}")
    province_properties = np.vstack(property_rows).astype(np.float64)
    property_kind = "grain_size" if province_properties.shape[1] == 1 else (
        "geoacoustic" if province_properties.shape[1] == 5 else "unknown"
    )
    if np.any(province_indices < 1) or np.any(province_indices > province_count):
        raise ValueError(f"AT 3D province matrix contains undefined province indices: {filename}")

    if coordinate_unit == "km":
        x = x * 1000.0
        y = y * 1000.0
    elif coordinate_unit != "m":
        raise ValueError("coordinate_unit must be either 'km' or 'm'")

    return ATBathymetry3DProvinces(
        interpolation=interpolation,
        x_m=np.asarray(x, dtype=np.float64),
        y_m=np.asarray(y, dtype=np.float64),
        z_m=z,
        province_indices=province_indices,
        province_properties=province_properties,
        province_property_kind=property_kind,
    )


def write_at_bathymetry_3d_provinces(filename, bathymetry_3d, *, coordinate_unit="km"):
    """Write Bellhop3D long-format province/material ``.bty`` data."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(bathymetry_3d.x_m, dtype=np.float64)
    y = np.asarray(bathymetry_3d.y_m, dtype=np.float64)
    z = np.asarray(bathymetry_3d.z_m, dtype=np.float64)
    province_indices = np.asarray(bathymetry_3d.province_indices, dtype=np.int64)
    province_properties = np.asarray(bathymetry_3d.province_properties, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
        raise ValueError("Bellhop3D province bathymetry requires at least two x and y coordinates")
    if z.shape != (y.size, x.size):
        raise ValueError("Bellhop3D province bathymetry z matrix must have shape (len(y), len(x))")
    if province_indices.shape != z.shape:
        raise ValueError("Bellhop3D province index matrix must have the same shape as z_m")
    if province_properties.ndim != 2 or province_properties.shape[0] < 1:
        raise ValueError("Bellhop3D province properties must be a non-empty 2D array")
    if np.any(province_indices < 1) or np.any(province_indices > province_properties.shape[0]):
        raise ValueError("Bellhop3D province indices must refer to defined province properties")
    if coordinate_unit == "km":
        x_out = x / 1000.0
        y_out = y / 1000.0
    elif coordinate_unit == "m":
        x_out = x
        y_out = y
    else:
        raise ValueError("coordinate_unit must be either 'km' or 'm'")

    interpolation = str(getattr(bathymetry_3d, "interpolation", "RL")).upper()
    if len(interpolation) < 2:
        interpolation = interpolation[0] + "L"
    if interpolation[1] != "L":
        raise ValueError("Bellhop3D province bathymetry interpolation must use long format")
    with path.open("w") as f:
        f.write(f"'{interpolation}' \n")
        f.write(f"{x.size}\n")
        f.write(" ".join(f"{value:.9g}" for value in x_out) + "\n")
        f.write(f"{y.size}\n")
        f.write(" ".join(f"{value:.9g}" for value in y_out) + "\n")
        for row in z:
            f.write(" ".join(f"{value:.9g}" for value in row) + "\n")
        for row in province_indices:
            f.write(" ".join(str(int(value)) for value in row) + "\n")
        f.write(f"{province_properties.shape[0]}\n")
        for row in province_properties:
            f.write(" ".join(f"{value:.9g}" for value in row) + "\n")


def read_at_ati(filename):
    """Read an AT surface altimetry ``.ati`` file."""
    return read_at_boundary_profile(filename, value_name="altimetry")

def read_at_bty(filename):
    """Read a simple 2D AT bathymetry ``.bty`` file."""
    return read_at_boundary_profile(filename, value_name="bathymetry")

def read_at_reflection_table(filename):
    """Read a simple AT reflection coefficient table from ``.irc``/``.trc``/``.brc``."""
    lines = _at_data_lines(filename)
    if len(lines) >= 3 and ("'" in lines[0] or '"' in lines[0]):
        count_values = _at_numeric_values(lines[1])
        if count_values.size == 1:
            n_rows = int(count_values[0])
            raw_rows = [_at_numeric_values(line) for line in lines[2:]]
            if len(raw_rows) == n_rows and raw_rows and all(row.size == 6 for row in raw_rows):
                raw = np.vstack(raw_rows)
                angles = raw[:, 0]
                if np.nanmax(np.abs(angles)) <= np.pi + 1.0e-6:
                    angles = np.rad2deg(angles)
                coefficients = raw[:, 2] + 1j * raw[:, 3]
                return ATReflectionTable(
                    angles_deg=angles,
                    amplitudes=np.abs(coefficients),
                    phases_deg=np.rad2deg(np.angle(coefficients)),
                )

    tokens = [_clean_at_token(tok) for tok in _numeric_tokens(filename)]
    if not tokens:
        raise ValueError(f"Empty AT reflection coefficient table: {filename}")

    start_idx = 0
    n_rows = None
    candidate = int(float(tokens[0]))
    if len(tokens) == 1 + 3 * candidate:
        n_rows = candidate
        start_idx = 1
    elif len(tokens) % 3 == 0:
        n_rows = len(tokens) // 3
    else:
        raise ValueError(f"AT reflection coefficient table must contain angle/amplitude/phase triples: {filename}")

    raw = np.array([float(v) for v in tokens[start_idx:start_idx + 3 * n_rows]], dtype=np.float64).reshape(n_rows, 3)
    return ATReflectionTable(
        angles_deg=raw[:, 0],
        amplitudes=raw[:, 1],
        phases_deg=raw[:, 2],
    )

def write_at_reflection_table(filename, table):
    """Write a simple AT reflection coefficient table."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    angles = np.asarray(table.angles_deg, dtype=np.float64)
    amplitudes = np.asarray(table.amplitudes, dtype=np.float64)
    phases = np.asarray(table.phases_deg, dtype=np.float64)
    if not (angles.shape == amplitudes.shape == phases.shape):
        raise ValueError("reflection table arrays must have the same shape")

    with path.open("w") as f:
        f.write(f"{angles.size}\n")
        for angle, amplitude, phase in zip(angles, amplitudes, phases):
            f.write(f"{angle:.9g} {amplitude:.9g} {phase:.9g}\n")

def read_at_irc(filename):
    """Read an AT ice/surface reflection coefficient ``.irc`` table."""
    return read_at_reflection_table(filename)

def read_at_trc(filename):
    """Read an AT Kraken top reflection coefficient ``.trc`` table."""
    return read_at_reflection_table(filename)

def read_at_brc(filename):
    """Read an AT bottom reflection coefficient ``.brc`` table."""
    return read_at_reflection_table(filename)


def _pack_at_mod_char8(value):
    return str(value).encode("ascii", errors="ignore")[:8].ljust(8, b" ")


def _pack_at_mod_complex64(value):
    value = complex(value)
    return struct.pack("<2f", float(value.real), float(value.imag))


def _unpack_at_mod_complex64(payload):
    real, imag = struct.unpack("<2f", payload)
    return complex(real, imag)


def write_at_mod(filename, modes, z_bins, title="PyAcoustics native Kraken", wavenumbers=None, metadata=None):
    """Write a minimal AT-compatible binary Kraken ``.mod`` file."""
    import struct
    import numpy as np
    from pathlib import Path

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    modes = np.asarray(modes, dtype=np.complex64)
    z_bins = np.asarray(z_bins, dtype=np.float32)
    if modes.ndim != 2:
        raise ValueError("modes must be a 2D array")
    if modes.shape[1] != z_bins.size:
        raise ValueError("modes depth dimension must match z_bins")
    if wavenumbers is not None:
        wavenumbers = np.asarray(wavenumbers, dtype=np.complex64)
        if wavenumbers.shape != (modes.shape[0],):
            raise ValueError("wavenumbers must have one value per mode")
    metadata = metadata or {}
    n_modes, n_z = modes.shape

    media = metadata.get("media")
    if media is None:
        media = [{"n": max(n_z - 1, 0), "material": "ACOUSTIC", "depth": float(z_bins[0]) if n_z else 0.0, "rho": 1.0}]
    nmedia = len(media)
    freq_vec = np.asarray(metadata.get("freq_vec", [metadata.get("frequency", 0.0)]), dtype=np.float32)
    if freq_vec.size == 0:
        freq_vec = np.array([0.0], dtype=np.float32)
    top = metadata.get("top", {})
    bottom = metadata.get("bottom", {})

    lrecl = max(32, 2 * n_z, 2 * n_modes, 3 * nmedia, 2 * freq_vec.size)
    rec_size = lrecl * 4

    def pack_record(payload):
        if len(payload) > rec_size:
            raise ValueError("record payload exceeds Kraken .mod record size")
        return payload + b"\x00" * (rec_size - len(payload))

    title_bytes = title.encode("ascii", errors="ignore")[:80].ljust(80, b" ")
    header = struct.pack("<i", lrecl) + title_bytes + struct.pack("<4i", freq_vec.size, nmedia, n_z, n_z)

    media_counts = b"".join(
        struct.pack("<i", int(medium.get("n", max(n_z - 1, 0)))) + _pack_at_mod_char8(medium.get("material", "ACOUSTIC"))
        for medium in media
    )
    media_depths = b"".join(
        struct.pack("<2f", float(medium.get("depth", 0.0)), float(medium.get("rho", 1.0)))
        for medium in media
    )
    top_bc = str(top.get("bc", "V"))[:1].encode("ascii", errors="ignore").ljust(1, b" ")
    bot_bc = str(bottom.get("bc", "A"))[:1].encode("ascii", errors="ignore").ljust(1, b" ")
    halfspace = (
        top_bc
        + _pack_at_mod_complex64(top.get("c_p", 0.0))
        + _pack_at_mod_complex64(top.get("c_s", 0.0))
        + struct.pack("<2f", float(top.get("rho", 0.0)), float(top.get("depth", float(z_bins[0]) if n_z else 0.0)))
        + bot_bc
        + _pack_at_mod_complex64(bottom.get("c_p", 0.0))
        + _pack_at_mod_complex64(bottom.get("c_s", 0.0))
        + struct.pack("<2f", float(bottom.get("rho", 0.0)), float(bottom.get("depth", float(z_bins[-1]) if n_z else 0.0)))
    )

    with filename.open("wb") as f:
        f.write(pack_record(header))
        f.write(pack_record(media_counts))
        f.write(pack_record(media_depths))
        f.write(pack_record(freq_vec.astype("<f4").tobytes()))
        f.write(pack_record(z_bins.astype("<f4").tobytes()))
        f.write(pack_record(struct.pack("<i", n_modes)))
        f.write(pack_record(halfspace))
        for mode in modes:
            f.write(pack_record(mode.astype("<c8").tobytes()))
        if wavenumbers is not None:
            first = 0
            values_per_record = lrecl // 2
            while first < n_modes:
                last = min(n_modes, first + values_per_record)
                f.write(pack_record(wavenumbers[first:last].astype("<c8").tobytes()))
                first = last


def read_at_mod_wavenumbers(filename):
    """Read the optional modal wavenumber record from a Kraken ``.mod`` file."""
    import os
    import struct
    import numpy as np

    if not os.path.exists(filename) or os.path.getsize(filename) < 100:
        return np.array([])

    with open(filename, "rb") as f:
        lrecl_raw = f.read(4)
        if len(lrecl_raw) < 4:
            return np.array([])
        lrecl = struct.unpack("<i", lrecl_raw)[0]
        rec_size = lrecl * 4

        f.seek(5 * rec_size)
        m_raw = f.read(4)
        if len(m_raw) < 4:
            return np.array([])
        n_modes = struct.unpack("<i", m_raw)[0]
        if n_modes <= 0 or n_modes > 100000:
            return np.array([])

        values_per_record = max(lrecl // 2, 1)
        chunks = []
        first = 0
        record = 7 + n_modes
        while first < n_modes:
            count = min(n_modes - first, values_per_record)
            f.seek(record * rec_size)
            raw = f.read(count * 8)
            if len(raw) < count * 8:
                return np.array([])
            floats = np.frombuffer(raw, dtype="<f4")
            chunks.append(floats[0::2] + 1j * floats[1::2])
            first += count
            record += 1
        return np.concatenate(chunks) if chunks else np.array([])


def read_at_mod_metadata(filename):
    """Read native/AT-style Kraken ``.mod`` header metadata."""
    import os
    import struct
    import numpy as np

    if not os.path.exists(filename) or os.path.getsize(filename) < 100:
        return {}

    with open(filename, "rb") as f:
        lrecl_raw = f.read(4)
        if len(lrecl_raw) < 4:
            return {}
        lrecl = struct.unpack("<i", lrecl_raw)[0]
        rec_size = lrecl * 4

        f.seek(0)
        header = f.read(104)
        if len(header) < 104:
            return {}
        title = header[4:84].decode("ascii", errors="ignore").strip()
        nfreq, nmedia, ntot, nmat = struct.unpack("<4i", header[84:100])

        f.seek(rec_size)
        media_raw = f.read(nmedia * 12)
        media = []
        for idx in range(nmedia):
            offset = idx * 12
            if offset + 12 > len(media_raw):
                break
            n_points = struct.unpack("<i", media_raw[offset : offset + 4])[0]
            material = media_raw[offset + 4 : offset + 12].decode("ascii", errors="ignore").strip()
            media.append({"n": n_points, "material": material})

        f.seek(2 * rec_size)
        depth_raw = f.read(nmedia * 8)
        for idx, medium in enumerate(media):
            offset = idx * 8
            if offset + 8 > len(depth_raw):
                break
            depth, rho = struct.unpack("<2f", depth_raw[offset : offset + 8])
            medium["depth"] = depth
            medium["rho"] = rho

        f.seek(3 * rec_size)
        freq_vec = np.frombuffer(f.read(nfreq * 4), dtype="<f4").astype(np.float64)

        f.seek(4 * rec_size)
        z_bins = np.frombuffer(f.read(ntot * 4), dtype="<f4").astype(np.float64)

        f.seek(5 * rec_size)
        mode_count = None
        mode_count_raw = f.read(4)
        if len(mode_count_raw) == 4:
            mode_count = struct.unpack("<i", mode_count_raw)[0]

        top = {}
        bottom = {}
        f.seek(6 * rec_size)
        halfspace_raw = f.read(50)
        if len(halfspace_raw) >= 42:
            top = {
                "bc": halfspace_raw[0:1].decode("ascii", errors="ignore").strip(),
                "c_p": _unpack_at_mod_complex64(halfspace_raw[1:9]),
                "c_s": _unpack_at_mod_complex64(halfspace_raw[9:17]),
                "rho": struct.unpack("<f", halfspace_raw[17:21])[0],
                "depth": struct.unpack("<f", halfspace_raw[21:25])[0],
            }
            bottom = {
                "bc": halfspace_raw[25:26].decode("ascii", errors="ignore").strip(),
                "c_p": _unpack_at_mod_complex64(halfspace_raw[26:34]),
                "c_s": _unpack_at_mod_complex64(halfspace_raw[34:42]),
            }
            if len(halfspace_raw) >= 50:
                bottom["rho"] = struct.unpack("<f", halfspace_raw[42:46])[0]
                bottom["depth"] = struct.unpack("<f", halfspace_raw[46:50])[0]

        return {
            "lrecl": lrecl,
            "title": title,
            "nfreq": nfreq,
            "nmedia": nmedia,
            "ntot": ntot,
            "nmat": nmat,
            "mode_count": mode_count,
            "media": media,
            "freq_vec": freq_vec,
            "z_bins": z_bins,
            "top": top,
            "bottom": bottom,
        }

def read_at_mod(filename):
    """
    Reads a binary .mod file produced by Kraken.
    Follows the exact logic of Acoustics Toolbox read_modes_bin.m
    """
    import os
    import struct
    import numpy as np

    if not os.path.exists(filename) or os.path.getsize(filename) < 100:
        return np.array([]), np.array([])
        
    with open(filename, 'rb') as f:
        # First 4 bytes is LRecl in WORDS
        lrecl_raw = f.read(4)
        if len(lrecl_raw) < 4: return np.array([]), np.array([])
        LRecl = struct.unpack('<i', lrecl_raw)[0]
        rec_size = LRecl * 4
        
        # Record 0 (Header)
        f.seek(4) # Skip the 4-byte LRecl
        title = f.read(80).decode('ascii', errors='ignore').strip()
        nfreq, nmedia, ntot, nmat = struct.unpack('<4i', f.read(16))
        
        if ntot <= 0 or ntot > 100000 or nmat <= 0:
            return np.array([]), np.array([])
            
        # Record 4 (0-based 4, which is the 5th record) is z
        f.seek(4 * rec_size)
        z_bins = np.frombuffer(f.read(ntot * 4), dtype='<f4')
        
        # Record 5 is M (number of modes)
        f.seek(5 * rec_size)
        M = struct.unpack('<i', f.read(4))[0]
        
        if M <= 0 or M > 100000:
            return np.array([]), z_bins
            
        # Modes start at Record 7
        modes = []
        for i in range(1, M + 1):
            f.seek((6 + i) * rec_size)
            phi_raw = f.read(nmat * 8)
            if len(phi_raw) < nmat * 8:
                break
            phi_floats = np.frombuffer(phi_raw, dtype='<f4')
            phi_complex = phi_floats[0::2] + 1j * phi_floats[1::2]
            modes.append(phi_complex)
            
        return np.array(modes), z_bins
