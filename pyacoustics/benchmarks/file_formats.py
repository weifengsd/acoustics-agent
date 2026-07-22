"""Canonical benchmark rows for AT file-format round trips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pyacoustics.postprocessing.delay_and_sum import ArrivalData, ArrivalRecord
from pyacoustics.solvers.benchmark_thresholds import BenchmarkRow, BenchmarkThresholds, evaluate_benchmark_rows
from pyacoustics.solvers.external_io import (
    ATBoundaryProfile,
    ATFieldFLP,
    ATRayRecord,
    ATReflectionTable,
    ATSourceBeamPattern,
    read_at_arr,
    read_at_bathymetry_3d,
    read_at_bathymetry_3d_provinces,
    read_at_env_summary,
    read_at_field_flp,
    read_at_fortran_source_summary,
    read_at_ati,
    read_at_brc,
    read_at_bty,
    read_at_irc,
    read_at_mod,
    read_at_mod_metadata,
    read_at_mod_wavenumbers,
    read_at_matlab_script_summary,
    read_at_prt,
    read_at_ram_in_summary,
    read_at_ray_records,
    read_at_shd_full,
    read_at_ssp2d,
    read_at_ssp3d,
    read_at_source_beam_pattern,
    read_at_trc,
    write_at_arr,
    write_at_bathymetry_3d,
    write_at_bathymetry_3d_provinces,
    write_at_boundary_profile,
    write_at_field_flp,
    write_at_mod,
    write_at_ray_records,
    write_at_reflection_table,
    write_at_shd,
    write_at_source_beam_pattern,
)


@dataclass(frozen=True)
class ATFileFormatBenchmarkMatrix:
    """AT file-format round-trip benchmark rows."""

    rows: tuple[BenchmarkRow, ...]


_CANONICAL_FLP_SAMPLES = (
    Path("tests/Munk/munk_at.flp"),
    Path("tests/Munk/munkk_at.flp"),
    Path("tests/3DAtlantic/lant.flp"),
    Path("tests/3DAtlantic/lant_90degrees.flp"),
    Path("tests/3DAtlantic/lantgbt.flp"),
    Path("tests/3DAtlantic/lant_gbt.flp"),
    Path("tests/3DAtlantic/lantvol.flp"),
    Path("tests/3DAtlantic/pacific.flp"),
    Path("tests/Bellhop3DTests/KoreanSeas/KoreanSea_3D.flp"),
)

_OPTIONAL_REAL_FLP_SAMPLES = (
    Path("Matlab/Krakel/Ahmad test case/ahmad.flp"),
    Path("Matlab/Krakel/Ahmad test case/field.flp"),
    Path("Matlab/Krakel/Ahmad test case/fields.flp"),
    Path("Matlab/Sparc/fields.flp"),
    Path("tests/BeamPattern/MunkKleaky.flp"),
    Path("tests/BeamPattern/MunkS.flp"),
    Path("tests/BeamPattern/shadedS.flp"),
    Path("tests/Bellhop3DTests/KermitRoosevelt/slice2d_flastS.flp"),
    Path("tests/Bellhop3DTests/Munk/munk3d.flp"),
    Path("tests/Bellhop3DTests/Provinces/provinces_2DS.flp"),
    Path("tests/Bellhop3DTests/Provinces/provinces_2DshearS.flp"),
    Path("tests/Bellhop3DTests/TruncatedWedge/slice2d_flastS.flp"),
    Path("tests/Bellhop3DTests/free/freeSPoint.flp"),
    Path("tests/Bellhop3DTests/halfspace/lower_halfS.flp"),
    Path("tests/Bellhop3DTests/halfspace/upper_halfS.flp"),
    Path("tests/BroadBand/MunkK.flp"),
    Path("tests/BroadBand/MunkS.flp"),
    Path("tests/BroadBand/MunkS_500Hz.flp"),
    Path("tests/Dickins/DickinsFlatK.flp"),
    Path("tests/Dickins/DickinsFlatS.flp"),
    Path("tests/Dickins/March/DickinsK_rd.flp"),
    Path("tests/Dickins/Precalc/DickinsK.flp"),
    Path("tests/Dickins/field.flp"),
    Path("tests/Dickins/field_rd.flp"),
    Path("tests/Dickins/field_ri.flp"),
    Path("tests/Gulf/GulfS.flp"),
    Path("tests/Gulf/gulf_ad.flp"),
    Path("tests/Gulf/gulf_cm.flp"),
    Path("tests/Gulf/gulf_ri.flp"),
    Path("tests/Munk/MunkAnalytic.flp"),
    Path("tests/Munk/MunkK.flp"),
    Path("tests/Munk/MunkS.flp"),
    Path("tests/Munk/Munk_shearS.flp"),
    Path("tests/MunkLeaky/MunkKbb.flp"),
    Path("tests/MunkLeaky/MunkKleaky.flp"),
    Path("tests/MunkLeaky/MunkKwb.flp"),
    Path("tests/MunkLeaky/MunkS.flp"),
    Path("tests/MunkTS/MunkS.flp"),
    Path("tests/Noise/ATOC/aetC.flp"),
    Path("tests/Noise/ATOC/aetC_TL.flp"),
    Path("tests/Noise/ATOC/aetK.flp"),
    Path("tests/Noise/ATOC/aetK_TL.flp"),
    Path("tests/Noise/ATOC/aetS.flp"),
    Path("tests/Noise/ATOC/aetS_TL.flp"),
    Path("tests/Noise/ATOC/aet_VLA_C.flp"),
    Path("tests/Noise/ATOC/aet_VLA_K.flp"),
    Path("tests/Noise/ATOC/aet_VLA_S.flp"),
    Path("tests/Noise/Halfspace/halfspace.flp"),
    Path("tests/Noise/KupermanIngenito/DownwdRef.flp"),
    Path("tests/Noise/KupermanIngenito/DownwdRef_C.flp"),
    Path("tests/Noise/KupermanIngenito/DownwdRef_S.flp"),
    Path("tests/Noise/KupermanIngenito/Iso.flp"),
    Path("tests/Noise/KupermanIngenito/Iso_C.flp"),
    Path("tests/Noise/KupermanIngenito/Iso_S.flp"),
    Path("tests/Noise/KupermanIngenito/Sduct.flp"),
    Path("tests/Noise/KupermanIngenito/Sduct_C.flp"),
    Path("tests/Noise/KupermanIngenito/Sduct_S.flp"),
    Path("tests/Noise/KupermanIngenito/HS_S.flp"),
    Path("tests/Noise/Pekeris/pekeris.flp"),
    Path("tests/PointLine/LloydSLine.flp"),
    Path("tests/PointLine/LloydSPoint.flp"),
    Path("tests/SedAtten/calibK.flp"),
    Path("tests/SedAtten/calibS_0.6dB.flp"),
    Path("tests/SedAtten/calibS_PowLaw.flp"),
    Path("tests/SedAtten/calibS_noloss.flp"),
    Path("tests/TLslices/field.flp"),
    Path("tests/TLslices/fieldbat.flp"),
    Path("tests/TLslices/fields.flp"),
    Path("tests/TabRefCoef/neggradC_brc.flp"),
    Path("tests/TabRefCoef/neggradC_geo.flp"),
    Path("tests/TabRefCoef/neggradC_irc.flp"),
    Path("tests/TabRefCoef/neggradK_brc.flp"),
    Path("tests/TabRefCoef/neggradK_geo.flp"),
    Path("tests/TabRefCoef/neggradK_irc.flp"),
    Path("tests/TabRefCoef/neggradS_brc.flp"),
    Path("tests/TabRefCoef/neggradS_geo.flp"),
    Path("tests/TabRefCoef/neggradS_irc.flp"),
    Path("tests/VolAtt/freeS.flp"),
    Path("tests/VolAtt/free_FGS.flp"),
    Path("tests/VolAtt/free_ThorpS.flp"),
    Path("tests/arctic/arcticK.flp"),
    Path("tests/arctic/arcticS.flp"),
    Path("tests/calib/calibK.flp"),
    Path("tests/calib/calibKgrad.flp"),
    Path("tests/calib/calibS.flp"),
    Path("tests/calib/calibSgrad.flp"),
    Path("tests/free/freeSLine.flp"),
    Path("tests/free/freeSPoint.flp"),
    Path("tests/halfspace/lower_halfS.flp"),
    Path("tests/halfspace/upper_halfS.flp"),
    Path("tests/head/head.flp"),
    Path("tests/head/headS.flp"),
    Path("tests/sduct/sductK.flp"),
    Path("tests/sduct/sductS.flp"),
    Path("tests/sduct/sduct_bbS.flp"),
    Path("tests/step/stepK_rd.flp"),
    Path("tests/wedge/fieldNew.flp"),
    Path("tests/wedge/wedge.flp"),
    Path("tests/wedge/wedge_flatS.flp"),
)

_CANONICAL_SHD_SAMPLES = (
    Path("tests/Munk/munk_at.shd"),
    Path("tests/Munk/munkk_at.shd"),
)

_CANONICAL_MOD_SAMPLES = (
    Path("tests/Munk/munk_at.mod"),
)

_OPTIONAL_REAL_SHD_SAMPLES = (
    Path("tests/Munk/munk_bellhop.shd"),
    Path("tests/Bellhop3DTests/scratch/temp_plot.shd"),
    Path("tests/arctic/arctic_at.shd"),
    Path("tests/bottom_gradient/bottom_gradient_at.shd"),
    Path("tests/deep_water/deep_water_at.shd"),
    Path("tests/dickins/dickins_at.shd"),
    Path("tests/free_space/free_space_at.shd"),
    Path("tests/gulf/gulf_at.shd"),
    Path("tests/head_wave/head_wave_at.shd"),
    Path("tests/iso/iso_at.shd"),
    Path("tests/leaky_munk/leaky_munk_at.shd"),
    Path("tests/lloyd/lloyd_at.shd"),
    Path("tests/pekeris/pekeris_at.shd"),
    Path("tests/rap_water/rap_water_at.shd"),
    Path("tests/rigid_pekeris/rigid_pekeris_at.shd"),
    Path("tests/sbcx/sbcx_at.shd"),
    Path("tests/sduct/sduct_at.shd"),
    Path("tests/sed_atten/sed_atten_at.shd"),
    Path("tests/shaded_source/shaded_source_at.shd"),
    Path("tests/shallow_water/shallow_water_at.shd"),
    Path("tests/thorp/thorp_at.shd"),
    Path("tests/calib/calibK.shd"),
)

_OPTIONAL_REAL_MOD_SAMPLES = (
    Path("tests/calib/calibK.mod"),
    Path("tests/Munk/munkk_at.mod"),
    Path("tests/pekeris/pekeris_at.mod"),
)

_OPTIONAL_REAL_BOUNDARY_PROFILE_SAMPLES = (
    Path("tests/Ellipse/Ellipse.ati"),
    Path("tests/Ellipse/EllipseTLGB.ati"),
    Path("tests/Ellipse/EllipseTLGeom.ati"),
    Path("tests/Ellipse/Ellipse.bty"),
    Path("tests/Ellipse/EllipseTLGB.bty"),
    Path("tests/Ellipse/EllipseTLGeom.bty"),
    Path("tests/ParaBot/ParaBot.ati"),
    Path("tests/ParaBot/ParaBotLTLGeom.ati"),
    Path("tests/ParaBot/ParaBotTLGB.ati"),
    Path("tests/ParaBot/ParaBotTLGeom.ati"),
    Path("tests/ParaBot/ParaBot.bty"),
    Path("tests/ParaBot/ParaBot39.bty"),
    Path("tests/ParaBot/ParaBotLTLGeom.bty"),
    Path("tests/ParaBot/ParaBotTLGB.bty"),
    Path("tests/ParaBot/ParaBotTLGeom.bty"),
    Path("tests/Bellhop3DTests/LloydMirror/LloydWallB.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/slice2dB.bty"),
    Path("tests/Bellhop3DTests/Seamount/Seamount_slice.bty"),
    Path("tests/Bellhop3DTests/halfspace/halfspaceJCP/lower_half_wallB.bty"),
    Path("tests/Dickins/Dickins.bty"),
    Path("tests/Dickins/DickinsB.bty"),
    Path("tests/Dickins/DickinsB_oneBeam.bty"),
    Path("tests/Dickins/DickinsBray.bty"),
    Path("tests/Dickins/DickinsCervenyB.bty"),
    Path("tests/Dickins/March/DickinsK.bty"),
    Path("tests/Dickins/Precalc/DickinsK.bty"),
    Path("tests/Gulf/GulfB_rd_gb.bty"),
    Path("tests/Gulf/GulfB_rd_geo.bty"),
    Path("tests/Gulf/Gulf_ray_rd.bty"),
    Path("tests/Gulf/Gulf_ray_ri.bty"),
    Path("tests/Munk/MunkB_geo_rot.bty"),
    Path("tests/Munk/MunkB_ray_rot.bty"),
    Path("tests/PekerisRD/PekerisRDB.bty"),
    Path("tests/block/blockB_gb.bty"),
    Path("tests/block/blockB_geo.bty"),
    Path("tests/block/blockB_ray.bty"),
    Path("tests/step/step.bty"),
    Path("tests/step/stepK.bty"),
    Path("tests/wedge/wedgeB.bty"),
)

_OPTIONAL_REAL_REFLECTION_TABLE_SAMPLES = (
    Path("tests/SedAtten/calibBounce.brc"),
    Path("tests/SedAtten/calibBounce.irc"),
)

_OPTIONAL_REAL_SOURCE_BEAM_PATTERN_SAMPLES = (
    Path("tests/BeamPattern/MunkKleaky.sbp"),
    Path("tests/BeamPattern/MunkS.sbp"),
    Path("tests/BeamPattern/shaded.sbp"),
    Path("tests/BeamPattern/shadedS.sbp"),
)

_CANONICAL_BATHYMETRY_3D_SAMPLES = (
    Path("tests/Bellhop3DTests/scratch/temp_plot.bty"),
)

_OPTIONAL_REAL_BATHYMETRY_3D_SAMPLES = (
    Path("tests/Bellhop3DTests/DoubleSeamount/DoubleSeamount3D_ray.bty"),
    Path("tests/Bellhop3DTests/Hyperbot/HyperBot.bty"),
    Path("tests/Bellhop3DTests/KermitRoosevelt/KR.bty"),
    Path("tests/Bellhop3DTests/KermitRoosevelt/KR3dGaussian.bty"),
    Path("tests/Bellhop3DTests/KermitRoosevelt/LS.bty"),
    Path("tests/Bellhop3DTests/KermitRoosevelt/LSCart.bty"),
    Path("tests/Bellhop3DTests/KermitRoosevelt/slice3d.bty"),
    Path("tests/Bellhop3DTests/KoreanSeas/KoreanSea.bty"),
    Path("tests/Bellhop3DTests/LloydMirror/LloydWall_1B3.bty"),
    Path("tests/Bellhop3DTests/LloydMirror/LloydWall_2B3.bty"),
    Path("tests/Bellhop3DTests/LloydMirror/LloydWall_3B3.bty"),
    Path("tests/Bellhop3DTests/LloydMirror/LloydWall_4B3.bty"),
    Path("tests/Bellhop3DTests/Munk/munk3d.bty"),
    Path("tests/Bellhop3DTests/Munk/munk3dFlat.bty"),
    Path("tests/Bellhop3DTests/Munk/munk3dMountain.bty"),
    Path("tests/Bellhop3DTests/Munk/munk3d_ray.bty"),
    Path("tests/Bellhop3DTests/MunkRot/Munk3D.bty"),
    Path("tests/Bellhop3DTests/MunkRot/foo.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/pwedge3d.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/pwedge3dCart.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/pwedge3dRayCen.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/pwedge3d_rot.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/pwedge_elastic_3dCart.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/pwedge2d.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/pwedge2d_rot.bty"),
    Path("tests/Bellhop3DTests/PenetrableWedge/slice2d.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge2d.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge2dray.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge3dGaussian.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge3dGaussianRayCen.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge3dHat.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge3dHatRayCen.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge3d_ray.bty"),
    Path("tests/Bellhop3DTests/PerfectWedge/wedge2dside.bty"),
    Path("tests/Bellhop3DTests/Seamount/Seamount.bty"),
    Path("tests/Bellhop3DTests/Seamount/Seamount2D_side.bty"),
    Path("tests/Bellhop3DTests/Taiwan/Taiwan.bty"),
    Path("tests/Bellhop3DTests/TruncatedWedge/wedge3dGaussian.bty"),
    Path("tests/Bellhop3DTests/TruncatedWedge/wedge3dGaussian_side.bty"),
    Path("tests/Bellhop3DTests/TruncatedWedge/wedge3dHat_side.bty"),
    Path("tests/Bellhop3DTests/Weymouth/Weymouth.bty"),
    Path("tests/Bellhop3DTests/halfspace/halfspaceJCP/lower_half_wallB3.bty"),
)

_OPTIONAL_REAL_BATHYMETRY_3D_PROVINCE_SAMPLES = (
    Path("tests/Bellhop3DTests/Provinces/provinces.bty"),
    Path("tests/Bellhop3DTests/Provinces/provinces_grain.bty"),
    Path("tests/Bellhop3DTests/Provinces/provinces_side2D.bty"),
    Path("tests/Bellhop3DTests/Provinces/provinces_side3D.bty"),
)

_CANONICAL_SSP3D_SAMPLES = (
    Path("tests/Bellhop3DTests/Taiwan/Taiwan.ssp"),
    Path("tests/Bellhop3DTests/KoreanSeas/KoreanSea.ssp"),
)

_OPTIONAL_REAL_SSP3D_SAMPLES = (
    Path("tests/Bellhop3DTests/MunkRot/Munk3D.ssp"),
    Path("tests/Bellhop3DTests/MunkRot/foo.ssp"),
)

_OPTIONAL_REAL_SSP2D_SAMPLES = (
    Path("tests/Gulf/GulfB_rd_gb.ssp"),
    Path("tests/Gulf/GulfB_rd_geo.ssp"),
    Path("tests/Gulf/Gulf_ray_rd.ssp"),
    Path("tests/Munk/MunkB_geo_rot.ssp"),
    Path("tests/Munk/MunkB_ray_rot.ssp"),
    Path("tests/MunkRot/MunkRot.ssp"),
)

_OPTIONAL_REAL_RAY_SAMPLES = (
    Path("tests/Bellhop3DTests/PerfectWedge/wedge3d_ray.ray"),
)

_CANONICAL_PRT_SAMPLES: tuple[tuple[Path, dict[str, object]], ...] = (
    (
        Path("tests/Munk/munk_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "case_title": "BELLHOP- Munk Profile Test",
            "frequency_hz": 50.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
            "box_depth_m": 5500.0,
            "box_range_km": 101.0,
        },
    ),
    (
        Path("tests/Munk/field.prt"),
        {
            "program": "Running FIELD",
            "case_title": "Sums modes, producing pressure",
            "profile_range_count": 1,
            "receiver_range_count": 200,
            "receiver_depth_count": 100,
            "receiver_range_offset_count": 100,
            "completed_without_errors": True,
        },
    ),
    (
        Path("tests/Munk/munkk_at.prt"),
        {
            "program": "KRAKEN- MunkK Reference",
            "frequency_hz": 50.0,
            "nmedia": 1,
            "receiver_depth_count": 100,
            "mode_count": 102,
            "mesh_point_count": 3221,
            "c_low_mps": 1500.0,
            "c_high_mps": 1600.0,
            "rmax_km": 100.0,
            "completed_without_errors": True,
        },
    ),
    (
        Path("tests/Bellhop3DTests/scratch/temp_plot.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "case_title": "BELLHOP3D- Munk 3D Slanted Terrain Coherent TL",
            "frequency_hz": 50.0,
            "source_x_count": 1,
            "source_y_count": 1,
            "receiver_depth_count": 100,
            "receiver_range_count": 100,
            "receiver_theta_count": 1,
            "beam_count": 20,
            "beam_bearing_count": 24,
            "box_depth_m": 20500.0,
        },
    ),
    (
        Path("tests/pekeris/pekeris_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "case_title": "BELLHOP- Pekeris Calib",
            "frequency_hz": 250.0,
            "bottom_depth_m": 100.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
            "box_depth_m": 110.0,
            "box_range_km": 5.05,
        },
    ),
)

_OPTIONAL_REAL_PRT_SAMPLES: tuple[tuple[Path, dict[str, object]], ...] = (
    (
        Path("tests/Munk/munk_bellhop.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 50.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 1000,
        },
    ),
    (
        Path("tests/arctic/arctic_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 200.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/bottom_gradient/bottom_gradient_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 250.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/deep_water/deep_water_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 1000.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/dickins/dickins_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 230.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/free_space/free_space_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 5.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/gulf/gulf_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 50.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/head_wave/head_wave_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 100.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/iso/iso_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 100.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/leaky_munk/leaky_munk_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 100.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/lloyd/lloyd_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 50.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/pekeris/field.prt"),
        {
            "program": "Running FIELD",
            "case_title": "Sums modes, producing pressure",
            "profile_range_count": 1,
            "receiver_range_count": 200,
            "receiver_depth_count": 100,
            "receiver_range_offset_count": 100,
            "completed_without_errors": True,
        },
    ),
    (
        Path("tests/rap_water/rap_water_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 500.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/rigid_pekeris/rigid_pekeris_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 100.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/sbcx/sbcx_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 200.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/sduct/sduct_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 200.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/sed_atten/sed_atten_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 250.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/shaded_source/shaded_source_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 100.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/shallow_water/shallow_water_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 100.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/thorp/thorp_at.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "frequency_hz": 5000.0,
            "receiver_depth_count": 501,
            "receiver_range_count": 1001,
            "beam_count": 2000,
        },
    ),
    (
        Path("tests/calib/calibK.prt"),
        {
            "program": "KRAKEN- Calibration case",
            "frequency_hz": 250.0,
            "nmedia": 1,
            "receiver_depth_count": 101,
            "mode_count": 33,
            "c_low_mps": 1400.0,
            "c_high_mps": 20000.0,
            "rmax_km": 1000.0,
            "completed_without_errors": True,
        },
    ),
    (
        Path("tests/calib/field.prt"),
        {
            "program": "Running FIELD",
            "case_title": "Sums modes, producing pressure",
            "profile_range_count": 1,
            "receiver_range_count": 1001,
            "receiver_depth_count": 101,
            "receiver_range_offset_count": 101,
            "completed_without_errors": True,
        },
    ),
    (
        Path("tests/Bellhop3DTests/PerfectWedge/wedge3d_ray.prt"),
        {
            "program": "BELLHOP/BELLHOP3D",
            "case_title": "BELLHOP3D- Perfect wedge, 3D",
            "frequency_hz": 10.0,
            "nmedia": 1,
            "source_x_count": 1,
            "source_y_count": 1,
            "receiver_depth_count": 1,
            "receiver_range_count": 301,
            "receiver_theta_count": 21,
            "beam_count": 21,
            "beam_bearing_count": 7,
            "bathymetry_x_count": 21,
            "bathymetry_y_count": 21,
            "box_depth_m": 20500.0,
        },
    ),
)

_OPTIONAL_REAL_RAM_IN_SAMPLES = (
    Path("tests/Dickins/ram.in"),
    Path("tests/wedge/ram.in"),
)

_FLP_SCALAR_FIELDS = (
    "title",
    "option",
    "max_modes",
    "is_3d",
    "profile_range_count",
    "receiver_range_count",
    "source_depth_count",
    "receiver_depth_count",
    "receiver_range_offset_count",
    "source_x_count",
    "source_y_count",
    "receiver_theta_count",
    "node_count",
    "node_env_files",
    "element_count",
    "alpha_min_deg",
    "alpha_max_deg",
    "alpha_count",
    "step_m",
    "step_count",
    "epmult",
)

_FLP_ARRAY_FIELDS = (
    "profile_ranges_km",
    "receiver_ranges_km",
    "source_depths_m",
    "receiver_depths_m",
    "receiver_range_offsets_m",
    "source_x_km",
    "source_y_km",
    "receiver_thetas_deg",
    "node_x_km",
    "node_y_km",
    "elements",
)


def at_file_format_smoke_thresholds() -> BenchmarkThresholds:
    """Return smoke thresholds for canonical AT file-format round trips."""

    return BenchmarkThresholds(
        min_values={
            "sample_count": 1.0,
        },
        equals={
            "prt_issue_count": 0,
        },
        required_true=("file_format_roundtrip_ok",),
        ignore_missing=True,
    )


def at_field_flp_smoke_thresholds() -> BenchmarkThresholds:
    """Return smoke thresholds for canonical AT Field/Field3D ``.flp`` round trips."""

    return BenchmarkThresholds(
        min_values={
            "sample_count": 1.0,
            "file_format_scalar_match_count": float(len(_FLP_SCALAR_FIELDS)),
            "file_format_array_match_count": float(len(_FLP_ARRAY_FIELDS)),
        },
        ignore_missing=True,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_optional_sample(root: Path, relative_path: Path) -> Path | None:
    candidate = root / relative_path
    return candidate if candidate.exists() else None


def _resolve_at_tests_root(root: Path) -> Path | None:
    candidate = root / "tests"
    return candidate if candidate.exists() else None


def _resolve_at_matlab_root(root: Path) -> Path | None:
    candidate = root / "Matlab"
    return candidate if candidate.exists() else None


def _resolve_at_source_root(root: Path) -> Path | None:
    return root if (root / "misc").exists() else None


def _flp_case_name(path: Path) -> str:
    return "field-flp-roundtrip-" + "-".join(path.with_suffix("").parts)


def _array_matches(reference, candidate) -> bool:
    if reference is None:
        return candidate is None
    if candidate is None:
        return False
    reference_arr = np.asarray(reference)
    candidate_arr = np.asarray(candidate)
    if reference_arr.dtype.kind in {"U", "S", "O"} or candidate_arr.dtype.kind in {"U", "S", "O"}:
        return reference_arr.shape == candidate_arr.shape and np.array_equal(reference_arr, candidate_arr)
    return reference_arr.shape == candidate_arr.shape and np.allclose(reference_arr, candidate_arr)


def _field_flp_roundtrip_row(source_path: Path, parsed: ATFieldFLP, reparsed: ATFieldFLP) -> BenchmarkRow:
    scalar_matches = [getattr(parsed, field) == getattr(reparsed, field) for field in _FLP_SCALAR_FIELDS]
    array_matches = [_array_matches(getattr(parsed, field), getattr(reparsed, field)) for field in _FLP_ARRAY_FIELDS]
    scalar_match_count = int(sum(scalar_matches))
    array_match_count = int(sum(array_matches))
    unmatched_fields = [
        field for field, matched in zip(_FLP_SCALAR_FIELDS, scalar_matches) if not matched
    ] + [
        field for field, matched in zip(_FLP_ARRAY_FIELDS, array_matches) if not matched
    ]
    return {
        "case": _flp_case_name(source_path),
        "kind": "at_field_flp_roundtrip",
        "sample_count": scalar_match_count + array_match_count,
        "file_format_roundtrip_ok": not unmatched_fields,
        "file_format_path": source_path.as_posix(),
        "file_format_scalar_match_count": scalar_match_count,
        "file_format_array_match_count": array_match_count,
        "file_format_unmatched_fields": ",".join(unmatched_fields),
        "field_flp_is_3d": parsed.is_3d,
        "field_flp_option": parsed.option,
        "field_flp_max_modes": parsed.max_modes,
        "field_flp_receiver_range_count": parsed.receiver_range_count,
        "field_flp_receiver_depth_count": parsed.receiver_depth_count,
        "field_flp_receiver_theta_count": parsed.receiver_theta_count,
        "field_flp_node_count": parsed.node_count,
        "field_flp_element_count": parsed.element_count,
    }


def _mod_metadata_multi_medium_case(work_dir: Path) -> BenchmarkRow:
    z_bins = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    modes = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.1j, 0.2 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.25 - 0.1j, 0.5 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    wavenumbers = np.array([0.1 + 0.001j, 0.08 + 0.002j], dtype=np.complex64)
    metadata = {
        "freq_vec": [50.0],
        "media": [
            {"n": 2, "material": "ACOUSTIC", "depth": 0.0, "rho": 1.0},
            {"n": 1, "material": "ACOUSTIC", "depth": 20.0, "rho": 1.5},
        ],
        "top": {"bc": "V", "c_p": 0.0, "c_s": 0.0, "rho": 0.0, "depth": 0.0},
        "bottom": {"bc": "A", "c_p": 1600.0, "c_s": 0.0, "rho": 1.8, "depth": 30.0},
    }
    mod_path = work_dir / "multi_medium.mod"
    write_at_mod(mod_path, modes, z_bins, title="canonical multi-medium mod", wavenumbers=wavenumbers, metadata=metadata)
    return _mod_roundtrip_row("mod-roundtrip-multi-medium-acoustic", mod_path, modes, z_bins, wavenumbers, metadata)


def _mod_metadata_complex_halfspace_case(work_dir: Path) -> BenchmarkRow:
    z_bins = np.array([0.0, 20.0, 40.0, 60.0], dtype=np.float32)
    modes = np.array(
        [
            [1.0 + 0.0j, 0.4 + 0.2j, 0.1 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.1 - 0.1j, 0.4 + 0.2j, 0.8 + 0.0j],
            [0.1 + 0.0j, 0.2 + 0.05j, 0.3 - 0.1j, 0.4 + 0.0j],
        ],
        dtype=np.complex64,
    )
    wavenumbers = np.array([0.15 + 0.001j, 0.12 + 0.002j, 0.1 + 0.003j], dtype=np.complex64)
    metadata = {
        "freq_vec": [25.0, 50.0],
        "media": [
            {"n": 2, "material": "ACOUSTIC", "depth": 0.0, "rho": 1.0},
            {"n": 1, "material": "ELASTIC", "depth": 40.0, "rho": 2.1},
        ],
        "top": {
            "bc": "A",
            "c_p": 1480.0 + 0.25j,
            "c_s": 0.0 + 0.0j,
            "rho": 1.0,
            "depth": 0.0,
        },
        "bottom": {
            "bc": "A",
            "c_p": 1700.0 + 0.75j,
            "c_s": 600.0 + 0.125j,
            "rho": 2.2,
            "depth": 60.0,
        },
    }
    mod_path = work_dir / "complex_halfspace.mod"
    write_at_mod(mod_path, modes, z_bins, title="canonical complex halfspace mod", wavenumbers=wavenumbers, metadata=metadata)
    return _mod_roundtrip_row("mod-roundtrip-complex-halfspace-elastic", mod_path, modes, z_bins, wavenumbers, metadata)


def _complex_close(actual, expected) -> bool:
    return bool(np.allclose(np.asarray(actual, dtype=np.complex128), np.asarray(expected, dtype=np.complex128)))


def _mod_roundtrip_row(
    case_name: str,
    mod_path: Path,
    modes: np.ndarray,
    z_bins: np.ndarray,
    wavenumbers: np.ndarray,
    metadata: dict,
) -> BenchmarkRow:
    parsed_modes, parsed_z = read_at_mod(mod_path)
    parsed_wavenumbers = read_at_mod_wavenumbers(mod_path)
    parsed_metadata = read_at_mod_metadata(mod_path)
    parsed_media = tuple(parsed_metadata.get("media") or ())
    expected_media = tuple(metadata.get("media") or ())
    checks = {
        "modes": _array_matches(modes, parsed_modes),
        "z_bins": _array_matches(z_bins, parsed_z),
        "wavenumbers": _array_matches(wavenumbers, parsed_wavenumbers),
        "freq_vec": _array_matches(np.asarray(metadata["freq_vec"], dtype=np.float64), parsed_metadata.get("freq_vec")),
        "media_count": parsed_metadata.get("nmedia") == len(metadata["media"]),
        "media_materials": tuple(medium.get("material") for medium in parsed_media) == tuple(
            medium.get("material") for medium in expected_media
        ),
        "media_points": _array_matches(
            np.asarray([medium.get("n", 0) for medium in expected_media], dtype=np.int32),
            np.asarray([medium.get("n", 0) for medium in parsed_media], dtype=np.int32),
        ),
        "media_depths": _array_matches(
            np.asarray([medium.get("depth", 0.0) for medium in expected_media], dtype=np.float64),
            np.asarray([medium.get("depth", 0.0) for medium in parsed_media], dtype=np.float64),
        ),
        "media_rhos": _array_matches(
            np.asarray([medium.get("rho", 0.0) for medium in expected_media], dtype=np.float64),
            np.asarray([medium.get("rho", 0.0) for medium in parsed_media], dtype=np.float64),
        ),
        "mode_count": parsed_metadata.get("mode_count") == modes.shape[0],
        "top_bc": parsed_metadata.get("top", {}).get("bc") == metadata["top"]["bc"],
        "top_c_p": _complex_close(parsed_metadata.get("top", {}).get("c_p"), metadata["top"]["c_p"]),
        "top_c_s": _complex_close(parsed_metadata.get("top", {}).get("c_s"), metadata["top"]["c_s"]),
        "top_rho": np.isclose(parsed_metadata.get("top", {}).get("rho"), metadata["top"]["rho"]),
        "top_depth": np.isclose(parsed_metadata.get("top", {}).get("depth"), metadata["top"]["depth"]),
        "bottom_bc": parsed_metadata.get("bottom", {}).get("bc") == metadata["bottom"]["bc"],
        "bottom_c_p": _complex_close(parsed_metadata.get("bottom", {}).get("c_p"), metadata["bottom"]["c_p"]),
        "bottom_c_s": _complex_close(parsed_metadata.get("bottom", {}).get("c_s"), metadata["bottom"]["c_s"]),
        "bottom_rho": np.isclose(parsed_metadata.get("bottom", {}).get("rho"), metadata["bottom"]["rho"]),
        "bottom_depth": np.isclose(parsed_metadata.get("bottom", {}).get("depth"), metadata["bottom"]["depth"]),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    parsed_modes_abs = np.abs(np.asarray(parsed_modes))
    parsed_wavenumber_abs = np.abs(np.asarray(parsed_wavenumbers))
    finite_mode_values = parsed_modes_abs[np.isfinite(parsed_modes_abs)]
    finite_wavenumbers = parsed_wavenumber_abs[np.isfinite(parsed_wavenumber_abs)]
    return {
        "case": case_name,
        "kind": "at_mod_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": mod_path.name,
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(
            checks["modes"] + checks["z_bins"] + checks["wavenumbers"] + checks["freq_vec"]
        ),
        "file_format_unmatched_fields": ",".join(unmatched),
        "mod_mode_count": int(parsed_metadata.get("mode_count") or 0),
        "mod_media_count": int(parsed_metadata.get("nmedia") or 0),
        "mod_frequency_count": int(parsed_metadata.get("nfreq") or 0),
        "mod_depth_sample_count": int(parsed_z.size),
        "mod_shape_mode_count": int(parsed_modes.shape[0]) if parsed_modes.ndim >= 1 else 0,
        "mod_shape_depth_count": int(parsed_modes.shape[1]) if parsed_modes.ndim >= 2 else 0,
        "mod_value_finite_count": int(finite_mode_values.size),
        "mod_value_max_abs": float(finite_mode_values.max()) if finite_mode_values.size else None,
        "mod_wavenumber_count": int(parsed_wavenumbers.size),
        "mod_wavenumber_min_abs": float(finite_wavenumbers.min()) if finite_wavenumbers.size else None,
        "mod_wavenumber_max_abs": float(finite_wavenumbers.max()) if finite_wavenumbers.size else None,
        "mod_top_bc": parsed_metadata.get("top", {}).get("bc"),
        "mod_top_rho": parsed_metadata.get("top", {}).get("rho"),
        "mod_top_depth": parsed_metadata.get("top", {}).get("depth"),
        "mod_bottom_bc": parsed_metadata.get("bottom", {}).get("bc"),
        "mod_bottom_rho": parsed_metadata.get("bottom", {}).get("rho"),
        "mod_bottom_depth": parsed_metadata.get("bottom", {}).get("depth"),
        "mod_media_materials": ",".join(str(medium.get("material") or "") for medium in parsed_media),
        "mod_media_point_count_min": min((int(medium.get("n") or 0) for medium in parsed_media), default=0),
        "mod_media_point_count_max": max((int(medium.get("n") or 0) for medium in parsed_media), default=0),
        "mod_media_depth_min": min((float(medium.get("depth") or 0.0) for medium in parsed_media), default=None),
        "mod_media_depth_max": max((float(medium.get("depth") or 0.0) for medium in parsed_media), default=None),
        "mod_media_rho_min": min((float(medium.get("rho") or 0.0) for medium in parsed_media), default=None),
        "mod_media_rho_max": max((float(medium.get("rho") or 0.0) for medium in parsed_media), default=None),
    }


def _mod_real_sample_roundtrip_row(display_path: Path, source_path: Path, work_dir: Path) -> BenchmarkRow:
    modes, z_bins = read_at_mod(source_path)
    wavenumbers = read_at_mod_wavenumbers(source_path)
    metadata = read_at_mod_metadata(source_path)
    roundtrip_path = work_dir / display_path
    write_at_mod(
        roundtrip_path,
        modes,
        z_bins,
        title=str(metadata.get("title") or display_path.stem),
        wavenumbers=wavenumbers,
        metadata=metadata,
    )
    row = _mod_roundtrip_row(
        "mod-roundtrip-real-" + "-".join(display_path.with_suffix("").parts),
        roundtrip_path,
        modes,
        z_bins,
        wavenumbers,
        metadata,
    )
    row["file_format_path"] = display_path.as_posix()
    return row


def _shd_multifrequency_source_grid_case(work_dir: Path) -> BenchmarkRow:
    depths = np.array([10.0], dtype=np.float32)
    ranges = np.array([100.0, 200.0], dtype=np.float64)
    source_x = np.array([0.0, 250.0], dtype=np.float64)
    source_y = np.array([0.0, 100.0], dtype=np.float64)
    source_depths = np.array([20.0], dtype=np.float32)
    bearings = np.array([0.0, 45.0], dtype=np.float64)
    frequencies = np.array([100.0, 200.0], dtype=np.float64)
    pressure = np.zeros((2, 2, 2, 2, 1, 1, 2), dtype=np.complex128)
    for ifreq in range(2):
        for isx in range(2):
            for isy in range(2):
                for itheta in range(2):
                    pressure[ifreq, isx, isy, itheta, 0, 0] = (
                        1000.0 * ifreq
                        + 100.0 * isx
                        + 10.0 * isy
                        + itheta
                        + np.array([1.0, 2.0])
                        + 1j * np.array([0.25, 0.5])
                    )
    shd_path = work_dir / "broadband_source_grid.shd"
    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="canonical broadband source grid",
        frequencies_hz=frequencies,
        source_x_m=source_x,
        source_y_m=source_y,
        source_depths_m=source_depths,
        bearings_deg=bearings,
        attenuation=0.125,
    )
    shade = read_at_shd_full(shd_path, frequency_hz=190.0, source_x_m=240.0, source_y_m=90.0)
    checks = {
        "frequency_vector": _array_matches(frequencies, shade.frequency_vector_hz),
        "source_x": _array_matches(source_x, shade.source_x_m),
        "source_y": _array_matches(source_y, shade.source_y_m),
        "bearings": _array_matches(bearings, shade.bearings_deg),
        "selected_frequency": shade.selected_frequency_index == 1 and np.isclose(shade.frequency_hz, 200.0),
        "selected_source_x": shade.selected_source_x_index == 1,
        "selected_source_y": shade.selected_source_y_index == 1,
        "pressure": _array_matches(pressure[1, 1, 1].astype(np.complex64), shade.pressure),
        "pressure_frequency_full": _array_matches(pressure.astype(np.complex64), shade.pressure_frequency_full),
        "attenuation": np.isclose(shade.attenuation, 0.125),
    }
    return _shd_roundtrip_row(
        "shd-roundtrip-broadband-source-grid",
        shd_path,
        shade,
        checks,
    )


def _shd_tl_compressed_source_xy_case(work_dir: Path) -> BenchmarkRow:
    depths = np.array([10.0, 20.0], dtype=np.float32)
    ranges = np.array([100.0, 500.0], dtype=np.float64)
    source_x = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    source_y = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    pressure = np.ones((3, 3, 1, 1, 2, 2), dtype=np.complex128)
    shd_path = work_dir / "tl_compressed_source_xy.shd"
    write_at_shd(
        shd_path,
        pressure,
        depths,
        ranges,
        title="canonical TL compressed source xy",
        frequency=150.0,
        source_x_m=source_x,
        source_y_m=source_y,
        plot_type="TL",
    )
    shade = read_at_shd_full(shd_path)
    checks = {
        "plot_type": shade.plot_type == "TL",
        "source_x": _array_matches(source_x, shade.source_x_m),
        "source_y": _array_matches(source_y, shade.source_y_m),
        "receiver_depths": _array_matches(depths, shade.receiver_depths_m),
        "receiver_ranges": _array_matches(ranges, shade.receiver_ranges_m),
        "pressure_full": _array_matches(pressure.astype(np.complex64), shade.pressure_full),
    }
    return _shd_roundtrip_row(
        "shd-roundtrip-tl-compressed-source-xy",
        shd_path,
        shade,
        checks,
    )


def _shd_real_sample_roundtrip_row(display_path: Path, source_path: Path, work_dir: Path) -> BenchmarkRow:
    parsed = read_at_shd_full(source_path)
    roundtrip_path = work_dir / display_path
    write_at_shd(
        roundtrip_path,
        parsed.pressure_frequency_full,
        parsed.receiver_depths_m,
        parsed.receiver_ranges_m,
        title=parsed.title or source_path.stem,
        frequencies_hz=parsed.frequency_vector_hz,
        source_x_m=parsed.source_x_m,
        source_y_m=parsed.source_y_m,
        source_depths_m=parsed.source_depths_m,
        bearings_deg=parsed.bearings_deg,
        plot_type="" if parsed.plot_type is None else parsed.plot_type,
        attenuation=0.0 if parsed.attenuation is None else parsed.attenuation,
    )
    reparsed = read_at_shd_full(roundtrip_path)
    checks = {
        "plot_type": parsed.plot_type == reparsed.plot_type,
        "frequency_vector": _array_matches(parsed.frequency_vector_hz, reparsed.frequency_vector_hz),
        "source_x": _array_matches(parsed.source_x_m, reparsed.source_x_m),
        "source_y": _array_matches(parsed.source_y_m, reparsed.source_y_m),
        "source_depths": _array_matches(parsed.source_depths_m, reparsed.source_depths_m),
        "bearings": _array_matches(parsed.bearings_deg, reparsed.bearings_deg),
        "receiver_depths": _array_matches(parsed.receiver_depths_m, reparsed.receiver_depths_m),
        "receiver_ranges": _array_matches(parsed.receiver_ranges_m, reparsed.receiver_ranges_m),
        "pressure_frequency_full": _array_matches(parsed.pressure_frequency_full, reparsed.pressure_frequency_full),
    }
    return _shd_roundtrip_row(
        "shd-roundtrip-real-" + "-".join(display_path.with_suffix("").parts),
        roundtrip_path,
        reparsed,
        checks,
    )


def _shd_roundtrip_row(
    case_name: str,
    shd_path: Path,
    shade,
    checks: dict[str, bool],
) -> BenchmarkRow:
    unmatched = [field for field, matched in checks.items() if not matched]
    frequency_count = int(shade.frequency_vector_hz.size) if shade.frequency_vector_hz is not None else 0
    source_x_count = int(shade.source_x_m.size) if shade.source_x_m is not None else 0
    source_y_count = int(shade.source_y_m.size) if shade.source_y_m is not None else 0
    pressure_array = (
        shade.pressure_frequency_full
        if shade.pressure_frequency_full is not None
        else shade.pressure_full
        if shade.pressure_full is not None
        else shade.pressure
    )
    pressure_abs = np.abs(np.asarray(pressure_array))
    finite_pressure = pressure_abs[np.isfinite(pressure_abs)]
    return {
        "case": case_name,
        "kind": "at_shd_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": shd_path.name,
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(sum(checks.values())),
        "file_format_unmatched_fields": ",".join(unmatched),
        "shd_plot_type": shade.plot_type,
        "shd_frequency_count": frequency_count,
        "shd_source_x_count": source_x_count,
        "shd_source_y_count": source_y_count,
        "shd_source_depth_count": int(shade.source_depths_m.size),
        "shd_bearing_count": int(shade.bearings_deg.size),
        "shd_receiver_depth_count": int(shade.receiver_depths_m.size),
        "shd_receiver_range_count": int(shade.receiver_ranges_m.size),
        "shd_pressure_sample_count": int(pressure_abs.size),
        "shd_pressure_finite_count": int(finite_pressure.size),
        "shd_pressure_max_abs": float(finite_pressure.max()) if finite_pressure.size else None,
        "shd_pressure_min_abs": float(finite_pressure.min()) if finite_pressure.size else None,
    }


def _boundary_profile_case(
    case_name: str,
    work_dir: Path,
    suffix: str,
    profile: ATBoundaryProfile,
) -> BenchmarkRow:
    path = work_dir / f"{case_name}{suffix}"
    write_at_boundary_profile(path, profile)
    parsed = read_at_ati(path) if suffix == ".ati" else read_at_bty(path)
    checks = {
        "interpolation": parsed.interpolation == profile.interpolation.upper()[0],
        "ranges_m": _array_matches(profile.ranges_m, parsed.ranges_m),
        "values_m": _array_matches(profile.values_m, parsed.values_m),
        "geoacoustics": _array_matches(profile.geoacoustics, parsed.geoacoustics),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    geoacoustics = parsed.geoacoustics
    return {
        "case": case_name,
        "kind": "at_boundary_profile_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": path.name,
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(checks["ranges_m"] + checks["values_m"] + checks["geoacoustics"]),
        "file_format_unmatched_fields": ",".join(unmatched),
        "boundary_profile_suffix": suffix,
        "boundary_interpolation": parsed.interpolation,
        "boundary_point_count": int(parsed.ranges_m.size),
        "boundary_has_geoacoustics": geoacoustics is not None,
    }


def _boundary_profile_cases(work_dir: Path) -> tuple[BenchmarkRow, BenchmarkRow]:
    ranges = np.array([0.0, 1250.0, 2500.0, 5000.0], dtype=np.float64)
    surface_altimetry = np.array([0.0, -1.5, 0.75, 0.0], dtype=np.float64)
    bottom_depths = np.array([90.0, 110.0, 150.0, 125.0], dtype=np.float64)
    geoacoustics = np.array(
        [
            [1500.0, 0.0, 1.0, 0.0, 0.0],
            [1520.0, 0.0, 1.2, 0.1, 0.0],
            [1580.0, 250.0, 1.8, 0.2, 0.05],
            [1625.0, 300.0, 2.0, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    return (
        _boundary_profile_case(
            "boundary-roundtrip-surface-ati-linear",
            work_dir,
            ".ati",
            ATBoundaryProfile("L", ranges, surface_altimetry),
        ),
        _boundary_profile_case(
            "boundary-roundtrip-bottom-bty-long-format",
            work_dir,
            ".bty",
            ATBoundaryProfile("C", ranges, bottom_depths, geoacoustics),
        ),
    )


def _boundary_profile_real_sample_roundtrip_row(display_path: Path, source_path: Path, work_dir: Path) -> BenchmarkRow:
    suffix = source_path.suffix.lower()
    parsed = read_at_ati(source_path) if suffix == ".ati" else read_at_bty(source_path)
    roundtrip_path = work_dir / display_path
    write_at_boundary_profile(roundtrip_path, parsed)
    reparsed = read_at_ati(roundtrip_path) if suffix == ".ati" else read_at_bty(roundtrip_path)
    checks = {
        "interpolation": parsed.interpolation == reparsed.interpolation,
        "ranges_m": _array_matches(parsed.ranges_m, reparsed.ranges_m),
        "values_m": _array_matches(parsed.values_m, reparsed.values_m),
        "geoacoustics": _array_matches(parsed.geoacoustics, reparsed.geoacoustics),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": "boundary-roundtrip-real-" + "-".join(display_path.with_suffix("").parts),
        "kind": "at_boundary_profile_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(checks["ranges_m"] + checks["values_m"] + checks["geoacoustics"]),
        "file_format_unmatched_fields": ",".join(unmatched),
        "boundary_profile_suffix": suffix,
        "boundary_interpolation": reparsed.interpolation,
        "boundary_point_count": int(reparsed.ranges_m.size),
        "boundary_has_geoacoustics": reparsed.geoacoustics is not None,
    }


def _reflection_table_case(
    case_name: str,
    work_dir: Path,
    suffix: str,
    table: ATReflectionTable,
) -> BenchmarkRow:
    path = work_dir / f"{case_name}{suffix}"
    write_at_reflection_table(path, table)
    if suffix == ".irc":
        parsed = read_at_irc(path)
    elif suffix == ".trc":
        parsed = read_at_trc(path)
    else:
        parsed = read_at_brc(path)
    checks = {
        "angles_deg": _array_matches(table.angles_deg, parsed.angles_deg),
        "amplitudes": _array_matches(table.amplitudes, parsed.amplitudes),
        "phases_deg": _array_matches(table.phases_deg, parsed.phases_deg),
        "coefficients": _array_matches(table.coefficients, parsed.coefficients),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": case_name,
        "kind": "at_reflection_table_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": path.name,
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(sum(checks.values())),
        "file_format_unmatched_fields": ",".join(unmatched),
        "reflection_table_suffix": suffix,
        "reflection_angle_count": int(parsed.angles_deg.size),
        "reflection_amplitude_min": float(np.min(parsed.amplitudes)),
        "reflection_amplitude_max": float(np.max(parsed.amplitudes)),
        "reflection_phase_min_deg": float(np.min(parsed.phases_deg)),
        "reflection_phase_max_deg": float(np.max(parsed.phases_deg)),
    }


def _reflection_table_cases(work_dir: Path) -> tuple[BenchmarkRow, BenchmarkRow, BenchmarkRow]:
    angles = np.array([0.0, 15.0, 35.0, 60.0, 85.0], dtype=np.float64)
    return (
        _reflection_table_case(
            "reflection-roundtrip-surface-irc-complex",
            work_dir,
            ".irc",
            ATReflectionTable(
                angles,
                np.array([0.02, 0.05, 0.15, 0.35, 0.9], dtype=np.float64),
                np.array([0.0, -12.0, -45.0, -120.0, -178.0], dtype=np.float64),
            ),
        ),
        _reflection_table_case(
            "reflection-roundtrip-bottom-brc-complex",
            work_dir,
            ".brc",
            ATReflectionTable(
                angles,
                np.array([0.95, 0.9, 0.78, 0.55, 0.25], dtype=np.float64),
                np.array([180.0, 150.0, 90.0, 25.0, -10.0], dtype=np.float64),
            ),
        ),
        _reflection_table_case(
            "reflection-roundtrip-top-trc-complex",
            work_dir,
            ".trc",
            ATReflectionTable(
                angles,
                np.array([0.85, 0.7, 0.45, 0.2, 0.05], dtype=np.float64),
                np.array([170.0, 125.0, 55.0, -20.0, -80.0], dtype=np.float64),
            ),
        ),
    )


def _reflection_table_real_sample_roundtrip_row(display_path: Path, source_path: Path, work_dir: Path) -> BenchmarkRow:
    suffix = source_path.suffix.lower()
    parsed = read_at_irc(source_path) if suffix == ".irc" else read_at_brc(source_path)
    roundtrip_path = work_dir / display_path
    write_at_reflection_table(roundtrip_path, parsed)
    reparsed = read_at_irc(roundtrip_path) if suffix == ".irc" else read_at_brc(roundtrip_path)
    checks = {
        "angles_deg": _array_matches(parsed.angles_deg, reparsed.angles_deg),
        "amplitudes": _array_matches(parsed.amplitudes, reparsed.amplitudes),
        "phases_deg": _array_matches(parsed.phases_deg, reparsed.phases_deg),
        "coefficients": _array_matches(parsed.coefficients, reparsed.coefficients),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": "reflection-roundtrip-real-" + "-".join(display_path.with_suffix("").parts),
        "kind": "at_reflection_table_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(sum(checks.values())),
        "file_format_unmatched_fields": ",".join(unmatched),
        "reflection_table_suffix": suffix,
        "reflection_angle_count": int(reparsed.angles_deg.size),
        "reflection_amplitude_min": float(np.min(reparsed.amplitudes)),
        "reflection_amplitude_max": float(np.max(reparsed.amplitudes)),
        "reflection_phase_min_deg": float(np.min(reparsed.phases_deg)),
        "reflection_phase_max_deg": float(np.max(reparsed.phases_deg)),
    }


def _source_beam_pattern_real_sample_roundtrip_row(
    display_path: Path,
    source_path: Path,
    work_dir: Path,
) -> BenchmarkRow:
    parsed = read_at_source_beam_pattern(source_path)
    roundtrip_path = work_dir / display_path
    write_at_source_beam_pattern(roundtrip_path, parsed)
    reparsed = read_at_source_beam_pattern(roundtrip_path)
    checks = {
        "angles_deg": _array_matches(parsed.angles_deg, reparsed.angles_deg),
        "levels_db": _array_matches(parsed.levels_db, reparsed.levels_db),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": "source-beam-pattern-roundtrip-real-" + "-".join(display_path.with_suffix("").parts),
        "kind": "at_source_beam_pattern_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(sum(checks.values())),
        "file_format_unmatched_fields": ",".join(unmatched),
        "source_beam_pattern_sample_count": int(reparsed.angles_deg.size),
        "source_beam_pattern_angle_min_deg": float(np.min(reparsed.angles_deg)),
        "source_beam_pattern_angle_max_deg": float(np.max(reparsed.angles_deg)),
        "source_beam_pattern_level_min_db": float(np.min(reparsed.levels_db)),
        "source_beam_pattern_level_max_db": float(np.max(reparsed.levels_db)),
    }


def _source_beam_pattern_case(work_dir: Path) -> BenchmarkRow:
    path = work_dir / "source_beam_pattern_canonical.sbp"
    pattern = ATSourceBeamPattern(
        angles_deg=np.array([-30.0, -10.0, 0.0, 10.0, 30.0], dtype=np.float64),
        levels_db=np.array([-12.0, -3.0, 0.0, -3.0, -12.0], dtype=np.float64),
    )
    write_at_source_beam_pattern(path, pattern)
    parsed = read_at_source_beam_pattern(path)
    checks = {
        "angles_deg": _array_matches(pattern.angles_deg, parsed.angles_deg),
        "levels_db": _array_matches(pattern.levels_db, parsed.levels_db),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": "source-beam-pattern-roundtrip-canonical",
        "kind": "at_source_beam_pattern_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": path.name,
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(sum(checks.values())),
        "file_format_unmatched_fields": ",".join(unmatched),
        "source_beam_pattern_sample_count": int(parsed.angles_deg.size),
        "source_beam_pattern_angle_min_deg": float(np.min(parsed.angles_deg)),
        "source_beam_pattern_angle_max_deg": float(np.max(parsed.angles_deg)),
        "source_beam_pattern_level_min_db": float(np.min(parsed.levels_db)),
        "source_beam_pattern_level_max_db": float(np.max(parsed.levels_db)),
    }


def _env_summary_row(case_name: str, path: Path, display_path: Path | None = None) -> BenchmarkRow:
    summary = read_at_env_summary(path)
    return {
        "case": case_name,
        "kind": "at_env_summary_parse",
        "sample_count": 1,
        "file_format_roundtrip_ok": True,
        "file_format_read_ok": True,
        "file_format_path": str(display_path or path.name),
        "env_model_family": summary.model_family,
        "env_title": summary.title,
        "env_frequency_hz": summary.frequency_hz,
        "env_nmedia": summary.nmedia,
        "env_ssp_option": summary.ssp_option,
        "env_top_option": summary.top_option,
        "env_bottom_option": summary.bottom_option,
        "env_run_type": summary.run_type,
        "env_source_depth_count": summary.source_depth_count,
        "env_receiver_depth_count": summary.receiver_depth_count,
        "env_receiver_range_count": summary.receiver_range_count,
        "env_source_x_count": summary.source_x_count,
        "env_source_y_count": summary.source_y_count,
        "env_receiver_bearing_count": summary.receiver_bearing_count,
        "env_beam_count": summary.beam_count,
        "env_beam_bearing_count": summary.beam_bearing_count,
        "env_box_range_km": summary.box_range_km,
        "env_box_y_km": summary.box_y_km,
        "env_box_z_m": summary.box_z_m,
    }


def _env_summary_cases(work_dir: Path) -> tuple[BenchmarkRow, BenchmarkRow, BenchmarkRow]:
    bellhop_path = work_dir / "env-summary-bellhop.env"
    bellhop_path.write_text(
        "'Bellhop summary'\n"
        "250.0\n"
        "1\n"
        "'CVW'\n"
        "2 0.0 100.0\n"
        "0.0 1500.0 /\n"
        "100.0 1500.0 /\n"
        "'A' 0.0\n"
        "100.0 1590.0 0.0 1.2 0.5 0.0 /\n"
        "1\n"
        "50.0 /\n"
        "101\n"
        "0.0 100.0 /\n"
        "501\n"
        "0.0 5.0 /\n"
        "'C'\n"
        "1789\n"
        "-89 89 /\n"
        "0.0 101.0 5.1\n"
    )
    kraken_path = work_dir / "env-summary-kraken.env"
    kraken_path.write_text(
        "'Kraken summary'\n"
        "250.0\n"
        "1\n"
        "'CVW'\n"
        "1000 0.0 100.0\n"
        "0.0 1500.0 /\n"
        "100.0 1500.0 /\n"
        "'A' 0.0\n"
        "100.0 1590.0 0.0 1.2 0.5 /\n"
        "1400.0 20000.0\n"
        "1000.0\n"
        "1\n"
        "50.0 /\n"
        "101\n"
        "0.0 100.0 /\n"
    )
    bellhop3d_path = work_dir / "env-summary-bellhop3d.env"
    bellhop3d_path.write_text(
        "'Bellhop3D summary'\n"
        "50.0\n"
        "1\n"
        "'CVW'\n"
        "2 0.0 20000.0\n"
        "0.0 1548.52 /\n"
        "20000.0 1551.91 /\n"
        "'A~' 0.0\n"
        "20000.0 1600.0 0.0 1.8 0.8 /\n"
        "1\n"
        "0.0 /\n"
        "1\n"
        "0.0 /\n"
        "1\n"
        "1000.0 /\n"
        "1\n"
        "1000.0 5000.0 /\n"
        "1001\n"
        "0.0 100.0 /\n"
        "37\n"
        "0.0 360.0 /\n"
        "'CG   3'\n"
        "200\n"
        "-14.66 20.0 /\n"
        "144\n"
        "0.0 360.0 /\n"
        "0.0 99.1 99.1 20500.0\n"
    )
    return (
        _env_summary_row("env-summary-bellhop", bellhop_path),
        _env_summary_row("env-summary-kraken", kraken_path),
        _env_summary_row("env-summary-bellhop3d", bellhop3d_path),
    )


def _env_summary_real_sample_row(display_path: Path, source_path: Path) -> BenchmarkRow:
    return _env_summary_row(
        "env-summary-real-" + "-".join(display_path.with_suffix("").parts),
        source_path,
        display_path,
    )


def _ram_in_summary_case_name(path: Path) -> str:
    return "ram-in-summary-real-" + "-".join(path.with_suffix("").parts)


def _ram_in_summary_real_sample_row(display_path: Path, source_path: Path) -> BenchmarkRow:
    summary = read_at_ram_in_summary(source_path)
    checks = {
        "frequency": summary.frequency_hz > 0.0,
        "range_grid": summary.rmax_m > 0.0 and summary.dr_m > 0.0,
        "depth_grid": summary.zmax_m > 0.0 and summary.dz_m > 0.0,
        "bathymetry": summary.bathymetry_point_count >= 2,
        "profiles": summary.range_dependent_profile_count >= 1,
        "ssp": summary.ssp_point_count >= summary.range_dependent_profile_count,
        "bottom": summary.bottom_sound_speed_point_count >= summary.range_dependent_profile_count,
        "density": summary.density_point_count >= summary.range_dependent_profile_count,
        "attenuation": summary.attenuation_point_count >= summary.range_dependent_profile_count,
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": _ram_in_summary_case_name(display_path),
        "kind": "at_ram_in_summary_parse",
        "sample_count": len(checks),
        "file_format_read_ok": not unmatched,
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": sum(int(value) for value in checks.values()),
        "file_format_array_match_count": 0,
        "file_format_unmatched_fields": ",".join(unmatched),
        "ram_title": summary.title,
        "ram_frequency_hz": summary.frequency_hz,
        "ram_source_depth_m": summary.source_depth_m,
        "ram_receiver_depth_m": summary.receiver_depth_m,
        "ram_rmax_m": summary.rmax_m,
        "ram_dr_m": summary.dr_m,
        "ram_receiver_range_decimation": summary.receiver_range_decimation,
        "ram_zmax_m": summary.zmax_m,
        "ram_dz_m": summary.dz_m,
        "ram_depth_decimation": summary.depth_decimation,
        "ram_plot_depth_m": summary.plot_depth_m,
        "ram_reference_sound_speed_mps": summary.reference_sound_speed_mps,
        "ram_profile_count": summary.range_dependent_profile_count,
        "ram_sediment_layer_count": summary.sediment_layer_count,
        "ram_source_range_m": summary.source_range_m,
        "ram_bathymetry_point_count": summary.bathymetry_point_count,
        "ram_profile_range_count": summary.profile_range_count,
        "ram_ssp_point_count": summary.ssp_point_count,
        "ram_bottom_sound_speed_point_count": summary.bottom_sound_speed_point_count,
        "ram_density_point_count": summary.density_point_count,
        "ram_attenuation_point_count": summary.attenuation_point_count,
        "ram_bathymetry_range_min_m": summary.bathymetry_range_min_m,
        "ram_bathymetry_range_max_m": summary.bathymetry_range_max_m,
        "ram_bathymetry_depth_min_m": summary.bathymetry_depth_min_m,
        "ram_bathymetry_depth_max_m": summary.bathymetry_depth_max_m,
        "ram_profile_range_min_m": summary.profile_range_min_m,
        "ram_profile_range_max_m": summary.profile_range_max_m,
        "ram_sound_speed_min_mps": summary.sound_speed_min_mps,
        "ram_sound_speed_max_mps": summary.sound_speed_max_mps,
        "ram_bottom_sound_speed_min_mps": summary.bottom_sound_speed_min_mps,
        "ram_bottom_sound_speed_max_mps": summary.bottom_sound_speed_max_mps,
        "ram_density_min_gpcm3": summary.density_min_gpcm3,
        "ram_density_max_gpcm3": summary.density_max_gpcm3,
        "ram_attenuation_min_db_per_wavelength": summary.attenuation_min_db_per_wavelength,
        "ram_attenuation_max_db_per_wavelength": summary.attenuation_max_db_per_wavelength,
    }


def _matlab_script_inventory_case_name(path: Path) -> str:
    return "matlab-script-inventory-real-" + "-".join(path.with_suffix("").parts)


def _matlab_script_inventory_real_sample_row(display_path: Path, source_path: Path) -> BenchmarkRow:
    summary = read_at_matlab_script_summary(source_path)
    total_commands = sum(summary.at_command_counts.values())
    total_external_commands = sum(summary.external_command_counts.values())
    checks = {
        "line_count": summary.line_count > 0,
        "path": source_path.suffix.lower() == ".m",
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    command_items = ",".join(f"{name}:{count}" for name, count in sorted(summary.at_command_counts.items()))
    external_command_items = ",".join(
        f"{name}:{count}" for name, count in sorted(summary.external_command_counts.items())
    )
    return {
        "case": _matlab_script_inventory_case_name(display_path),
        "kind": "at_matlab_script_inventory_parse",
        "sample_count": len(checks),
        "file_format_read_ok": not unmatched,
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": sum(int(value) for value in checks.values()),
        "file_format_array_match_count": 0,
        "file_format_unmatched_fields": ",".join(unmatched),
        "matlab_script_line_count": summary.line_count,
        "matlab_script_code_line_count": summary.code_line_count,
        "matlab_script_comment_line_count": summary.comment_line_count,
        "matlab_script_function_count": summary.function_count,
        "matlab_script_function_names": ",".join(summary.function_names),
        "matlab_script_at_command_count": total_commands,
        "matlab_script_at_commands": command_items,
        "matlab_script_external_command_count": total_external_commands,
        "matlab_script_external_commands": external_command_items,
        "matlab_script_referenced_extensions": ",".join(summary.referenced_extensions),
    }


def _fortran_source_inventory_case_name(path: Path) -> str:
    return "fortran-source-inventory-real-" + "-".join(path.with_suffix("").parts)


def _fortran_source_inventory_real_sample_row(display_path: Path, source_path: Path) -> BenchmarkRow:
    summary = read_at_fortran_source_summary(source_path)
    checks = {
        "line_count": summary.line_count > 0,
        "path": source_path.suffix.lower() == ".f90",
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": _fortran_source_inventory_case_name(display_path),
        "kind": "at_fortran_source_inventory_parse",
        "sample_count": len(checks),
        "file_format_read_ok": not unmatched,
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": sum(int(value) for value in checks.values()),
        "file_format_array_match_count": 0,
        "file_format_unmatched_fields": ",".join(unmatched),
        "fortran_source_family": display_path.parts[0] if display_path.parts else "",
        "fortran_source_line_count": summary.line_count,
        "fortran_source_code_line_count": summary.code_line_count,
        "fortran_source_comment_line_count": summary.comment_line_count,
        "fortran_source_program_count": len(summary.program_names),
        "fortran_source_program_names": ",".join(summary.program_names),
        "fortran_source_module_count": len(summary.module_names),
        "fortran_source_module_names": ",".join(summary.module_names),
        "fortran_source_subroutine_count": len(summary.subroutine_names),
        "fortran_source_subroutine_names": ",".join(summary.subroutine_names),
        "fortran_source_function_count": len(summary.function_names),
        "fortran_source_function_names": ",".join(summary.function_names),
        "fortran_source_used_module_count": len(summary.used_modules),
        "fortran_source_used_modules": ",".join(summary.used_modules),
    }


def _bathymetry_3d_real_sample_roundtrip_row(display_path: Path, source_path: Path, work_dir: Path) -> BenchmarkRow:
    parsed = read_at_bathymetry_3d(source_path)
    roundtrip_path = work_dir / display_path
    write_at_bathymetry_3d(roundtrip_path, parsed)
    reparsed = read_at_bathymetry_3d(roundtrip_path)
    checks = {
        "interpolation": parsed.interpolation == reparsed.interpolation,
        "x": _array_matches(parsed.x_m, reparsed.x_m),
        "y": _array_matches(parsed.y_m, reparsed.y_m),
        "z": _array_matches(parsed.z_m, reparsed.z_m),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    array_match_count = sum(checks[field] for field in ("x", "y", "z"))
    return {
        "case": "bathymetry-3d-roundtrip-real-" + "-".join(display_path.with_suffix("").parts),
        "kind": "at_bathymetry_3d_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(checks["interpolation"]),
        "file_format_array_match_count": int(array_match_count),
        "file_format_unmatched_fields": ",".join(unmatched),
        "bathymetry_3d_interpolation": parsed.interpolation,
        "bathymetry_3d_x_count": int(parsed.x_m.size),
        "bathymetry_3d_y_count": int(parsed.y_m.size),
        "bathymetry_3d_z_count": int(parsed.z_m.size),
    }


def _bathymetry_3d_province_real_sample_roundtrip_row(
    display_path: Path, source_path: Path, work_dir: Path
) -> BenchmarkRow:
    parsed = read_at_bathymetry_3d_provinces(source_path)
    roundtrip_path = work_dir / display_path
    write_at_bathymetry_3d_provinces(roundtrip_path, parsed)
    reparsed = read_at_bathymetry_3d_provinces(roundtrip_path)
    checks = {
        "interpolation": parsed.interpolation == reparsed.interpolation,
        "province_property_kind": parsed.province_property_kind == reparsed.province_property_kind,
        "x": _array_matches(parsed.x_m, reparsed.x_m),
        "y": _array_matches(parsed.y_m, reparsed.y_m),
        "z": _array_matches(parsed.z_m, reparsed.z_m),
        "province_indices": np.array_equal(parsed.province_indices, reparsed.province_indices),
        "province_properties": _array_matches(parsed.province_properties, reparsed.province_properties),
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    array_match_count = sum(checks[field] for field in ("x", "y", "z", "province_indices", "province_properties"))
    return {
        "case": "bathymetry-3d-province-roundtrip-real-" + "-".join(display_path.with_suffix("").parts),
        "kind": "at_bathymetry_3d_province_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(checks["interpolation"]) + int(checks["province_property_kind"]),
        "file_format_array_match_count": int(array_match_count),
        "file_format_unmatched_fields": ",".join(unmatched),
        "bathymetry_3d_interpolation": parsed.interpolation,
        "bathymetry_3d_x_count": int(parsed.x_m.size),
        "bathymetry_3d_y_count": int(parsed.y_m.size),
        "bathymetry_3d_z_count": int(parsed.z_m.size),
        "bathymetry_3d_province_count": int(parsed.province_properties.shape[0]),
        "bathymetry_3d_province_property_kind": parsed.province_property_kind,
        "bathymetry_3d_province_property_column_count": int(parsed.province_properties.shape[1]),
        "bathymetry_3d_province_index_min": int(np.min(parsed.province_indices)),
        "bathymetry_3d_province_index_max": int(np.max(parsed.province_indices)),
        "bathymetry_3d_province_index_sample_count": int(parsed.province_indices.size),
    }


def _ssp3d_read_case_name(path: Path) -> str:
    return "ssp3d-read-real-" + "-".join(path.with_suffix("").parts)


def _ssp3d_real_sample_read_row(display_path: Path, source_path: Path) -> BenchmarkRow:
    x_arr, y_arr, z_arr, c_grid = read_at_ssp3d(source_path)
    finite = c_grid[np.isfinite(c_grid)]
    expected_shape = (x_arr.size, y_arr.size, z_arr.size)
    checks = {
        "x_count": x_arr.size >= 2,
        "y_count": y_arr.size >= 2,
        "z_count": z_arr.size >= 2,
        "shape": c_grid.shape == expected_shape,
        "finite": finite.size == c_grid.size,
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": _ssp3d_read_case_name(display_path),
        "kind": "at_ssp3d_read",
        "sample_count": len(checks),
        "file_format_read_ok": not unmatched,
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(checks["x_count"] + checks["y_count"] + checks["z_count"]),
        "file_format_array_match_count": int(checks["shape"] + checks["finite"]),
        "file_format_unmatched_fields": ",".join(unmatched),
        "ssp3d_x_count": int(x_arr.size),
        "ssp3d_y_count": int(y_arr.size),
        "ssp3d_z_count": int(z_arr.size),
        "ssp3d_sample_count": int(c_grid.size),
        "ssp3d_finite_count": int(finite.size),
        "ssp3d_c_min_mps": float(finite.min()) if finite.size else None,
        "ssp3d_c_max_mps": float(finite.max()) if finite.size else None,
    }


def _ssp2d_read_case_name(path: Path) -> str:
    return "ssp2d-read-real-" + "-".join(path.with_suffix("").parts)


def _ssp2d_real_sample_read_row(display_path: Path, source_path: Path) -> BenchmarkRow:
    parsed = read_at_ssp2d(source_path)
    finite = parsed.c_matrix[np.isfinite(parsed.c_matrix)]
    checks = {
        "range_count": parsed.ranges_m.size >= 1,
        "depth_row_count": parsed.c_matrix.shape[0] >= 1,
        "shape": parsed.c_matrix.ndim == 2 and parsed.c_matrix.shape[1] == parsed.ranges_m.size,
        "finite": finite.size == parsed.c_matrix.size,
    }
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": _ssp2d_read_case_name(display_path),
        "kind": "at_ssp2d_read",
        "sample_count": len(checks),
        "file_format_read_ok": not unmatched,
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(checks["range_count"] + checks["depth_row_count"]),
        "file_format_array_match_count": int(checks["shape"] + checks["finite"]),
        "file_format_unmatched_fields": ",".join(unmatched),
        "ssp2d_range_count": int(parsed.ranges_m.size),
        "ssp2d_depth_row_count": int(parsed.c_matrix.shape[0]),
        "ssp2d_sample_count": int(parsed.c_matrix.size),
        "ssp2d_finite_count": int(finite.size),
        "ssp2d_c_min_mps": float(finite.min()) if finite.size else None,
        "ssp2d_c_max_mps": float(finite.max()) if finite.size else None,
    }


def _ray_records_case(
    case_name: str,
    work_dir: Path,
    records: tuple[ATRayRecord, ...],
    *,
    file_format_path: Path | None = None,
) -> BenchmarkRow:
    path = work_dir / f"{case_name}.ray"
    write_at_ray_records(path, list(records), title=case_name, frequency=100.0, top_depth=0.0, bottom_depth=200.0)
    return _ray_records_roundtrip_row(case_name, path, records, file_format_path or path.name)


def _ray_records_roundtrip_row(
    case_name: str,
    path: Path,
    records: tuple[ATRayRecord, ...],
    file_format_path: Path,
) -> BenchmarkRow:
    parsed = tuple(read_at_ray_records(path))
    checks: dict[str, bool] = {
        "ray_count": len(parsed) == len(records),
    }
    for index, (expected, actual) in enumerate(zip(records, parsed)):
        checks[f"launch_angle_{index}"] = bool(np.isclose(actual.launch_angle_deg, expected.launch_angle_deg))
        checks[f"coordinate_system_{index}"] = actual.coordinate_system == expected.coordinate_system
        checks[f"top_bounces_{index}"] = actual.num_top_bounces == expected.num_top_bounces
        checks[f"bottom_bounces_{index}"] = actual.num_bottom_bounces == expected.num_bottom_bounces
        checks[f"ranges_{index}"] = _array_matches(expected.ranges_m, actual.ranges_m)
        checks[f"depths_{index}"] = _array_matches(expected.depths_m, actual.depths_m)
        if expected.coordinate_system == "xyz":
            checks[f"y_{index}"] = _array_matches(expected.y_m, actual.y_m)
            checks[f"launch_azimuth_{index}"] = bool(
                np.isclose(actual.launch_azimuth_deg, expected.launch_azimuth_deg)
            )
    unmatched = [field for field, matched in checks.items() if not matched]
    coordinate_systems = {record.coordinate_system for record in parsed}
    point_count = int(sum(record.ranges_m.size for record in parsed))
    max_top_bounces = max((record.num_top_bounces for record in parsed), default=0)
    max_bottom_bounces = max((record.num_bottom_bounces for record in parsed), default=0)
    launch_angles = np.asarray([record.launch_angle_deg for record in parsed], dtype=np.float64)
    launch_azimuths = np.asarray(
        [record.launch_azimuth_deg for record in parsed if record.launch_azimuth_deg is not None],
        dtype=np.float64,
    )
    range_arrays = [record.ranges_m for record in parsed if record.ranges_m.size]
    depth_arrays = [record.depths_m for record in parsed if record.depths_m.size]
    y_arrays = [record.y_m for record in parsed if record.y_m is not None and record.y_m.size]
    range_values = np.concatenate(range_arrays) if range_arrays else np.array([])
    depth_values = np.concatenate(depth_arrays) if depth_arrays else np.array([])
    y_values = np.concatenate(y_arrays) if y_arrays else np.array([])
    return {
        "case": case_name,
        "kind": "at_ray_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(file_format_path),
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(
            sum(matched for key, matched in checks.items() if key.startswith(("ranges_", "depths_", "y_")))
        ),
        "file_format_unmatched_fields": ",".join(unmatched),
        "ray_record_count": len(parsed),
        "ray_point_count": point_count,
        "ray_coordinate_system": ",".join(sorted(coordinate_systems)),
        "ray_max_top_bounces": int(max_top_bounces),
        "ray_max_bottom_bounces": int(max_bottom_bounces),
        "ray_launch_angle_min_deg": float(launch_angles.min()) if launch_angles.size else None,
        "ray_launch_angle_max_deg": float(launch_angles.max()) if launch_angles.size else None,
        "ray_launch_azimuth_min_deg": float(launch_azimuths.min()) if launch_azimuths.size else None,
        "ray_launch_azimuth_max_deg": float(launch_azimuths.max()) if launch_azimuths.size else None,
        "ray_range_min_m": float(range_values.min()) if range_values.size else None,
        "ray_range_max_m": float(range_values.max()) if range_values.size else None,
        "ray_depth_min_m": float(depth_values.min()) if depth_values.size else None,
        "ray_depth_max_m": float(depth_values.max()) if depth_values.size else None,
        "ray_y_min_m": float(y_values.min()) if y_values.size else None,
        "ray_y_max_m": float(y_values.max()) if y_values.size else None,
    }


def _ray_record_cases(work_dir: Path) -> tuple[BenchmarkRow, BenchmarkRow]:
    return (
        _ray_records_case(
            "ray-roundtrip-2d-rz-bounce-metadata",
            work_dir,
            (
                ATRayRecord(
                    launch_angle_deg=-3.5,
                    ranges_m=np.array([0.0, 100.0, 250.0], dtype=np.float64),
                    depths_m=np.array([20.0, 35.0, 50.0], dtype=np.float64),
                    amplitudes=np.ones(3, dtype=np.float64),
                    num_top_bounces=1,
                    num_bottom_bounces=0,
                ),
                ATRayRecord(
                    launch_angle_deg=4.0,
                    ranges_m=np.array([0.0, 125.0], dtype=np.float64),
                    depths_m=np.array([20.0, 10.0], dtype=np.float64),
                    amplitudes=np.ones(2, dtype=np.float64),
                    num_top_bounces=0,
                    num_bottom_bounces=1,
                ),
            ),
        ),
        _ray_records_case(
            "ray-roundtrip-3d-xyz-azimuth",
            work_dir,
            (
                ATRayRecord(
                    launch_angle_deg=-3.5,
                    launch_azimuth_deg=25.0,
                    coordinate_system="xyz",
                    ranges_m=np.array([0.0, 100.0, 250.0], dtype=np.float64),
                    y_m=np.array([0.0, 5.0, 15.0], dtype=np.float64),
                    depths_m=np.array([20.0, 35.0, 50.0], dtype=np.float64),
                    amplitudes=np.ones(3, dtype=np.float64),
                    num_top_bounces=1,
                    num_bottom_bounces=0,
                ),
            ),
        ),
    )


def _ray_real_sample_roundtrip_row(display_path: Path, source_path: Path, work_dir: Path) -> BenchmarkRow:
    records = tuple(read_at_ray_records(source_path))
    roundtrip_path = work_dir / display_path
    roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
    max_depth = max((float(record.depths_m.max()) for record in records if record.depths_m.size), default=200.0)
    write_at_ray_records(
        roundtrip_path,
        records,
        title=display_path.stem,
        frequency=100.0,
        top_depth=0.0,
        bottom_depth=max_depth + 1.0,
    )
    case_name = "ray-roundtrip-real-" + "-".join(display_path.with_suffix("").parts)
    return _ray_records_roundtrip_row(case_name, roundtrip_path, records, display_path)


def _empty_arrival_record(is_3d: bool) -> ArrivalRecord:
    empty_float = np.zeros(0, dtype=np.float64)
    empty_complex = np.zeros(0, dtype=np.complex128)
    return ArrivalRecord(
        amplitudes=empty_complex,
        delays=empty_complex,
        src_decl_angles=empty_float,
        rcvr_decl_angles=empty_float,
        src_azim_angles=empty_float if is_3d else None,
        rcvr_azim_angles=empty_float if is_3d else None,
        phase_angles=empty_float,
        num_top_bounces=np.zeros(0, dtype=np.int16),
        num_bottom_bounces=np.zeros(0, dtype=np.int16),
    )


def _record_close(expected: ArrivalRecord, actual: ArrivalRecord) -> dict[str, bool]:
    checks = {
        "count": actual.count == expected.count,
        "amplitudes": _array_matches(expected.amplitudes, actual.amplitudes),
        "delays": _array_matches(expected.delays, actual.delays),
        "src_decl_angles": _array_matches(expected.src_decl_angles, actual.src_decl_angles),
        "rcvr_decl_angles": _array_matches(expected.rcvr_decl_angles, actual.rcvr_decl_angles),
        "num_top_bounces": _array_matches(expected.num_top_bounces, actual.num_top_bounces),
        "num_bottom_bounces": _array_matches(expected.num_bottom_bounces, actual.num_bottom_bounces),
    }
    if expected.src_azim_angles is not None:
        checks["src_azim_angles"] = _array_matches(expected.src_azim_angles, actual.src_azim_angles)
    if expected.rcvr_azim_angles is not None:
        checks["rcvr_azim_angles"] = _array_matches(expected.rcvr_azim_angles, actual.rcvr_azim_angles)
    return checks


def _arr_roundtrip_row(case_name: str, work_dir: Path, arrivals: ArrivalData) -> BenchmarkRow:
    path = work_dir / f"{case_name}.arr"
    write_at_arr(path, arrivals)
    parsed = read_at_arr(path)
    checks: dict[str, bool] = {
        "is_3d": parsed.is_3d == arrivals.is_3d,
        "frequency": bool(np.isclose(parsed.frequency, arrivals.frequency)),
        "source_depths": _array_matches(arrivals.source_depths, parsed.source_depths),
        "receiver_depths": _array_matches(arrivals.receiver_depths, parsed.receiver_depths),
        "receiver_ranges": _array_matches(arrivals.receiver_ranges, parsed.receiver_ranges),
        "source_x": _array_matches(arrivals.source_x, parsed.source_x),
        "source_y": _array_matches(arrivals.source_y, parsed.source_y),
        "receiver_bearings": _array_matches(arrivals.receiver_bearings, parsed.receiver_bearings),
        "arrival_shape": parsed.arrivals.shape == arrivals.arrivals.shape,
    }
    if checks["arrival_shape"]:
        for index, expected in enumerate(arrivals.arrivals.flat):
            actual = parsed.arrivals.flat[index]
            for name, matched in _record_close(expected, actual).items():
                checks[f"record_{index}_{name}"] = matched
    unmatched = [field for field, matched in checks.items() if not matched]
    records = [record for record in parsed.arrivals.flat if record is not None]
    arrival_count = int(sum(record.count for record in records))
    top_bounces = [int(np.max(record.num_top_bounces)) for record in records if record.count]
    bottom_bounces = [int(np.max(record.num_bottom_bounces)) for record in records if record.count]
    return {
        "case": case_name,
        "kind": "at_arr_roundtrip",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": path.name,
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": int(
            sum(
                matched
                for key, matched in checks.items()
                if key.endswith(
                    (
                        "depths",
                        "ranges",
                        "source_x",
                        "source_y",
                        "receiver_bearings",
                        "amplitudes",
                        "delays",
                        "angles",
                        "bounces",
                    )
                )
            )
        ),
        "file_format_unmatched_fields": ",".join(unmatched),
        "arr_is_3d": parsed.is_3d,
        "arr_record_count": len(records),
        "arr_arrival_count": arrival_count,
        "arr_source_depth_count": int(parsed.source_depths.size),
        "arr_receiver_depth_count": int(parsed.receiver_depths.size),
        "arr_receiver_range_count": int(parsed.receiver_ranges.size),
        "arr_receiver_bearing_count": int(0 if parsed.receiver_bearings is None else parsed.receiver_bearings.size),
        "arr_source_x_count": int(0 if parsed.source_x is None else parsed.source_x.size),
        "arr_source_y_count": int(0 if parsed.source_y is None else parsed.source_y.size),
        "arr_max_top_bounces": max(top_bounces, default=0),
        "arr_max_bottom_bounces": max(bottom_bounces, default=0),
    }


def _arrival_record_cases(work_dir: Path) -> tuple[BenchmarkRow, BenchmarkRow]:
    arrivals_2d = np.empty((1, 2, 2), dtype=object)
    arrivals_2d[0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([1.0 + 0.0j, 0.5j], dtype=np.complex128),
        delays=np.array([0.01 + 0.0j, 0.014 + 0.0001j], dtype=np.complex128),
        src_decl_angles=np.array([-5.0, 5.0], dtype=np.float64),
        rcvr_decl_angles=np.array([4.5, -4.5], dtype=np.float64),
        phase_angles=np.array([0.0, np.pi / 2.0], dtype=np.float64),
        num_top_bounces=np.array([0, 1], dtype=np.int16),
        num_bottom_bounces=np.array([0, 1], dtype=np.int16),
    )
    arrivals_2d[0, 1, 0] = _empty_arrival_record(False)
    arrivals_2d[0, 0, 1] = _empty_arrival_record(False)
    arrivals_2d[0, 1, 1] = ArrivalRecord(
        amplitudes=np.array([0.25 * np.exp(1j * np.deg2rad(-30.0))], dtype=np.complex128),
        delays=np.array([0.02 + 0.0j], dtype=np.complex128),
        src_decl_angles=np.array([8.0], dtype=np.float64),
        rcvr_decl_angles=np.array([-7.5], dtype=np.float64),
        phase_angles=np.array([np.deg2rad(-30.0)], dtype=np.float64),
        num_top_bounces=np.array([1], dtype=np.int16),
        num_bottom_bounces=np.array([2], dtype=np.int16),
    )
    data_2d = ArrivalData(
        is_3d=False,
        frequency=100.0,
        source_depths=np.array([25.0, 75.0], dtype=np.float64),
        receiver_depths=np.array([50.0, 100.0], dtype=np.float64),
        receiver_ranges=np.array([1.0], dtype=np.float64),
        arrivals=arrivals_2d,
    )

    arrivals_3d = np.empty((1, 1, 2, 1, 2, 1), dtype=object)
    arrivals_3d.fill(_empty_arrival_record(True))
    arrivals_3d[0, 0, 0, 0, 0, 0] = ArrivalRecord(
        amplitudes=np.array([0.125 * np.exp(1j * np.deg2rad(20.0))], dtype=np.complex128),
        delays=np.array([0.012 + 0.0j], dtype=np.complex128),
        src_decl_angles=np.array([2.0], dtype=np.float64),
        src_azim_angles=np.array([15.0], dtype=np.float64),
        rcvr_decl_angles=np.array([-1.0], dtype=np.float64),
        rcvr_azim_angles=np.array([5.0], dtype=np.float64),
        phase_angles=np.array([np.deg2rad(20.0)], dtype=np.float64),
        num_top_bounces=np.array([0], dtype=np.int16),
        num_bottom_bounces=np.array([1], dtype=np.int16),
    )
    arrivals_3d[0, 0, 1, 0, 1, 0] = ArrivalRecord(
        amplitudes=np.array([0.25 + 0.0j, 0.1j], dtype=np.complex128),
        delays=np.array([0.02 + 0.0j, 0.025 + 0.0002j], dtype=np.complex128),
        src_decl_angles=np.array([-3.0, 3.0], dtype=np.float64),
        src_azim_angles=np.array([35.0, 40.0], dtype=np.float64),
        rcvr_decl_angles=np.array([2.5, -2.5], dtype=np.float64),
        rcvr_azim_angles=np.array([45.0, 50.0], dtype=np.float64),
        phase_angles=np.array([0.0, np.pi / 2.0], dtype=np.float64),
        num_top_bounces=np.array([1, 1], dtype=np.int16),
        num_bottom_bounces=np.array([1, 2], dtype=np.int16),
    )
    data_3d = ArrivalData(
        is_3d=True,
        frequency=200.0,
        source_depths=np.array([40.0], dtype=np.float64),
        receiver_depths=np.array([80.0], dtype=np.float64),
        receiver_ranges=np.array([1000.0], dtype=np.float64),
        receiver_bearings=np.array([0.0, 30.0], dtype=np.float64),
        source_x=np.array([0.0, 250.0], dtype=np.float64),
        source_y=np.array([0.0], dtype=np.float64),
        arrivals=arrivals_3d,
    )
    return (
        _arr_roundtrip_row("arr-roundtrip-2d-multi-source-depth", work_dir, data_2d),
        _arr_roundtrip_row("arr-roundtrip-3d-source-xy-bearing", work_dir, data_3d),
    )


def _prt_row_fields(parsed) -> BenchmarkRow:
    return {
        "prt_program": parsed.program,
        "prt_title": parsed.title,
        "prt_case_title": parsed.case_title,
        "prt_completed_successfully": parsed.completed_successfully,
        "prt_completed_without_errors": parsed.completed_without_errors,
        "prt_warning_count": parsed.warning_count,
        "prt_error_count": parsed.error_count,
        "prt_fatal_error_count": parsed.fatal_error_count,
        "prt_issue_count": parsed.warning_count + parsed.error_count + parsed.fatal_error_count,
        "prt_parameter_count": None if parsed.parameters is None else len(parsed.parameters),
        "prt_frequency_hz": parsed.frequency_hz,
        "prt_cpu_time_s": parsed.cpu_time_s,
        "prt_nmedia": parsed.nmedia,
        "prt_bottom_depth_m": parsed.bottom_depth_m,
        "prt_rms_roughness_m": parsed.rms_roughness_m,
        "prt_c_low_mps": parsed.c_low_mps,
        "prt_c_high_mps": parsed.c_high_mps,
        "prt_rmax_km": parsed.rmax_km,
        "prt_attenuation_units": parsed.attenuation_units,
        "prt_run_type_description": parsed.run_type_description,
        "prt_beam_description": parsed.beam_description,
        "prt_source_description": parsed.source_description,
        "prt_receiver_grid_description": parsed.receiver_grid_description,
        "prt_source_beam_description": parsed.source_beam_description,
        "prt_mode_addition_description": parsed.mode_addition_description,
        "prt_calculation_description": parsed.calculation_description,
        "prt_beam_shift_description": parsed.beam_shift_description,
        "prt_mode_count": parsed.mode_count,
        "prt_mesh_point_count": parsed.mesh_point_count,
        "prt_max_root_finder_restarts": parsed.max_root_finder_restarts,
        "prt_max_arrivals": parsed.max_arrivals,
        "prt_receiver_range_count": parsed.receiver_range_count,
        "prt_receiver_depth_count": parsed.receiver_depth_count,
        "prt_receiver_theta_count": parsed.receiver_theta_count,
        "prt_receiver_range_min_km": parsed.receiver_range_min_km,
        "prt_receiver_range_max_km": parsed.receiver_range_max_km,
        "prt_receiver_depth_min_m": parsed.receiver_depth_min_m,
        "prt_receiver_depth_max_m": parsed.receiver_depth_max_m,
        "prt_receiver_theta_min_deg": parsed.receiver_theta_min_deg,
        "prt_receiver_theta_max_deg": parsed.receiver_theta_max_deg,
        "prt_source_x_count": parsed.source_x_count,
        "prt_source_y_count": parsed.source_y_count,
        "prt_source_depth_count": parsed.source_depth_count,
        "prt_source_x_min_km": parsed.source_x_min_km,
        "prt_source_x_max_km": parsed.source_x_max_km,
        "prt_source_y_min_km": parsed.source_y_min_km,
        "prt_source_y_max_km": parsed.source_y_max_km,
        "prt_source_depth_min_m": parsed.source_depth_min_m,
        "prt_source_depth_max_m": parsed.source_depth_max_m,
        "prt_source_coordinate_x_m": parsed.source_coordinate_x_m,
        "prt_source_coordinate_y_m": parsed.source_coordinate_y_m,
        "prt_source_coordinate_z_m": parsed.source_coordinate_z_m,
        "prt_profile_range_count": parsed.profile_range_count,
        "prt_profile_range_min_km": parsed.profile_range_min_km,
        "prt_profile_range_max_km": parsed.profile_range_max_km,
        "prt_receiver_range_offset_count": parsed.receiver_range_offset_count,
        "prt_receiver_range_offset_min_m": parsed.receiver_range_offset_min_m,
        "prt_receiver_range_offset_max_m": parsed.receiver_range_offset_max_m,
        "prt_field_option": parsed.field_option,
        "prt_field_alpha_count": parsed.field_alpha_count,
        "prt_field_alpha_min_deg": parsed.field_alpha_min_deg,
        "prt_field_alpha_max_deg": parsed.field_alpha_max_deg,
        "prt_field_step_m": parsed.field_step_m,
        "prt_field_step_count": parsed.field_step_count,
        "prt_node_count": parsed.node_count,
        "prt_element_count": parsed.element_count,
        "prt_beam_count": parsed.beam_count,
        "prt_beam_bearing_count": parsed.beam_bearing_count,
        "prt_beam_angle_min_deg": parsed.beam_angle_min_deg,
        "prt_beam_angle_max_deg": parsed.beam_angle_max_deg,
        "prt_beam_bearing_angle_min_deg": parsed.beam_bearing_angle_min_deg,
        "prt_beam_bearing_angle_max_deg": parsed.beam_bearing_angle_max_deg,
        "prt_beam_halfwidth1": parsed.beam_halfwidth1,
        "prt_beam_halfwidth2": parsed.beam_halfwidth2,
        "prt_eps_multiplier": parsed.eps_multiplier,
        "prt_step_length_m": parsed.step_length_m,
        "prt_box_x_m": parsed.box_x_m,
        "prt_box_y_m": parsed.box_y_m,
        "prt_box_depth_m": parsed.box_depth_m,
        "prt_box_range_km": parsed.box_range_km,
        "prt_bathymetry_x_count": parsed.bathymetry_x_count,
        "prt_bathymetry_y_count": parsed.bathymetry_y_count,
        "prt_bathymetry_x_min_km": parsed.bathymetry_x_min_km,
        "prt_bathymetry_x_max_km": parsed.bathymetry_x_max_km,
        "prt_bathymetry_y_min_km": parsed.bathymetry_y_min_km,
        "prt_bathymetry_y_max_km": parsed.bathymetry_y_max_km,
        "prt_mode_wavenumber_min_1pm": parsed.mode_wavenumber_min_1pm,
        "prt_mode_wavenumber_max_1pm": parsed.mode_wavenumber_max_1pm,
        "prt_mode_attenuation_min_1pm": parsed.mode_attenuation_min_1pm,
        "prt_mode_attenuation_max_1pm": parsed.mode_attenuation_max_1pm,
        "prt_mode_phase_speed_min_mps": parsed.mode_phase_speed_min_mps,
        "prt_mode_phase_speed_max_mps": parsed.mode_phase_speed_max_mps,
        "prt_mode_group_speed_min_mps": parsed.mode_group_speed_min_mps,
        "prt_mode_group_speed_max_mps": parsed.mode_group_speed_max_mps,
        "prt_mesh_multiplier_cpu_count": (
            None if parsed.mesh_multiplier_cpu_s is None else len(parsed.mesh_multiplier_cpu_s)
        ),
    }


def _prt_parse_row(case_name: str, work_dir: Path, text: str, expected: dict[str, object]) -> BenchmarkRow:
    path = work_dir / f"{case_name}.prt"
    path.write_text(text, encoding="utf-8")
    parsed = read_at_prt(path)
    checks: dict[str, bool] = {}
    for field, expected_value in expected.items():
        actual = getattr(parsed, field)
        if isinstance(expected_value, float):
            checks[field] = bool(np.isclose(actual, expected_value))
        else:
            checks[field] = actual == expected_value
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": case_name,
        "kind": "at_prt_parse",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": path.name,
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": 0,
        "file_format_unmatched_fields": ",".join(unmatched),
        **_prt_row_fields(parsed),
    }


def _prt_parse_cases(work_dir: Path) -> tuple[BenchmarkRow, BenchmarkRow, BenchmarkRow]:
    bellhop_text = (
        "BELLHOP/BELLHOP3D\n"
        "BELLHOP- canonical prt parse\n"
        " frequency =   50.00     Hz\n"
        " Dummy parameter NMedia =            1\n"
        " Depth =    5000.00 m\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Number of Receiver z-coordinates, Rz =          501\n"
        "    Receiver z-coordinates, Rz (m)\n"
        "       0.0  10.0\n"
        "       ...\n"
        "       5000.0\n"
        "    Number of Receiver r-coordinates, Rr =         1001\n"
        "    Receiver r-coordinates, Rr (km)\n"
        "       0.0  0.1\n"
        "       ...\n"
        "       100.0\n"
        " Coherent TL calculation\n"
        " Geometric hat beams in Cartesian coordinates\n"
        " Point source (cylindrical coordinates)\n"
        " Rectilinear receiver grid: Receivers at Rr( : ) x Rz( : )\n"
        "    Number of beams in elevation   =         1000\n"
        "    Beam take-off angles (degrees)\n"
        "      -20.0 -19.96\n"
        "       ...\n"
        "       20.0\n"
        " Step length,       deltas =   50.00     m\n"
        "   Box%z  =   5500.     m\n"
        "   Box%r  =   101.0    km\n"
        "( Maximum # of arrivals = 10 )\n"
        " CPU Time =       0.545    s\n"
        "BELLHOP completed successfully\n"
    )
    field3d_text = (
        "Running FIELD3D\n"
        "Sums modes, producing pressure\n"
        "Option = STDFM\n"
        "Alpha scan =     0.5000       360.000          720\n"
        "Step =       250.000 m    NSTEPS =          1100\n"
        "Number of modes =           60\n"
        "    Number of Source   x-coordinates, Sx =            2\n"
        "    Source   x-coordinates, Sx (km)\n"
        "                 0.00000      1.00000\n"
        "    Number of Source   y-coordinates, Sy =            1\n"
        "    Source   y-coordinates, Sy (km)\n"
        "                 0.50000\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Source   z-coordinates, Sz (m)\n"
        "                 400.000\n"
        "    Number of Receiver z-coordinates, Rz =            1\n"
        "    Receiver z-coordinates, Rz (m)\n"
        "                 400.000\n"
        "    Number of Receiver r-coordinates, Rr =         1001\n"
        "    Receiver r-coordinates, Rr (km)\n"
        "                 0.00000      0.275000\n"
        "          ...\n"
        "   275.000\n"
        "    Number of Receiver theta-coordinates, theta =          181\n"
        "    Receiver theta-coordinates, theta (degrees)\n"
        "                 0.00000       2.00000\n"
        "          ...\n"
        "   360.000\n"
        " NNodes =          397\n"
        " NElts  =          727\n"
        "CPU time =      0.167    s\n"
        "Field3D completed successfully\n"
    )
    kraken_text = (
        "KRAKENC-canonical prt parse\n"
        "Nominal Frequency =   250.0    Hz\n"
        "NMedia    =   1\n"
        "Attenuation units: dB/wavelength\n"
        "cLow =   1450.0     m/s      cHigh =   1650.0     m/s\n"
        "RMax =   0.0000     km\n"
        "    Number of Source   z-coordinates, Sz =            1\n"
        "    Number of Receiver z-coordinates, Rz =         1001\n"
        "Mesh multiplier   CPU seconds\n"
        "Max. number of restarts in root finder, MaxTries =            1\n"
        "       1            0.545E-03s\n"
        "    I       k             alpha     Phase Speed     Group Speed\n"
        "          (1/m)           (1/m)        (m/s)           (m/s)\n"
        "    1   1.046756917     -0.21E-05   1500.631428       1499.41\n"
        "    2   1.045433990     -0.86E-05   1502.530377       1497.64\n"
        "    3   1.043225728     -0.19E-04   1505.710878       1494.69\n"
        "End of environmental file\n"
    )
    return (
        _prt_parse_row(
            "prt-parse-bellhop-grid-beam",
            work_dir,
            bellhop_text,
            {
                "program": "BELLHOP/BELLHOP3D",
                "case_title": "BELLHOP- canonical prt parse",
                "frequency_hz": 50.0,
                "receiver_depth_count": 501,
                "receiver_range_count": 1001,
                "beam_count": 1000,
                "completed_without_errors": True,
            },
        ),
        _prt_parse_row(
            "prt-parse-field3d-mesh",
            work_dir,
            field3d_text,
            {
                "program": "Running FIELD3D",
                "field_option": "STDFM",
                "field_alpha_count": 720,
                "field_step_count": 1100,
                "mode_count": 60,
                "receiver_theta_count": 181,
                "node_count": 397,
                "element_count": 727,
                "completed_without_errors": True,
            },
        ),
        _prt_parse_row(
            "prt-parse-krakenc-mode-table",
            work_dir,
            kraken_text,
            {
                "program": "KRAKENC-canonical prt parse",
                "frequency_hz": 250.0,
                "nmedia": 1,
                "receiver_depth_count": 1001,
                "mode_count": 3,
                "mesh_point_count": None,
                "completed_without_errors": True,
            },
        ),
    )


def _prt_real_sample_parse_row(display_path: Path, source_path: Path, expected: dict[str, object]) -> BenchmarkRow:
    parsed = read_at_prt(source_path)
    checks: dict[str, bool] = {}
    for field, expected_value in expected.items():
        actual = getattr(parsed, field)
        if isinstance(expected_value, float):
            checks[field] = bool(np.isclose(actual, expected_value))
        else:
            checks[field] = actual == expected_value
    unmatched = [field for field, matched in checks.items() if not matched]
    return {
        "case": "prt-parse-real-" + "-".join(display_path.with_suffix("").parts),
        "kind": "at_prt_parse",
        "sample_count": len(checks),
        "file_format_roundtrip_ok": not unmatched,
        "file_format_path": str(display_path),
        "file_format_scalar_match_count": int(sum(checks.values())),
        "file_format_array_match_count": 0,
        "file_format_unmatched_fields": ",".join(unmatched),
        **_prt_row_fields(parsed),
    }


def run_at_file_format_benchmark_matrix(
    *,
    sample_root: str | Path | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> ATFileFormatBenchmarkMatrix:
    """Run canonical AT file-format round-trip checks."""

    root = Path(sample_root).resolve() if sample_root is not None else _project_root()
    rows: list[BenchmarkRow] = []
    with TemporaryDirectory(prefix="pyacoustics-at-file-format-benchmark-") as temp_dir:
        work_dir = Path(temp_dir)
        rows.extend(_env_summary_cases(work_dir))
        seen_env_paths: set[Path] = set()

        def append_env_summary_row(source_path: Path, display_root: Path) -> None:
            resolved = source_path.resolve()
            if resolved in seen_env_paths:
                return
            seen_env_paths.add(resolved)
            rows.append(
                _env_summary_real_sample_row(
                    source_path.relative_to(display_root),
                    source_path,
                )
            )

        at_tests_root = _resolve_at_tests_root(root)
        if at_tests_root is not None:
            for source_path in sorted(at_tests_root.rglob("*.env")):
                append_env_summary_row(source_path, at_tests_root.parent)
        at_matlab_root = _resolve_at_matlab_root(root)
        if at_matlab_root is not None:
            for source_path in sorted(at_matlab_root.rglob("*.env")):
                append_env_summary_row(source_path, at_matlab_root.parent)
        at_source_root = _resolve_at_source_root(root)
        if at_source_root is not None:
            for source_path in sorted(at_source_root.rglob("*.env")):
                append_env_summary_row(source_path, at_source_root)
        for relative_path in _OPTIONAL_REAL_RAM_IN_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_ram_in_summary_real_sample_row(relative_path, source_path))
        if at_tests_root is not None:
            for source_path in sorted(at_tests_root.rglob("*.m")):
                rows.append(
                    _matlab_script_inventory_real_sample_row(
                        source_path.relative_to(at_tests_root.parent),
                        source_path,
                    )
                )
        at_matlab_root = _resolve_at_matlab_root(root)
        if at_matlab_root is not None:
            for source_path in sorted(at_matlab_root.rglob("*.m")):
                rows.append(
                    _matlab_script_inventory_real_sample_row(
                        source_path.relative_to(at_matlab_root.parent),
                        source_path,
                    )
                )
        at_source_root = _resolve_at_source_root(root)
        if at_source_root is not None:
            for source_path in sorted(at_source_root.rglob("*.f90")):
                rows.append(
                    _fortran_source_inventory_real_sample_row(
                        source_path.relative_to(at_source_root),
                        source_path,
                    )
                )
        for relative_path in _CANONICAL_FLP_SAMPLES:
            source_path = root / relative_path
            if not source_path.exists():
                rows.append(
                    {
                        "case": _flp_case_name(relative_path),
                        "kind": "at_field_flp_roundtrip",
                        "passed": False,
                        "skipped": True,
                        "failure_count": 1,
                        "failures": f"AT .flp sample not found: {source_path}",
                    }
                )
                continue
            parsed = read_at_field_flp(source_path)
            roundtrip_path = work_dir / relative_path
            roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
            write_at_field_flp(parsed, roundtrip_path)
            reparsed = read_at_field_flp(roundtrip_path)
            rows.append(_field_flp_roundtrip_row(relative_path, parsed, reparsed))
        for relative_path in _OPTIONAL_REAL_FLP_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                parsed = read_at_field_flp(source_path)
                roundtrip_path = work_dir / relative_path
                roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
                write_at_field_flp(parsed, roundtrip_path)
                reparsed = read_at_field_flp(roundtrip_path)
                rows.append(_field_flp_roundtrip_row(relative_path, parsed, reparsed))
        rows.append(_mod_metadata_multi_medium_case(work_dir))
        rows.append(_mod_metadata_complex_halfspace_case(work_dir))
        for relative_path in _CANONICAL_MOD_SAMPLES:
            source_path = root / relative_path
            if not source_path.exists():
                rows.append(
                    {
                        "case": "mod-roundtrip-real-" + "-".join(relative_path.with_suffix("").parts),
                        "kind": "at_mod_roundtrip",
                        "passed": False,
                        "skipped": True,
                        "failure_count": 1,
                        "failures": f"AT .mod sample not found: {source_path}",
                    }
                )
                continue
            rows.append(_mod_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        for relative_path in _OPTIONAL_REAL_MOD_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_mod_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        for relative_path in _CANONICAL_SHD_SAMPLES:
            source_path = root / relative_path
            if not source_path.exists():
                continue
            rows.append(_shd_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        for relative_path in _OPTIONAL_REAL_SHD_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_shd_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        rows.append(_shd_multifrequency_source_grid_case(work_dir))
        rows.append(_shd_tl_compressed_source_xy_case(work_dir))
        rows.extend(_boundary_profile_cases(work_dir))
        for relative_path in _OPTIONAL_REAL_BOUNDARY_PROFILE_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_boundary_profile_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        for relative_path in _CANONICAL_BATHYMETRY_3D_SAMPLES:
            source_path = root / relative_path
            if not source_path.exists():
                continue
            rows.append(_bathymetry_3d_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        for relative_path in _OPTIONAL_REAL_BATHYMETRY_3D_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_bathymetry_3d_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        for relative_path in _OPTIONAL_REAL_BATHYMETRY_3D_PROVINCE_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_bathymetry_3d_province_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        for relative_path in _CANONICAL_SSP3D_SAMPLES:
            source_path = root / relative_path
            if not source_path.exists():
                rows.append(
                    {
                        "case": _ssp3d_read_case_name(relative_path),
                        "kind": "at_ssp3d_read",
                        "passed": False,
                        "skipped": True,
                        "failure_count": 1,
                        "failures": f"AT 3D .ssp sample not found: {source_path}",
                    }
                )
                continue
            rows.append(_ssp3d_real_sample_read_row(relative_path, source_path))
        for relative_path in _OPTIONAL_REAL_SSP3D_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_ssp3d_real_sample_read_row(relative_path, source_path))
        for relative_path in _OPTIONAL_REAL_SSP2D_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_ssp2d_real_sample_read_row(relative_path, source_path))
        rows.extend(_reflection_table_cases(work_dir))
        for relative_path in _OPTIONAL_REAL_REFLECTION_TABLE_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_reflection_table_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        rows.append(_source_beam_pattern_case(work_dir))
        for relative_path in _OPTIONAL_REAL_SOURCE_BEAM_PATTERN_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_source_beam_pattern_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        rows.extend(_ray_record_cases(work_dir))
        for relative_path in _OPTIONAL_REAL_RAY_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_ray_real_sample_roundtrip_row(relative_path, source_path, work_dir))
        rows.extend(_arrival_record_cases(work_dir))
        rows.extend(_prt_parse_cases(work_dir))
        for relative_path, expected in _CANONICAL_PRT_SAMPLES:
            source_path = root / relative_path
            if not source_path.exists():
                continue
            rows.append(_prt_real_sample_parse_row(relative_path, source_path, expected))
        for relative_path, expected in _OPTIONAL_REAL_PRT_SAMPLES:
            source_path = _resolve_optional_sample(root, relative_path)
            if source_path is not None:
                rows.append(_prt_real_sample_parse_row(relative_path, source_path, expected))
    matrix_rows = evaluate_benchmark_rows(rows, thresholds) if thresholds is not None else tuple(rows)
    return ATFileFormatBenchmarkMatrix(rows=tuple(matrix_rows))


__all__ = [
    "ATFileFormatBenchmarkMatrix",
    "at_field_flp_smoke_thresholds",
    "at_file_format_smoke_thresholds",
    "run_at_file_format_benchmark_matrix",
]
