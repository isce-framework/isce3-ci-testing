#!/usr/bin/env python3
from __future__ import annotations
from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import reduce
import h5py
from itertools import chain
import json
import logging
import math
import os
from nisar.antenna import AntennaPattern, get_calib_range_line_idx
from nisar.noise.noise_estimation_from_raw import (
    est_noise_power_in_focus, NoiseEquivalentBackscatterProduct)
from nisar.mixed_mode import (PolChannel, PolChannelSet, Band,
    find_overlapping_channel)
from nisar.products.readers.antenna import AntennaParser
from nisar.products.readers.instrument import InstrumentParser
from nisar.products.readers.Raw import Raw, open_rrsd
from nisar.products.readers.rslc_cal import (RslcCalibration,
    parse_rslc_calibration, get_scale_and_delay, check_cal_validity_dates)
from nisar.products.writers import SLC
from nisar.products.writers.SLC import fill_partial_granule_id
from isce3.core.types import (complex32, to_complex32, read_c4_dataset_as_c8,
    truncate_mantissa)
import nisar
import numpy as np
import isce3
from isce3.core import DateTime, TimeDelta, LUT2d, Attitude, Orbit
from isce3.focus import make_los_luts, fill_gaps, make_cal_luts, Notch
from isce3.geometry import los2doppler
from isce3.io.gdal import Raster, GDT_CFloat32
from isce3.product import RadarGridParameters
from nisar.workflows.yaml_argparse import YamlArgparse
import nisar.workflows.helpers as helpers
from ruamel.yaml import YAML
import shapely
import sys
import tempfile
from typing import Union, Optional, Callable, Iterable, overload
from isce3.io import Raster as RasterIO
from io import StringIO


# TODO some CSV logger
log = logging.getLogger("focus")

# https://stackoverflow.com/a/6993694/112699
class Struct(object):
    "Convert nested dict to object, assuming keys are valid identifiers."
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


def load_config(yaml):
    "Load default runconfig, override with user input, and convert to Struct"
    parser = YAML(typ='safe')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg = parser.load(open(f'{dir_path}/defaults/focus.yaml', 'r'))
    with open(yaml) as f:
        user = parser.load(f)
    helpers.deep_update(cfg, user, flag_none_is_valid=False)
    return Struct(cfg)


def dump_config(cfg: Struct, stream):
    def struct2dict(s: Struct):
        d = s.__dict__.copy()
        for k in d:
            if isinstance(d[k], Struct):
                d[k] = struct2dict(d[k])
            elif isinstance(d[k], list):
                d[k] = [struct2dict(v) if isinstance(v, Struct) else v for v in d[k]]
        return d
    parser = YAML()
    parser.indent = 4
    d = struct2dict(cfg)
    parser.dump(d, stream)


def dump_config_str(cfg: Struct) -> str:
    with StringIO() as f:
        dump_config(cfg, f)
        f.seek(0)
        return f.read()


def validate_config(x):
    # TODO
    log.warning("Skipping input validation.")
    return x


def cosine_window(n: int, pedestal: float):
    if not (0.0 <= pedestal <= 1.0):
        raise ValueError(f"Expected pedestal between 0 and 1, got {pedestal}.")
    b = 0.5 * (1 - pedestal)
    t = np.arange(n) - (n - 1) / 2.0
    return (1 - b) + b * np.cos(2 * np.pi * t / (n - 1))


def apply_window(kind: str, shape: float, z: np.ndarray) -> np.ndarray:
    """Return product of an array with a window of the same length.
    """
    n = len(z)
    if kind == 'kaiser':
        return np.kaiser(n, shape) * z
    elif kind == 'cosine':
        return cosine_window(n, shape) * z
    raise NotImplementedError(f"window {kind} not in (Kaiser, Cosine).")


def check_window_input(win: Struct, msg='') -> tuple[str, float]:
    """Check user input window kind and shape, log both, and
    return (kind, shape).
    """
    kind = win.kind.lower()
    if kind == 'kaiser':
        log.info(f"{msg}Kaiser(beta={win.shape})")
        if win.shape < 0.0:
            raise ValueError("Require positive Kaiser window shape")
    elif kind == 'cosine':
        log.info(f'{msg}Cosine(pedestal_height={win.shape})')
        if not (0.0 <= win.shape <= 1.0):
            raise ValueError("Require Cosine window parameter in [0, 1]")
    else:
        raise NotImplementedError(
            f"window '{kind}' not in ('Kaiser', 'Cosine').")
    return kind, win.shape


def get_max_chirp_duration(cfg: Struct):
    """Return maximum chirp duration (in seconds) among all sidebands,
    polarizations, and files referenced in the runconfig.
    """
    maxlen = 0.0
    for filename in cfg.input_file_group.input_file_path:
        raw = open_rrsd(filename)
        for freq, polarizations in raw.polarizations.items():
            for pol in polarizations:
                _, _, _, T = raw.getChirpParameters(freq, pol[0])
                maxlen = max(maxlen, T)
    return maxlen


def parse_rangecomp_mode(mode: str):
    lut = {"full": isce3.focus.RangeComp.Mode.Full,
           "same": isce3.focus.RangeComp.Mode.Same,
           "valid": isce3.focus.RangeComp.Mode.Valid}
    mode = mode.lower()
    if mode not in lut:
        raise ValueError(f"Invalid RangeComp mode {mode}")
    return lut[mode]


@overload
def prep_ephemeris(ephemeris: Attitude, rawnames: list[str], time_pad: float,
                   num_pad: int) -> Attitude:
    pass

@overload
def prep_ephemeris(ephemeris: Orbit, rawnames: list[str], time_pad: float,
                   num_pad: int) -> Orbit:
    pass

def prep_ephemeris(ephemeris, rawnames, time_pad, num_pad):
    """
    Try to crop orbit/attitude to raw data bounds (with padding) and always set
    reference epoch to match the one used for the raw data.  Failure to crop
    triggers a warning message.

    Parameters
    ----------
    ephemeris : isce3.core.Attitude or isce3.core.Orbit
        The input ephemeris to be cropped.
    rawnames : list of str
        List of input L0B filenames used to obtain the raw data start and
        stop times.
    time_pad : float
        Min amount of padding (in seconds) beyond raw data start and stop
        times to retain when cropping the orbit and attitude data.
        Must be >= 0.
    num_pad : int
        Min number of samples beyond raw data start and stop times to
        retain when cropping the orbit and attitude data. Must be >= 0.

    Returns
    -------
    isce3.core.Attitude or isce3.core.Orbit
        The cropped attitude or orbit data. The reference epoch will be
        the same as the first input L0B file.
    """
    epoch, t0, t1, _, _ = get_total_grid_bounds(rawnames)
    start = epoch + TimeDelta(t0 - time_pad)
    end = epoch + TimeDelta(t1 + time_pad)

    if isinstance(ephemeris, Attitude):
        name = "attitude"
    elif isinstance(ephemeris, Orbit):
        name = "orbit"
    else:
        raise TypeError("bad type for ephemeris, expected Attitude or Orbit "
                        f"instance, got {type(ephemeris)}")

    log.info(f"Original {name} data file spans time interval "
             f"[{ephemeris.start_datetime}, {ephemeris.end_datetime}]")
    log.info(f"Cropping {name} to {num_pad} points beyond [{start}, {end}]")
    try:
        eph = ephemeris.crop(start, end, num_pad)
    except ValueError:
        eph = ephemeris.copy()  # be sure not to modify user input
        log.warning(f"Failed to crop {name}.  Trying to proceed anyway.")
    else:
        log.info(f"Cropped {name} to interval "
            f"[{eph.start_datetime}, {eph.end_datetime}]")

    eph.update_reference_epoch(epoch)
    return eph


def get_orbit(cfg: Struct) -> Orbit:
    """Read orbit from XML if available or else from first L0B, try to crop it,
    and set epoch to match first L0B.
    """
    # num_pad=3 to guarantee enough points for Hermite interp
    num_pad = 3
    time_pad = cfg.processing.ephemeris_crop_pad
    rawfiles = cfg.input_file_group.input_file_path

    xml = cfg.dynamic_ancillary_file_group.orbit
    if xml is not None:
        log.info("Loading orbit from external XML file.")
        orbit = nisar.products.readers.orbit.load_orbit_from_xml(xml)
    else:
        log.info("Loading orbit from L0B file.")
        if len(rawfiles) > 1:
            raise NotImplementedError("Can't concatenate orbit data.")
        raw = open_rrsd(rawfiles[0])
        orbit = raw.getOrbit()

    return prep_ephemeris(orbit, rawfiles, time_pad, num_pad)


def get_attitude(cfg: Struct) -> Attitude:
    """Read attitude from XML if available or else from first L0B, try to crop
    it, and set epoch to match first L0B.
    """
    # num_pad = 1 to guarantee enough points for slerp
    num_pad = 1
    time_pad = cfg.processing.ephemeris_crop_pad
    rawfiles = cfg.input_file_group.input_file_path

    xml = cfg.dynamic_ancillary_file_group.pointing
    if xml is not None:
        log.info("Loading attitude from external XML file")
        attitude = nisar.products.readers.attitude.load_attitude_from_xml(xml)
    else:
        log.info("Loading attitude from L0B file.")
        rawfiles = cfg.input_file_group.input_file_path
        if len(rawfiles) > 1:
            raise NotImplementedError("Can't concatenate attitude data.")
        raw = open_rrsd(rawfiles[0])
        attitude = raw.getAttitude()

    return prep_ephemeris(attitude, rawfiles, time_pad, num_pad)


def get_total_grid_bounds(rawfiles: list[str]):
    times, ranges = [], []
    for fn in rawfiles:
        raw = open_rrsd(fn)
        for frequency, pols in raw.polarizations.items():
            for pol in pols:
                ranges.append(raw.getRanges(frequency, tx=pol[0]))
                times.append(raw.getPulseTimes(frequency, tx=pol[0]))
    rmin = min(r[0] for r in ranges)
    rmax = max(r[-1] for r in ranges)
    dtmin = min(epoch + isce3.core.TimeDelta(t[0]) for (epoch, t) in times)
    dtmax = max(epoch + isce3.core.TimeDelta(t[-1]) for (epoch, t) in times)
    epoch = min(epoch for (epoch, t) in times)
    tmin = (dtmin - epoch).total_seconds()
    tmax = (dtmax - epoch).total_seconds()
    return epoch, tmin, tmax, rmin, rmax


def get_total_grid(rawfiles: list[str], dt, dr):
    epoch, tmin, tmax, rmin, rmax = get_total_grid_bounds(rawfiles)
    nt = int(np.ceil((tmax - tmin) / dt)) + 1
    nr = int(np.ceil((rmax - rmin) / dr)) + 1
    t = isce3.core.Linspace(tmin, dt, nt)
    r = isce3.core.Linspace(rmin, dr, nr)
    return epoch, t, r


def convert_epoch(t: list[float], epoch_in: DateTime, epoch_out: DateTime) -> list[float]:
    TD = isce3.core.TimeDelta
    return [(epoch_in - epoch_out + TD(ti)).total_seconds() for ti in t]


def get_dem(cfg: Struct):
    dem = isce3.geometry.DEMInterpolator(
        height=cfg.processing.dem.reference_height,
        method=cfg.processing.dem.interp_method)
    fn = cfg.dynamic_ancillary_file_group.dem_file
    if fn:
        log.info(f"Loading DEM {fn}")
        dem.load_dem(RasterIO(fn))
        dem.compute_min_max_mean_height()
    else:
        log.warning("No DEM given, using ref height "
                    f"{cfg.processing.dem.reference_height} (m).")
    return dem


def make_doppler_lut(rawfiles: list[str],
        az: float = 0.0,
        orbit: Optional[isce3.core.Orbit] = None,
        attitude: Optional[isce3.core.Attitude] = None,
        dem: Optional[isce3.geometry.DEMInterpolator] = None,
        azimuth_spacing: float = 1.0,
        range_spacing: float = 1e3,
        interp_method: str = "bilinear",
        epoch: Optional[DateTime] = None):
    """Generate Doppler look up table (LUT).

    Parameters
    ----------
    rawfiles
        List of NISAR L0B format raw data files.
    az : optional
        Complement of the angle between the along-track axis of the antenna and
        its electrical boresight, in radians.  Zero for non-scanned, flush-
        mounted antennas like ALOS-1.
    orbit : optional
        Path of antenna phase center.  Defaults to orbit in first L0B file.
    attitude : optional
        Orientation of antenna.  Defaults to attitude in first L0B file.
    dem : optional
        Digital elevation model, height in m above WGS84 ellipsoid. Default=0 m.
        Will calculate stats (modifying input object) if they haven't already
        been calculated.
    azimuth_spacing : optional
        LUT grid spacing in azimuth, in seconds.  Default=1 s.
    range_spacing : optional
        LUT grid spacing in range, in meters.  Default=1000 m.
    interp_method : optional
        LUT interpolation method. Default="bilinear".
    epoch : isce3.core.DateTime, optional
        Time reference for output table.  Defaults to orbit.reference_epoch

    Returns
    -------
    fc
        Center frequency, in Hz, assumed for Doppler calculation.
        It is among those found in the input raw data files.
    LUT
        Look up table of Doppler = f(r,t)
    """
    # Input wrangling.
    if len(rawfiles) < 1:
        raise ValueError("Need at least one L0B file.")
    if azimuth_spacing <= 0.0:
        raise ValueError(f"Require azimuth spacing > 0, got {azimuth_spacing}")
    if range_spacing <= 0.0:
        raise ValueError(f"Require range spacing > 0, got {range_spacing}")
    raw = open_rrsd(rawfiles[0])
    if orbit is None:
        orbit = raw.getOrbit()
    if attitude is None:
        attitude = raw.getAttitude()
    if dem is None:
        dem = isce3.geometry.DEMInterpolator()
    elif not dem.have_stats:
        dem.compute_min_max_mean_height()
    if epoch is None:
        epoch = orbit.reference_epoch
    # Ensure consistent time reference (while avoiding side effects).
    if orbit.reference_epoch != epoch:
        orbit = orbit.copy()
        orbit.update_reference_epoch(epoch)
    if attitude.reference_epoch != epoch:
        attitude = attitude.copy()
        attitude.update_reference_epoch(epoch)

    side = require_constant_look_side(open_rrsd(fn) for fn in rawfiles)
    # Use a nominal center frequency, which we'll return for user reference.
    frequency = next(iter(raw.polarizations))
    fc = raw.getCenterFrequency(frequency)

    # Now do the actual calculations.
    wvl = isce3.core.speed_of_light / fc
    epoch_in, t, r = get_total_grid(rawfiles, azimuth_spacing, range_spacing)

    # If timespan is too small, only one time may be provided, causing the LUT
    # construction to fail. Fall back to t ± Δt/2 to preserve az spacing.
    # Also clip to orbit start/end times if the orbit timespan is too small.
    if len(t) == 1:
        tmin = max(t[0] - azimuth_spacing / 2, orbit.start_time)
        tmax = min(t[0] + azimuth_spacing / 2, orbit.end_time)
        t = [tmin, tmax]

    t = convert_epoch(t, epoch_in, epoch)
    lut = isce3.geometry.make_doppler_lut_from_attitude(
        az_time=t,
        slant_range=r,
        orbit=orbit,
        attitude=attitude,
        wavelength=wvl,
        dem=dem,
        az_angle=az,
        interp_method=interp_method,
        bounds_error=False,
    )
    return fc, lut


def make_doppler(cfg: Struct, *, epoch: Optional[DateTime] = None,
        orbit: Optional[Orbit] = None, attitude: Optional[Attitude] = None,
        dem: Optional[DEMInterpolator] = None):
    """
    Generate Doppler LUT based on RSLC config file.  Optional inputs can
    be used to avoid unnecessarily loading files again.
    """
    log.info("Generating Doppler LUT from pointing")
    if orbit is None:
        orbit = get_orbit(cfg)
    if attitude is None:
        attitude = get_attitude(cfg)
    if dem is None:
        dem = get_dem(cfg)
    opt = cfg.processing.doppler
    az = np.radians(opt.azimuth_boresight_deg)
    rawfiles = cfg.input_file_group.input_file_path

    fc, lut = make_doppler_lut(rawfiles,
                               az=az, orbit=orbit, attitude=attitude,
                               dem=dem, azimuth_spacing=opt.spacing.azimuth,
                               range_spacing=opt.spacing.range,
                               interp_method=opt.interp_method,  epoch=epoch)

    log.info(f"Made Doppler LUT for fc={fc} Hz, "
        f"az={opt.azimuth_boresight_deg} deg with mean={lut.data.mean()} Hz")
    return fc, lut


def zero_doppler_like(dop: LUT2d):
    x = np.zeros_like(dop.data)
    # Assume we don't care about interp method or bounds when all values == 0.
    method, check_bounds = "nearest", False
    return LUT2d(dop.x_start, dop.y_start, dop.x_spacing, dop.y_spacing, x,
                 method, check_bounds)


def scale_doppler(dop: LUT2d, c: float):
    if dop.have_data:
        x = c * dop.data
        return LUT2d(dop.x_start, dop.y_start, dop.x_spacing, dop.y_spacing, x,
                    dop.interp_method, dop.bounds_error)
    if dop.ref_value == 0.0:
        return LUT2d()
    raise NotImplementedError("No way to scale Doppler with nonzero ref_value")


def make_output_grid(cfg: Struct,
                     epoch: DateTime, t0: float, t1: float, max_prf: float,
                     r0: float, r1: float, dr: float,
                     side: Union[str, isce3.core.LookSide],
                     orbit: Orbit,
                     fc_ref: float, doppler: LUT2d,
                     chirplen_meters: float,
                     dem: isce3.geometry.DEMInterpolator) -> RadarGridParameters:
    """
    Given the available raw data extent (in slow time and slant range) figure
    out a reasonable output extent that:
        * Accounts for the reskew to zero-Doppler,
        * Takes care not to exceed the available ephemeris data, and
        * Excludes the coherent processing interval in range and azimuth.
    These extents are then overridden by any user-provided runconfig data and
    used to construct a RadarGridParameters object suitable for focusing.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig data.
    epoch : DateTime
        Reference for all time tags.
    t0 : float
        First pulse time available in all raw data, in seconds since epoch.
    t1 : float
        Last pulse time available in all raw data, in seconds since epoch.
    max_prf : float
        The highest PRF used in all raw data, in Hz.
    r0 : float
        Minimum one-way range among all raw data, in meters.
    r1 : float
        Maximum one-way range among all raw data, in meters.
    dr : float
        Desired range spacing of output grid, in meters.
    side : {"left", "right"} or isce3.core.LookSide
        Radar look direction
    orbit : Orbit
        Radar orbit
    fc_ref : float
        Radar center frequency corresponding to `doppler` object.  Also
        used to determine CPI and populate wavelength in output grid object.
    doppler : LUT2d
        Doppler centroid of the raw data.
    chirplen_meters : float
        Maximum chirp length among raw data, expressed as range in meters.
        Used to help determine the region that can be fully focused.
    dem : isce3.geometry.DEMInterpolator
        Digital elevation model containing height above WGS84 ellipsoid,
        in meters.

    Returns
    -------
    grid : RadarGridParameters
        Zero-Doppler grid suitable for focusing.
    """
    assert orbit.reference_epoch == epoch
    ac = cfg.processing.azcomp
    wavelength = isce3.core.speed_of_light / fc_ref

    # Calc approx synthetic aperture duration (coherent processing interval).
    tmid = 0.5 * (t0 + t1)
    cpi = isce3.focus.get_sar_duration(tmid, r1, orbit, isce3.core.Ellipsoid(),
                                       ac.azimuth_resolution, wavelength)
    log.debug(f"Approximate synthetic aperture duration is {cpi} s.")

    # Crop to fully focused region, ignoring range-dependence of CPI.
    # Ignore sampling of convolution kernels, accepting possibility of a few
    # pixels that are only 99.9% focused.
    # CPI is symmetric about Doppler centroid.
    t0 = t0 + cpi / 2
    t1 = t1 - cpi / 2
    # Range delay is defined relative to _start_ of TX pulse.
    r1 = r1 - chirplen_meters

    # Output grid is zero Doppler, so reskew the four corners and assume they
    # enclose the image.  Take extrema as default processing box.
    # Define a capture to save some typing
    zerodop = isce3.core.LUT2d()
    def reskew_to_zerodop(t, r):
        return isce3.geometry.rdr2rdr(t, r, orbit, side, doppler, wavelength,
            dem, doppler_out=zerodop,
            rdr2geo_params=get_rdr2geo_params(cfg),
            geo2rdr_params=get_geo2rdr_params(cfg, orbit))

    # One annoying case is where the orbit data covers the raw pulse times
    # and nothing else.  The code can crash when trying to compute positions on
    # the output zero-Doppler grid because the reskew time offset causes it to
    # run off the end of the available orbit data.  Also the Newton solvers need
    # some wiggle room. As a workaround, let's nudge the default bounds until
    # we're sure the code doesn't crash.
    def reskew_near_far_with_nudge(t, r0, r1, step, tstop):
        assert (tstop - t) * step > 0, "Sign of step must bring t towards tstop"
        offset = 0.0
        # Give up when we've nudged t past tstop (nudging forwards or
        # backwards).
        while (tstop - (t + offset)) * step > 0:
            try:
                ta, ra = reskew_to_zerodop(t + offset, r0)
                tb, rb = reskew_to_zerodop(t + offset, r1)
                return offset, ta, ra, tb, rb
            except RuntimeError:
                log.info(f"nudging by step={step}")
                offset += step
        raise RuntimeError("No valid geometry.  Invalid orbit data?")

    # Solve for points at near range (a) and far range (b) at start time.
    offset0, ta, ra, tb, rb = reskew_near_far_with_nudge(t0, r0, r1, 0.1, t1)
    log.debug(f"offset0 = {offset0}")
    if abs(offset0) > 0.0:
        log.warning(f"Losing up to {offset0} seconds of image data at start due"
            " to insufficient orbit data.")
    # Solve for points at near range (c) and far range (d) at end time.
    offset1, tc, rc, td, rd = reskew_near_far_with_nudge(t1, r0, r1, -0.1, t0)
    log.debug(f"offset1 = {offset1}")
    if abs(offset1) > 0.0:
        log.warning(f"Losing up to {offset1} seconds of image data at end due"
            " to insufficient orbit data.")

    # "z" for zero Doppler.  Reskew varies with range, so take most conservative
    # bounding box to ensure fully focused data everywhere.
    t0z = max(ta, tb)
    r0z = max(ra, rc)
    t1z = min(tc, td)
    r1z = min(rb, rd)
    log.debug(f"Reskew time offset at start {t0z - t0 - offset0} s")
    log.debug(f"Reskew time offset at end {t1z - t1 - offset1} s")
    log.debug(f"Reskew range offset at start {r0z - r0} m")
    log.debug(f"Reskew range offset at end {r1z - r1} m")

    dt0 = epoch + isce3.core.TimeDelta(t0z)
    dt1 = epoch + isce3.core.TimeDelta(t1z)
    log.info(f"Approximate fully focusable time interval is [{dt0}, {dt1}]")
    log.info(f"Approximate fully focusable range interval is [{r0z}, {r1z}]")

    # Save typing when handling defaults.
    p = cfg.processing.output_grid

    # Usually for NISAR grid PRF should be specified as 1520 in the runconfig.
    # If not then take max() as most conservative choice.
    prf = p.output_prf if (p.output_prf is not None) else max_prf

    # Snap default start time & range to integer grid.
    # NOTE Current defaults mean range_snap_interval is never None, but cover
    # that scenario in case somebody changes the default under our nose.  The
    # reason for the non-null default is that with a dual-band system like NISAR
    # you'd want to choose a default range snap based on knowledge of both
    # subbands, which would require some refactoring.  For example, you'd want
    # to snap the frequencyA grid to the coarser frequencyB spacing.
    dt0 = isce3.math.snap_datetime(dt0,
        p.time_snap_interval if p.time_snap_interval is not None else 1 / prf)
    t0z = (dt0 - epoch).total_seconds()
    r0z = isce3.math.snap(r0z,
        p.range_snap_interval if p.range_snap_interval is not None else dr)

    log.info(f"Snapped default start time to {dt0}")
    log.info(f"Snapped default start range to {r0z} m")

    if p.start_time:
        t0z = (DateTime(p.start_time) - epoch).total_seconds()
    if p.end_time:
        t1z = (DateTime(p.end_time) - epoch).total_seconds()
    r0z = p.start_range if (p.start_range is not None) else r0z
    r1z = p.end_range if (p.end_range is not None) else r1z

    nr = round((r1z - r0z) / dr)
    nt = round((t1z - t0z) * prf)
    assert (nr > 0) and (nt > 0)
    return RadarGridParameters(t0z, wavelength, prf, r0z, dr, side, nt, nr,
                               epoch)


def get_rdr2geo_params(cfg: Struct) -> dict:
    """
    Geo rdr2geo parameters from RSLC config structure.
    This converts config keys look_{min,max}_deg to radians.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig structure, stripped of leading `runconfig.groups`.

    Returns
    -------
    rdr2geo_params : dict
        A dict with the three keys {"look_min", "look_max", "tol_height"}
    """
    # will always contain necessary fields since filled with defaults
    g = cfg.processing.rdr2geo
    return dict(
        tol_height = g.tol_height,
        look_min = np.deg2rad(g.look_min_deg),
        look_max = np.deg2rad(g.look_max_deg))


def get_geo2rdr_params(cfg: Struct, orbit: Optional[Orbit] = None) -> dict:
    """
    Geo geo2rdr parameters from RSLC config structure.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig structure, stripped of leading `runconfig.groups`.
    orbit : isce3.core.Orbit, optional
        If orbit is provided and time_{start,end} are null in the runconfig,
        sets time_{start,end} to the orbit time bounds.  This effectively
        overrides the default geo2rdr_bracket behavior which sets the default
        search interval to the intersection of the orbit interval and the
        Doppler LUT interval.  As a consequence, providing the orbit may cause
        out-of-bounds Doppler lookups, which may throw an exception or result in
        extrapolation depending on the `have_data` and `bounds_error` properties
        of the Doppler LUT2d object.

    Returns
    -------
    geo2rdr_params : dict
        A dict with the three keys {"time_start", "time_end", "tol_aztime"}
    """
    geo2rdr_params = vars(cfg.processing.geo2rdr)
    t0 = geo2rdr_params.get("time_start", None)
    t1 = geo2rdr_params.get("time_end", None)
    if orbit is not None:
        if t0 is None:
            geo2rdr_params["time_start"] = orbit.start_time
        if t1 is None:
            geo2rdr_params["time_end"] = orbit.end_time
    return geo2rdr_params


Selection2d = tuple[slice, slice]
TimeBounds = tuple[float, float]
BlockPlan = list[tuple[Selection2d, TimeBounds]]

def plan_processing_blocks(cfg: Struct, grid: RadarGridParameters,
                           doppler: LUT2d, dem: isce3.geometry.DEMInterpolator,
                           orbit: Orbit, pad: float = 0.1) -> BlockPlan:
    """
    Subdivide output grid into processing blocks and find time bounds of raw
    data needed to focus each one.

    This has the added benefit of coarsely checking that we can calculate the
    geometry over the requested domain.  By default the synthetic aperture
    length is padded by 10% to make the bounds a bit conservative, but this can
    be adjusted via the `pad` parameter.
    """
    assert pad >= 0.0
    ac = cfg.processing.azcomp
    nr = ac.block_size.range
    na = ac.block_size.azimuth

    if nr < 1:
        nr = grid.width

    blocks = []
    for i in range(0, grid.length, na):
        imax = min(i + na, grid.length)
        for j in range(0, grid.width, nr):
            jmax = min(j + nr, grid.width)
            blocks.append(np.s_[i:imax, j:jmax])

    zerodop = isce3.core.LUT2d()
    results = []
    for rows, cols in blocks:
        raw_times = []
        # Compute zero-to-native Doppler reskew times for four corners.
        # NOTE if this eats up a lot of time it can be sped up 4x by
        # computing each vertex and later associating them with blocks.
        for (u, v) in ((rows.start, cols.start), (rows.stop-1, cols.start),
                       (rows.start, cols.stop-1), (rows.stop-1, cols.stop-1)):
            t = grid.sensing_start + u / grid.prf
            r = grid.starting_range + v * grid.range_pixel_spacing
            try:
                traw, _ = isce3.geometry.rdr2rdr(t, r, orbit, grid.lookside,
                    zerodop, grid.wavelength, dem, doppler_out=doppler,
                    rdr2geo_params=get_rdr2geo_params(cfg),
                    geo2rdr_params=get_geo2rdr_params(cfg, orbit))
            except RuntimeError as e:
                dt = grid.ref_epoch + isce3.core.TimeDelta(t)
                log.error(f"Reskew zero-to-native failed at t={dt} r={r}")
                raise RuntimeError("Could not compute imaging geometry") from e
            raw_times.append(traw)
        sub_grid = grid[rows, cols]
        cpi = isce3.focus.get_sar_duration(sub_grid.sensing_mid,
                                    sub_grid.end_range, orbit,
                                    isce3.core.Ellipsoid(),
                                    ac.azimuth_resolution, sub_grid.wavelength)
        cpi *= 1.0 + pad
        raw_begin = min(raw_times) - cpi / 2
        raw_end = max(raw_times) + cpi / 2
        results.append(((rows, cols), (raw_begin, raw_end)))
    return results


def total_bounds(blocks_bounds: BlockPlan) -> TimeBounds:
    begin = min(t0 for _, (t0, t1) in blocks_bounds)
    end = max(t1 for _, (t0, t1) in blocks_bounds)
    return (begin, end)


def is_overlapping(a, b, c, d):
    assert (b >= a) and (d >= c)
    return (d >= a) and (c <= b)

def get_kernel(cfg: Struct):
    # TODO
    opt = cfg.processing.azcomp.kernel
    if opt.type.lower() != 'knab':
        raise NotImplementedError("Only Knab kernel implemented.")
    n = 1 + 2 * opt.halfwidth
    kernel = isce3.core.KnabKernel(n, 1 / 1.2)
    assert opt.fit.lower() == "table"
    table = isce3.core.TabulatedKernelF32(kernel, opt.fit_order)
    return table


# Work around for fact that slices are not hashable and can't be used as
# dictionary keys or entries in sets
# https://bugs.python.org/issue1733184
def unpack_slices(slices: tuple[slice, slice]):
    rows, cols = slices
    return ((rows.start, rows.stop, rows.step),
            (cols.start, cols.stop, cols.step))


class BackgroundWriter(isce3.io.BackgroundWriter):
    """
    Compute statistics and write RSLC data in a background thread.

    Parameters
    ----------
    range_cor : np.ndarray
        1D range correction to apply to data before writing, e.g., phasors that
        shift the frequency to baseband.  Length should match number of columns
        (range bins) in `dset`.
    dset : h5py.Dataset
        HDF5 dataset to write blocks to.  Shape should be 2D (azimuth, range).
    data_type : str in {"complex32", "complex64", "complex64_zero_mantissa"}
        Output data type is complex32 (pairs of float16), complex64 (pairs of
        float32), or complex64_zero_mantissa (pairs of float32 with least
        significant mantissa bits set to zero).
    mantissa_nbits : int, optional
        Precision parameter for data_type=complex64_zero_mantissa.  Sets the
        number of nonzero bits in the floating point mantissa.  Must be between
        0 and 23, inclusive.  Defaults to 10 bits (half precision).

    Attributes
    ----------
    stats : isce3.math.StatsRealImagFloat32
        Statistics of all data that have been written.
    """
    def __init__(self, range_cor, dset, data_type, mantissa_nbits=10, **kw):
        log.debug(f"Preparing BackgroundWriter with data_type={data_type}"
            f" and mantissa_nbits={mantissa_nbits}")
        self.range_cor = range_cor
        self.dset = dset
        # Keep track of which blocks have been written and the image stats
        # for each one.
        self._visited_blocks = dict()

        if data_type == "complex64":
            self.encode = lambda x: x  # no-op
            self.readback = lambda key: self.dset[key]
        elif data_type == "complex32":
            self.encode = to_complex32
            self.readback = lambda key: read_c4_dataset_as_c8(self.dset, key)
        elif data_type == "complex64_zero_mantissa":
            # in-place tuncation returns None, so use "or x" to return data
            self.encode = lambda x: truncate_mantissa(x, mantissa_nbits) or x
            self.readback = lambda key: self.dset[key]
        else:
            raise ValueError(f"Requested invalid data_type {data_type}")

        # Make sure output HDF5 type is compatible with requested type string.
        # XXX Old versions of h5py crash when accessing dtype when complex32.
        try:
            h5type = dset.dtype
        except TypeError:
            h5type = "unknown"  # hopefully complex32
        c8match = data_type.startswith("complex64") and h5type == np.complex64
        c4match = data_type == "complex32" and h5type in ("unknown", complex32)
        if (not c8match) and (not c4match):
            raise TypeError(f"Requested {data_type} encoding but type of HDF5 "
                f"dataset is {h5type}")

        super().__init__(**kw)

    @property
    def stats(self):
        total_stats = isce3.math.StatsRealImagFloat32()
        for block_stats in self._visited_blocks.values():
            total_stats.update(block_stats)
        return total_stats

    def write(self, z, block):
        """
        Scale `z` by `range_cor` (in-place), apply encoding (possibly in-place)
        then write to file and accumulate statistics.  If the block has been
        written already, then the current data will be added to the existing
        results (dset[block] += ...).

        Parameters
        ----------
        z : np.ndarray
            An arbitrary 2D chunk of data to store in `dset`.
        block : tuple[slice]
            Pair of slices describing the (azimuth, range) selection of `dset`
            where the chunk should be stored.

        Notes
        -----
        For efficiency, no check is made for partially overlapping block
        selections.  In those cases data will be written directly without
        accumulating previous results.
        """
        # scale and deramp
        z *= self.range_cor[None, block[1]]
        # Expect in mixed-mode cases that each file will contribute partially
        # focused blocks at mode-change boundaries.  In that case accumulate
        # data, but avoid slow reads if possible by keeping track of which
        # blocks we've already visited.
        # XXX Slices aren't hashable, so convert them to a type that is so that
        # we can still do O(1) lookups.
        key = unpack_slices(block)
        if key in self._visited_blocks:
            log.debug("reading back SLC data at block %s", block)
            z += self.readback(block)
        # Calculate block stats.  Don't accumulate since image is mutable in
        # mixed-mode case.
        s = isce3.math.StatsRealImagFloat32(z)
        amax = np.max(np.abs([s.real.max, s.real.min, s.imag.max, s.imag.min]))
        log.debug(f"scaled max component = {amax}")
        self._visited_blocks[key] = s
        # Encode and write to HDF5
        self.dset.write_direct(self.encode(z), dest_sel=block)


def get_dataset_creation_options(cfg: Struct, shape: tuple[int, int]) -> dict:
    """
    Get h5py keyword arguments needed for image dataset creation.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig data. Only reads `cfg.output` group.
    shape : tuple[int, int]
        Shape of dataset.  Used to determine upper bounds on chunk sizes.

    Returns
    -------
    opts : dict
        Dictionary containing the keys {"chunks", "compression",
        "compression_opts", "shuffle", "dtype"} suitable for forwarding to
        `h5py.Group.create_dataset`.
    """
    opts = dict(chunks=None, compression=None, compression_opts=None,
        shuffle=None)
    g = cfg.output
    # XXX GSLC always chunks, while it's optional for RSLC since we may want to
    # disable chunks in order to make a dataset mmap-able.  However, since
    # default value is not null, we need another non-null sentinel value to
    # indicate this.  Choose [-1, -1], which implies one full-sized chunk.
    if g.chunk_size != [-1, -1]:
        opts["chunks"] = tuple(min(dims) for dims in zip(g.chunk_size, shape))
    if g.compression_enabled:
        if opts["chunks"] is None:
            raise ValueError("Chunk size cannot be None when "
                "compression is enabled")
        opts['compression'] = g.compression_type
        if g.compression_type == "gzip":
            opts['compression_opts'] = g.compression_level
        opts['shuffle'] = g.shuffle
    if g.data_type in ("complex64", "complex64_zero_mantissa"):
        opts["dtype"] = np.complex64
    elif g.data_type == "complex32":
        opts["dtype"] = complex32
    else:
        raise ValueError(f"invalid runconfig output.data_type")
    return opts


def resample(raw: np.ndarray, t: np.ndarray,
             grid: RadarGridParameters, swaths: np.ndarray,
             orbit: isce3.core.Orbit, doppler: isce3.core.LUT2d, L=12.0,
             fn="regridded.c8"):
    """
    Fill gaps and resample raw data to uniform grid using BLU method.

    Parameters
    ----------
    raw : array-like [complex float32, rows=pulses, cols=range bins]
        Decoded raw data.
    t : np.ndarray [float64]
        Pulse times (seconds since orbit/grid epoch).
    grid : isce3.product.RadarGridParameters
        Raw data grid.  Output will have same parameters.
    swaths : np.ndarray [int]
        Valid subswath samples, dims = (ns, nt, 2) where ns is the number of
        sub-swaths, nt is the number of pulses, and the trailing dimension is
        the [start, stop) indices of the sub-swath.
    orbit : isce3.core.Orbit
        Orbit.  Used to determine velocity for scaling autocorrelation function.
    doppler : isce3.core.LUT2d [double]
        Raw data Doppler look up table.  Must be valid over entire grid.
    L : float
        Antenna azimuth dimension, in meters.  Used for scaling sinc antenna
        pattern model for azimuth autocorrelation function.
    fn : string, optional
        Filename for output memory map.

    Returns
    -------
    regridded : array-like [complex float32, rows=pulses, cols=range bins]
        Gridded, gap-free raw data.
    """
    assert raw.shape == (grid.length, grid.width)
    assert len(t) == raw.shape[0]
    assert grid.ref_epoch == orbit.reference_epoch
    # Compute uniform time samples for given raw data grid
    out_times = t[0] + np.arange(grid.length) / grid.prf
    # Ranges are the same.
    r = grid.starting_range + grid.range_pixel_spacing * np.arange(grid.width)
    regridded = np.memmap(fn, mode="w+", shape=grid.shape, dtype=np.complex64)
    for i, tout in enumerate(out_times):
        # Get velocity for scaling autocorrelation function.  Won't change much
        # but update every pulse to avoid artifacts across images.
        v = np.linalg.norm(orbit.interpolate(tout)[1])
        acor = isce3.core.AzimuthKernel(L / v)
        # Figure out what pulses are in play by computing weights without mask.
        # TODO All we really need is offset and len(weights)... maybe refactor.
        offset, weights = isce3.focus.get_presum_weights(acor, t, tout)
        nw = len(weights)
        # Compute valid data mask (transposed).
        # NOTE Could store the whole mask instead of recomputing blocks.
        mask = np.zeros((grid.width, nw), dtype=bool)
        for iw in range(nw):
            it = offset + iw
            for swath in swaths:
                start, end = swath[it]
                mask[start:end, iw] = True
        # The pattern of missing samples in any given column can change
        # depending on the gap structure.  Recomputing weights is expensive,
        # though, so compute a hash we can use to cache the unique weight
        # vectors.
        twiddle = 1 << np.arange(nw)
        ids = isce3.focus.compute_ids_from_mask(mask)
        # Compute weights for each unique mask pattern.
        lut = dict()
        for uid in isce3.focus.get_unique_ids(ids):
            # Invert the hash to get the mask back
            valid = (uid & twiddle).astype(bool)
            # Pull out valid times for this mask config and compute weights.
            tj = t[offset:offset+nw][valid]
            joff, jwgt = isce3.focus.get_presum_weights(acor, tj, tout)
            assert joff == 0
            # Now insert zeros where data is invalid to get full-length weights.
            jwgt_full = np.zeros_like(weights)
            jwgt_full[valid] = jwgt
            lut[uid] = jwgt_full
        # Fill weights for entire block using look up table.
        w = isce3.focus.fill_weights(ids, lut)
        # Read raw data.
        block = np.s_[offset:offset+nw, :]
        x = raw[block]
        # Apply weights and Doppler deramp.  Zero phase at tout means no need to
        # re-ramp afterwards.
        trel = t[offset:offset+nw] - tout
        fd = doppler.eval(tout, r)
        isce3.focus.apply_presum_weights(regridded[i,:], trel, fd, w, x)
    return regridded


def process_rfi(cfg: Struct, raw_data: np.ndarray,
                tmpfile: Callable = lambda name: open(name, "wb")):
    """
    Run radio frequency interference (RFI) detection and mitigation as
    configured by user input.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig data
    raw_data : np.ndarray[np.complex64]
        Raw data layer.  May be modified in-place if mitigation is enabled.
    tmpfile : Callable
        Function of a single string argument that returns an open file handle.

    Returns
    -------
    raw_data_mitigated : np.ndarray[np.complex64]
        Buffer containing mitigated data.
        Usually `raw_data is raw_data_mitigated == True` (in-place processing)
        unless mitigation is enabled and debug layers are requested in the
        config file, in which case a separate array is memory mapped for
        easy comparison.
    rfi_likelihood : float
        Ratio of number of CPIs detected with RFI Eigenvalues over that of total
        number of CPIs.  Returns numpy.nan if detection is not enabled.
    """
    opt = cfg.processing.radio_frequency_interference
    if not opt.detection_enabled:
        if opt.mitigation_enabled:
            raise ValueError("Requested RFI mitigation but disabled detection.")
        log.info("Configured to skip RFI processing")
        return raw_data, np.nan
    if opt.mitigation_algorithm != "ST-EVD" and opt.mitigation_algorithm != "FDNF":
        raise NotImplementedError("Only ST-EVD and FDNF RFI algorithms are supported")
    msg = f"Running {opt.mitigation_algorithm} radio frequency interference (RFI) detection"
    if opt.mitigation_enabled:
        msg += " and mitigation"
    log.info(msg)

    # Mitigate in place unless user wants a debug file to compare raw and
    # mitigated data.  This means you'd need to run the workflow twice to find
    # a bug specific to in-place vs out-of-place processing.
    raw_data_mitigated = raw_data
    if opt.mitigation_enabled and not cfg.processing.delete_tempfiles:
        fd = tmpfile("_raw_clean.c8")
        log.info(f"Writing RFI mitigated raw data to memory map {fd.name}.")
        raw_data_mitigated = np.memmap(fd, mode="w+", shape=raw_data.shape,
                           dtype=np.complex64)

    if opt.mitigation_algorithm == "ST-EVD":
        opt_evd = opt.slow_time_evd
        threshold_params = isce3.signal.rfi_detection_evd.ThresholdParams(
            opt_evd.threshold_hyperparameters.x, opt_evd.threshold_hyperparameters.y)

        rfi_likelihood = isce3.signal.rfi_process_evd.run_slow_time_evd(
            raw_data,
            opt_evd.cpi_length,
            opt_evd.max_emitters,
            num_max_trim=opt_evd.num_max_trim,
            num_min_trim=opt_evd.num_min_trim,
            max_num_rfi_ev=opt_evd.max_num_rfi_ev,
            num_rng_blks=opt.num_range_blocks,
            threshold_params=threshold_params,
            num_cpi_tb=opt_evd.num_cpi_per_threshold_block,
            mitigate_enable=opt.mitigation_enabled,
            raw_data_mitigated=raw_data_mitigated)
    else:
        opt_fnf = opt.freq_notch_filter
        rfi_likelihood = isce3.signal.rfi_freq_null.run_freq_notch(
            raw_data,
            opt_fnf.num_pulses_az,
            num_rng_blks=opt.num_range_blocks,
            az_winsize=opt_fnf.az_winsize,
            rng_winsize=opt_fnf.rng_winsize,
            trim_frac=opt_fnf.trim_frac,
            pvalue_threshold=opt_fnf.pvalue_threshold,
            cdf_threshold=opt_fnf.cdf_threshold,
            nb_detect=opt_fnf.nb_detect,
            wb_detect=opt_fnf.wb_detect,
            mitigate_enable=opt.mitigation_enabled,
            raw_data_mitigated=raw_data_mitigated)


    log.info(f"RFI likelihood = {rfi_likelihood}")
    return raw_data_mitigated, rfi_likelihood


def delete_safely(filename):
    # Careful to avoid race on file deletion.  Use pathlib in Python 3.8+
    try:
        os.unlink(filename)
    except FileNotFoundError:
        pass


def get_range_deramp(grid: RadarGridParameters) -> np.ndarray:
    """Compute the phase ramp required to shift a backprojected grid to
    baseband in range.
    """
    r = grid.starting_range + grid.range_pixel_spacing * np.arange(grid.width)
    return np.exp(-1j * 4 * np.pi / grid.wavelength * r)


def require_ephemeris_overlap(ephemeris: Ephemeris,
                              t0: float, t1: float, name: str = "Ephemeris"):
    """Raise exception if ephemeris doesn't fully overlap time interval [t0, t1]
    """
    if ephemeris.contains(t0) and ephemeris.contains(t1):
        return
    dt0 = ephemeris.reference_epoch + isce3.core.TimeDelta(t0)
    dt1 = ephemeris.reference_epoch + isce3.core.TimeDelta(t1)
    msg = (f"{name} time span "
        f"[{ephemeris.start_datetime}, {ephemeris.end_datetime}] does not fully"
        f"overlap required time span [{dt0}, {dt1}]")
    log.error(msg)
    raise ValueError(msg)


def require_constant_look_side(rawlist: Iterable[Raw]) -> str:
    side_set = {raw.identification.lookDirection for raw in rawlist}
    if len(side_set) > 1:
        raise ValueError("Cannot combine left- and right-looking data.")
    return side_set.pop()


def get_common_mode(rawlist: list[Raw]) -> PolChannelSet:
    assert len(rawlist) > 0
    modes = [PolChannelSet.from_raw(raw) for raw in rawlist]
    common = reduce(lambda mode1, mode2: mode1.intersection(mode2), modes)
    # Make sure we regularize even if only one mode.
    return common.regularized()


def get_bands(mode: PolChannelSet) -> dict[str, Band]:
    assert mode == mode.regularized()
    bands = dict()
    for channel in mode:
        bands[channel.freq_id] = channel.band
    return bands


def get_max_prf(rawlist: Iterable[Raw]) -> float:
    """Calculate the average PRF in each Raw file and return the largest one.
    """
    prfs = []
    for raw in rawlist:
        freq, pols = next(iter(raw.polarizations.items()))
        tx = pols[0][0]
        _, grid = raw.getRadarGrid(frequency=freq, tx=tx)
        prfs.append(grid.prf)
    return max(prfs)


def prep_rangecomp(cfg, raw, raw_grid, channel_in, channel_out, cal=None):
    """Setup range compression.

    Parameters
    ----------
    cfg : Struct
        RSLC runconfig data
    raw : Raw
        NISAR L0B reader object
    raw_grid : RadarGridParameters
        Grid parameters for the raw data that will be compressed
    channel_in : PolChannel
        Input polarimetric channel info
    channel_out : PolChannel
        Output polarimetric channel info, different in mixed-mode case
    cal : Optional[RslcCalibration]
        RSLC calibration data.  Will apply gain and delay calibrations to chirp
        and grid if provided.

    Returns
    -------
    rc : RangeComp
        Range compression execution object
    rc_grid : RadarGridParameters
        Grid parameters for the output range-compressed data
    shift : float
        Frequency shift in rad/sample required to shift range-compressed data
        to desired center frequency.
    deramp : np.ndarray[np.complex64]
        Phase array (1D) that can be multiplied into the output data to shift
        to the desired center frequency.  Zero phase is referenced to range=0.
    """
    log.info("Generating chirp")
    tx = channel_in.pol[0]
    chirp = raw.getChirp(channel_in.freq_id, tx)
    log.info(f"Chirp length = {len(chirp)}")
    win_kind, win_shape = check_window_input(cfg.processing.range_window)

    fc, fs, K, _ = raw.getChirpParameters(channel_in.freq_id, tx)

    if channel_in.band != channel_out.band:
        log.info("Filtering chirp for mixed-mode processing.")
        # NOTE In mixed-mode case window is part of the filter design.
        design = cfg.processing.range_common_band_filter
        cb_filt, shift = nisar.mixed_mode.get_common_band_filter(
                                        channel_in.band, channel_out.band, fs,
                                        attenuation=design.attenuation,
                                        width=design.width,
                                        window=(win_kind, win_shape))
        log.info("Common-band filter length = %d", len(cb_filt))
        if len(cb_filt) > len(chirp):
            log.warning("Common-band filter is longer than chirp!  "
                        "Consider relaxing the filter design parameters.")
        chirp = np.convolve(cb_filt, chirp, mode="full")
    else:
        # In nominal case window the time-domain LFM chirp.
        chirp = apply_window(win_kind, win_shape, chirp)
        cb_filt, shift = [1.0], 0.0

    log.info("Normalizing chirp to unit white noise gain.")
    chirp *= 1.0 / np.linalg.norm(chirp)

    # Careful to use effective TBP after mixed-mode filtering.
    time_bw_product = channel_out.band.width**2 / abs(K)

    # Now we collect signal gain terms that could vary across NISAR modes so
    # that we can hopefully support all of them with a single set of cal
    # parameters.
    # The wavelength term here normalizes the azimuth reference length, which
    # is proportional to range * wavelength / azimuth_resolution.  The range
    # term is included in the range fading correction later (range_cor=True).
    # The number of pulses in the aperture is proportional to the input PRF.
    # Note that we assume the velocity is roughly constant (0.4% variation is
    # predicted for NISAR).
    # We don't try to compensate changes in azimuth resolution since we'll
    # always use the same one for nominal mission processing.
    # Other wavelength terms in the radar equation are already included in the
    # antenna pattern (eap=True).
    wavelength = isce3.core.speed_of_light / channel_out.band.center
    signal_gain = wavelength * np.sqrt(time_bw_product) * raw_grid.prf
    log.info("Renormalizing chirp by signal gain terms "
        f"wavelength * PRF * sqrt(TBP) = {signal_gain}")
    chirp *= 1.0 / signal_gain

    if cal:
        scale, delay = get_scale_and_delay(cal, channel_in.pol)
        log.info(f"Scaling chirp by calibration factor = {scale}")
        chirp *= scale

    rcmode = parse_rangecomp_mode(cfg.processing.rangecomp.mode)
    log.info(f"Preparing range compressor with mode={rcmode}")
    nr = raw_grid.shape[1]
    na = cfg.processing.rangecomp.block_size.azimuth
    rc = isce3.focus.RangeComp(chirp, nr, maxbatch=na, mode=rcmode)

    for notch_struct in cfg.processing.rangecomp.notches:
        notch = Notch(**vars(notch_struct))
        log.info("Applying notch %s", notch)
        notch = notch.normalized(fs, fc)
        rc.apply_notch(notch.frequency, notch.bandwidth)

    # Rangecomp modifies range grid.  Also update wavelength.
    # Careful that common-band filter delay is only half the filter
    # length but RangeComp bookkeeps the entire length of the ref. function.
    rc_grid = raw_grid.copy()
    rc_grid.starting_range -= rc_grid.range_pixel_spacing * (
        rc.first_valid_sample - (len(cb_filt) - 1) / 2)
    rc_grid.width = rc.output_size
    rc_grid.wavelength = wavelength

    if cal:
        log.info(f"Adjusting starting range by delay calibration = {delay} m")
        rc_grid.starting_range += delay

    r = rc_grid.starting_range + (
        rc_grid.range_pixel_spacing * np.arange(rc_grid.width))
    deramp = np.exp(1j * shift / rc_grid.range_pixel_spacing * r)

    return rc, rc_grid, shift, deramp


def get_antpat_inst(cfg: Struct) -> tuple[AntennaParser, InstrumentParser]:
    antfile = cfg.dynamic_ancillary_file_group.antenna_pattern
    insfile = cfg.dynamic_ancillary_file_group.internal_calibration
    if (antfile is None) or (insfile is None):
        if cfg.processing.is_enabled.eap:
            raise RuntimeError("Requested elevation antenna pattern "
                "(processing.is_enabled.eap=True) but did not provide "
                "both inputs needed to compute the antenna patterns "
                "(antenna_pattern and internal_calibration)")
        return (None, None)
    ant = AntennaParser(antfile)
    inst = InstrumentParser(insfile)
    return ant, inst


def get_calibration(cfg: Struct, bandwidth: Optional[float] = None) -> RslcCalibration:
    filename = cfg.dynamic_ancillary_file_group.external_calibration
    if filename is None:
        log.info("No calibration file provided.  Using defaults.")
        return RslcCalibration()
    log.info(f"Loading calibration file {filename} for bandwidth={bandwidth}")
    return parse_rslc_calibration(filename, bandwidth)


def get_identification_data_from_runconfig(cfg: Struct) -> dict:
    """
    Populate a dict containing the keys
        {"product_version", "processing_type", "composite_release_id",
        "mission_id", "processing_center", "track", "frame", "product_doi"}
    using data from an RSLC runconfig.
    """
    keys = ["product_version", "processing_type", "composite_release_id",
        "mission_id", "processing_center", "product_doi"]
    exe = vars(cfg.primary_executable)
    d = {key: exe[key] for key in keys}
    d["track"] = cfg.geometry.relative_orbit_number
    d["frame"] = cfg.geometry.frame_number
    return d


def get_identification_data_from_raw(rawlist: list[Raw]) -> dict:
    """
    Populate a dict containing the keys
        {"planned_datatake_id", "planned_observation_id", "is_urgent"}
    by combining the relevant identification metadata keys from all raw data
    files in the provided list.
    """
    return dict(
        # L0B always have a single entry
        planned_datatake_id = [raw.identification.plannedDatatake[0]
            for raw in rawlist],
        planned_observation_id = [raw.identification.plannedObservation[0]
            for raw in rawlist],
        is_urgent = any(raw.identification.isUrgentObservation
            for raw in rawlist),
        is_joint = any(raw.identification.isJointObservation
            for raw in rawlist)
    )


def set_algorithm_metadata(cfg: Struct, slc: SLC, is_dithered: bool = False):
    rfi = cfg.processing.radio_frequency_interference
    slc.set_algorithms(
        demInterpolation=cfg.processing.dem.interp_method,
        rfiDetection="ST-EVD" if rfi.detection_enabled else "disabled",
        rfiMitigation="ST-EVD" if rfi.mitigation_enabled else "disabled",
        elevationAntennaPatternCorrection=cfg.processing.is_enabled.eap,
        rangeSpreadingLossCorrection=cfg.processing.is_enabled.range_cor,
        azimuthPresumming="BLU" if is_dithered else "disabled")


def set_input_file_metadata(cfg: Struct, slc: SLC, runconfig_path: str = ""):
    anc = cfg.dynamic_ancillary_file_group
    value_or_blank = lambda x: x if x is not None else ""
    slc.set_inputs(
        l0bGranules=cfg.input_file_group.input_file_path,
        orbitFiles=[value_or_blank(anc.orbit)],
        attitudeFiles=[value_or_blank(anc.pointing)],
        auxcalFiles=[value_or_blank(x) for x in (anc.external_calibration,
            anc.internal_calibration, anc.antenna_pattern)],
        configFiles=[runconfig_path],
        demSource=value_or_blank(anc.dem_file_description))


def reduce_swath_parameters(rawlist: list[Raw],
        channel_out: PolChannel) -> tuple[float, float, float]:
    """Determine mixed-mode swath metadata.

    Parameters
    ----------
    rawlist : list[nisar.products.readers.Raw.Raw]
        List of L0B (raw) file reader objects.
    channel_out : nisar.mixed_mode.PolChannel
        Desired output polarimetric channel.

    Returns
    -------
    prf : float
        Minimum PRF (in Hz) among all intersecting input modes
    bandwidth : float
        Maximum bandwidth (in Hz) among all intersecting input modes
    center_frequency : float
        Maximum center frequency (in Hz) among all intersecting input modes
    """
    prfs, bandwidths, center_freqs = [], [], []
    for raw in rawlist:
        chan = find_overlapping_channel(raw, channel_out)
        prfs.append(raw.getNominalPRF(chan.freq_id, chan.txpol))
        bandwidths.append(raw.getRangeBandwidth(chan.freq_id, chan.txpol))
        center_freqs.append(raw.getCenterFrequency(chan.freq_id, chan.txpol))
    return min(prfs), max(bandwidths), max(center_freqs)


def get_output_range_spacings(rawlist: list[Raw], common_mode: PolChannelSet):
    """
    Get the output RSLC range spacing associated with each subband.  The
    spacings will be chosen from among the range spacings in the input L0B data.

    For example, if we do mixed mode processing of 20+5 and 40+5 data then the
    common mode will be 20+5 and the returned range spacings will be the same as
    the 20+5 L0B data.

    Usually NISAR data are oversampled by a factor of 1.2, but this is not the
    case for 77 MHz modes and may not be the case for other sensors.

    Parameters
    ----------
    rawlist : list[Raw]
        List of input L0B product readers.
    common_mode : PolChannelSet
        Set of PolChannel objects that will be generated in the RSLC.

    Returns
    -------
    range_spacings : dict[str, float]
        Range spacing in m for each subband.
    """
    # Get a PolChannel associated with the largest output bandwidth, e.g., a
    # 20 MHz channel if we're generating 20+5 output.  Also want the smallest
    # bandwidth channel.  If these are equal, e.g., 20+20 or 5+5 mode, make
    # sure we get one from each frequency.
    channels = sorted(common_mode,
        key = lambda channel: (channel.band.width, channel.freq_id))
    big_channel = channels[-1]
    small_channel = channels[0]
    # (These will be the same if there's no secondary band).

    range_spacings = dict()
    for channel in (small_channel, big_channel):
        # Find the raw data PolChannels associated with that output, e.g.,
        # corresponding to [20, 40, 20] MHz bands.
        raw_spacings = []
        for raw in rawlist:
            raw_channel = find_overlapping_channel(raw, channel)
            freq, tx = raw_channel.freq_id, raw_channel.pol[0]
            # Get the range spacing (sample rate) for the associated raw data.
            raw_spacings.append(raw.getRanges(freq, tx).spacing)

        # We're filtering everything down to the coarsest mode, so return the
        # max of these spacings, e.g., the one for bw=20 MHz
        # (where usually fs=24 MHz).
        range_spacings[channel.freq_id] = max(raw_spacings)

    return range_spacings


def get_focused_sub_swaths(rawlist, out_chan, grid, orbit, doppler, dem, azres,
                           rdr2geo_params=dict(), geo2rdr_params=dict(),
                           ignore_failure=False):
    """
    Determine fully-focused regions of the image in a format suitable for
    populating the validSamplesSubSwathX RSLC datasets.

    Parameters
    ----------
    rawlist : list[Raw]
        List of raw data files (observations) that will be processed.
    out_chan : PolChannel
        Desired channel to process (will be matched with available raw data
        using mixed-mode logic).
    grid : RadarGridParameters
        Grid for focused image (zero-Doppler).
    orbit : Orbit
        Trajectory of antenna phase center.  Its time span must cover the entire
        collection of raw data plus any reskew time offset between the native-
        and zero-Doppler radar coordinate systems.
    doppler : LUT2d
        Doppler centroid of raw data, in Hz.
    dem : DEMInterpolator
        Digital elevation model.
    azres : float
        Processed azimuth resolution, in meters.
    rdr2geo_params : dict
        Parameters for rdr2geo_bracket
    geo2rdr_params : dict
        Parameters for geo2rdr_bracket
    ignore_failure : bool
        If set to True and isce3.focus.get_focused_sub_swaths fails for any
        reason, then a mask corresponding to all-pixels-valid will be returned.
        Otherwise an exception will be raised on failures.  This can be useful
        for datasets where the orbit data covers all the raw data but without
        enough extra for the reskew to the zero-Doppler image grid.

    Returns
    -------
    swaths : numpy.ndarray[np.uint32]
        Array of [start, stop) valid data regions, shape = (nswath, npulse, 2)
        where nswath is the number of valid sub-swaths and npulse is the length
        of the focused image grid.
    """
    raw_bbox_lists = []
    chirp_durations = []
    for raw in rawlist:
        raw_chan = find_overlapping_channel(raw, out_chan)

        freq = raw_chan.freq_id
        bboxes = raw.getSubSwathBboxes(freq, epoch=orbit.reference_epoch)
        raw_bbox_lists.append(bboxes)

        txpol = raw_chan.pol[0]
        chirp_durations.append(raw.getChirpParameters(freq, txpol)[3])

    try:
        swaths = isce3.focus.get_focused_sub_swaths(raw_bbox_lists,
            chirp_durations, orbit, doppler, azres, grid, dem=dem,
            rdr2geo_params=rdr2geo_params, geo2rdr_params=geo2rdr_params)
    except Exception as e:
        if ignore_failure:
            log.error("Failed to calculate valid subswath masks!  "
                "The entire radar grid will be assumed valid.")
            swaths = np.zeros((1, grid.length, 2), dtype=np.uint32)
            swaths[..., 1] = grid.width
        else:
            raise e
    return swaths


def focus(runconfig, runconfig_path=""):
    # Strip off two leading namespaces.
    cfg = runconfig.runconfig.groups
    rawnames = cfg.input_file_group.input_file_path
    if len(rawnames) <= 0:
        raise IOError("need at least one raw data file")

    rawlist = [open_rrsd(os.path.abspath(name)) for name in rawnames]
    dem = get_dem(cfg)
    zerodop = isce3.core.LUT2d()
    azres = cfg.processing.azcomp.azimuth_resolution
    atmos = cfg.processing.dry_troposphere_model or "nodelay"
    kernel = get_kernel(cfg)
    scale = cfg.processing.encoding_scale_factor
    antparser, instparser = get_antpat_inst(cfg)

    common_mode = get_common_mode(rawlist)
    log.info(f"output mode = {common_mode}")

    use_gpu = isce3.core.gpu_check.use_gpu(cfg.worker.gpu_enabled, cfg.worker.gpu_id)
    if use_gpu:
        # Set the current CUDA device.
        device = isce3.cuda.core.Device(cfg.worker.gpu_id)
        isce3.cuda.core.set_device(device)

        log.info(f"Processing using CUDA device {device.id} ({device.name})")

        backproject = isce3.cuda.focus.backproject
    else:
        backproject = isce3.focus.backproject

    # Generate output grids.
    grid_epoch, t0, t1, r0, r1 = get_total_grid_bounds(rawnames)
    log.info(f"Raw data time spans [{t0}, {t1}] seconds since {grid_epoch}.")
    log.info(f"Raw data range swath spans [{r0}, {r1}] meters.")
    orbit = get_orbit(cfg)
    attitude = get_attitude(cfg)
    # Need orbit and attitude over whole raw domain in order to generate
    # Doppler LUT.  Check explicitly in order to provide a sensible error.
    log.info("Verifying ephemeris covers time span of raw data.")
    require_ephemeris_overlap(orbit, t0, t1, "Orbit")
    require_ephemeris_overlap(attitude, t0, t1, "Attitude")
    fc_ref, dop_ref = make_doppler(cfg, epoch=grid_epoch, orbit=orbit,
        attitude=attitude, dem=dem)

    max_chirplen = get_max_chirp_duration(cfg) * isce3.core.speed_of_light / 2
    range_spacings = get_output_range_spacings(rawlist, common_mode)
    dr = min(range_spacings.values())
    max_prf = get_max_prf(rawlist)
    side = require_constant_look_side(rawlist)
    ref_grid = make_output_grid(cfg, grid_epoch, t0, t1, max_prf, r0, r1, dr,
                                side, orbit, fc_ref, dop_ref, max_chirplen, dem)

    wvl_ref = isce3.core.speed_of_light / fc_ref
    el_lut, inc_lut, _ = make_los_luts(orbit, attitude, side, dop_ref, wvl_ref,
                                       dem, get_rdr2geo_params(cfg))
    beta0_lut, sigma0_lut, gamma0_lut = make_cal_luts(inc_lut)

    # Frequency A/B specific setup for output grid, doppler, and blocks.
    ogrid, dop, blocks_bounds = dict(), dict(), dict()
    for frequency, band in get_bands(common_mode).items():
        # Ensure aligned grids between A and B by just using an integer skip.
        # Sample rate of A is always an integer multiple of B for NISAR.
        rskip = int(np.round(range_spacings[frequency] / dr))
        ogrid[frequency] = ref_grid[:, ::rskip]
        ogrid[frequency].wavelength = isce3.core.speed_of_light / band.center
        log.info("Output grid %s is %s", frequency, ogrid[frequency])
        # Doppler depends on center frequency.
        dop[frequency] = scale_doppler(dop_ref, band.center / fc_ref)
        blocks_bounds[frequency] = plan_processing_blocks(cfg, ogrid[frequency],
                                        dop[frequency], dem, orbit)

    # NOTE SAR duration depends on frequency, so check all subbands.
    proc_begin, proc_end = total_bounds(list(chain(*blocks_bounds.values())))
    log.info(f"Need to process raw data time span [{proc_begin}, {proc_end}]"
             f" seconds since {grid_epoch} to produce requested output grid.")

    polygon = isce3.geometry.get_geo_perimeter_wkt(ref_grid, orbit,
                                                   zerodop, dem)

    output_slc_path = os.path.abspath(cfg.product_path_group.sas_output_file)

    output_dir = os.path.dirname(output_slc_path)
    os.makedirs(output_dir, exist_ok=True)

    product = cfg.primary_executable.product_type
    log.info(f"Creating output {product} product {output_slc_path}")
    helpers.validate_fs_page_size(cfg.output.fs_page_size, cfg.output.chunk_size)
    slc = SLC(output_slc_path, mode="w", product=product,
        fs_strategy=cfg.output.fs_strategy,
        fs_page_size=cfg.output.fs_page_size)
    slc.set_orbit(orbit)
    slc.set_attitude(attitude, orbit)

    id_data = get_identification_data_from_runconfig(cfg)
    id_data.update(get_identification_data_from_raw(rawlist))

    og = next(iter(ogrid.values()))
    start_time = og.sensing_datetime(0)
    end_time = og.sensing_datetime(og.length - 1)
    granule_id, is_full_frame, overlap = fill_partial_granule_id(
        cfg.primary_executable.partial_granule_id, common_mode, start_time,
        end_time, shapely.geometry.shape(json.loads(cfg.geometry.track_frame_polygon)),
        shapely.wkt.loads(polygon),
        coverage_threshold = cfg.geometry.full_coverage_threshold_percent / 100)

    is_dithered=any(raw.isDithered(raw.frequencies[0]) for raw in rawlist)
    slc.copy_identification(rawlist[0], polygon=polygon,
        start_time=start_time, end_time=end_time,
        frequencies=common_mode.frequencies,
        is_full_frame=is_full_frame, frame_coverage=overlap,
        coverage_threshold=cfg.geometry.full_coverage_threshold_percent / 100,
        is_dithered=is_dithered, granule_id=granule_id,
        is_mixed_mode=any(PolChannelSet.from_raw(raw) != common_mode
            for raw in rawlist),
        **id_data)
    set_algorithm_metadata(cfg, slc, is_dithered)
    set_input_file_metadata(cfg, slc, runconfig_path)

    # Get reference range for radiometric correction and warn user if it's not
    # a good value (especially since the default is catered to NISAR).
    ref_range = get_calibration(cfg).reference_range
    ratio = ref_range / og.mid_range
    if not (0.5 < ratio < 2.0) and cfg.processing.is_enabled.range_cor:
        log.warning(f"Reference range ({ref_range} m) is not within a factor "
            f"of two of the mid-swath range ({og.mid_range} m).  Range "
            "fading correction will impart a large scaling.")

    vs, _ = isce3.focus.get_radar_velocities(orbit)
    azenv = isce3.focus.predict_azimuth_envelope(azres, og.prf, vs,
        L=cfg.processing.nominal_antenna_size.azimuth)
    azimuth_bandwidth = vs / azres

    # store metadata for each frequency
    for frequency, band in get_bands(common_mode).items():
        rgres = isce3.core.speed_of_light / (2 * band.width)
        oversample = rgres / og.range_pixel_spacing
        slc.set_parameters(dop[frequency], orbit.reference_epoch, frequency,
            cfg.processing.range_window.kind, cfg.processing.range_window.shape,
            oversample, azenv, dump_config_str(cfg))
        og = ogrid[frequency]

        # Support nominal != processed parameters for mixed-mode case.
        pols = [chan.pol for chan in common_mode if chan.freq_id == frequency]
        chan = PolChannel(frequency, pols[0], band)
        acquired_prf, acquired_bw, acquired_fc = reduce_swath_parameters(
            rawlist, chan)

        log.info("computing valid swaths")
        valid_swaths = get_focused_sub_swaths(rawlist, chan, og, orbit,
            dop[frequency], dem, azres, rdr2geo_params=get_rdr2geo_params(cfg),
            geo2rdr_params=get_geo2rdr_params(cfg), ignore_failure=False)

        slc.update_swath(og, orbit, band.width, frequency,  azimuth_bandwidth,
            acquired_prf, acquired_bw, acquired_fc, valid_swaths)
        cal = get_calibration(cfg, band.width)
        slc.set_calibration(cal, frequency)

        # add calibration section for each polarization
        for pol in pols:
            slc.add_calibration_section(frequency, pol, og.sensing_times,
                                        orbit.reference_epoch, og.slant_ranges,
                                        beta0_lut, sigma0_lut, gamma0_lut)


    freq = next(iter(get_bands(common_mode)))
    slc.set_geolocation_grid(orbit, ogrid[freq], dop[freq],
                             epsg=cfg.processing.metadata_cube_epsg, dem=dem,
                             **get_geo2rdr_params(cfg, orbit))

    # Scratch directory for intermediate outputs
    scratch_dir = os.path.abspath(cfg.product_path_group.scratch_path)
    os.makedirs(scratch_dir, exist_ok=True)

    def temp(suffix):
        return tempfile.NamedTemporaryFile(dir=scratch_dir, suffix=suffix,
            delete=cfg.processing.delete_tempfiles)

    dump_height = (cfg.processing.debug_dump_height and
                   not cfg.processing.delete_tempfiles)

    if cfg.processing.dem.require_full_coverage:
        log.info("Checking DEM coverage.")
        fraction_outside = isce3.geometry.compute_dem_overlap(polygon, dem,
            plot=temp("_dem_overlap.png"))
        if fraction_outside > 0.0:
            percent_outside = f"{100 * fraction_outside:.1f}%"
            raise ValueError(f"{percent_outside} of the swath falls outside of "
                "the area covered by the DEM.  If you enabled tempfiles you "
                "can find a plot in the scratch directory.  You can disable "
                "this coverage check by setting dem.require_full_coverage to "
                "False in the runconfig.groups.processing section.")


    rfi_results = defaultdict(list)

    # main processing loop
    for channel_out in common_mode:
        frequency, pol = channel_out.freq_id, channel_out.pol
        log.info(f"Processing frequency{channel_out.freq_id} {channel_out.pol}")
        acdata = slc.create_image(frequency, pol, shape=ogrid[frequency].shape,
            **get_dataset_creation_options(cfg, ogrid[frequency].shape))
        deramp_ac = get_range_deramp(ogrid[frequency])
        writer = BackgroundWriter(scale * deramp_ac, acdata,
            cfg.output.data_type, mantissa_nbits=cfg.output.mantissa_nbits)

        # store noise powers and its azimuth times in containers
        # over all Raw files for a common band and pol.
        azt_noise_all = []
        pow_noise_all = []

        for raw in rawlist:
            channel_in = find_overlapping_channel(raw, channel_out)
            log.info("Using raw data channel %s", channel_in)
            cal = get_calibration(cfg, channel_in.band.width)
            check_cal_validity_dates(cal, raw.identification.zdStartTime,
                raw.identification.zdEndTime)

            # NOTE In some cases frequency != channel_in.freq_id, for example
            # 80 MHz (A) being mixed with 5 MHz sideband (B).
            rawdata = raw.getRawDataset(channel_in.freq_id, pol)
            log.info(f"Raw data shape = {rawdata.shape}")
            raw_times, raw_grid = raw.getRadarGrid(channel_in.freq_id,
                                                   tx=pol[0], epoch=grid_epoch)

            pulse_begin = bisect_left(raw_times, proc_begin)
            pulse_end = bisect_right(raw_times, proc_end)
            log.info(f"Using pulses [{pulse_begin}, {pulse_end}]")
            if pulse_begin >= pulse_end:
                log.info("Output does not depend on file %s", raw.filename)
                continue
            raw_times = raw_times[pulse_begin:pulse_end]
            raw_grid = raw_grid[pulse_begin:pulse_end, :]

            na = cfg.processing.rangecomp.block_size.azimuth
            nr = rawdata.shape[1]
            swaths = raw.getSubSwaths(channel_in.freq_id, tx=pol[0])
            log.info(f"Number of sub-swaths = {swaths.shape[0]}")

            rawfd = temp("_raw.c8")
            log.info(f"Decoding raw data to memory map {rawfd.name}.")
            raw_mm = np.memmap(rawfd, mode="w+", shape=raw_grid.shape,
                               dtype=np.complex64)
            if cfg.processing.zero_fill_gaps:
                log.info("Will fill gaps between sub-swaths with zeros.")

            for i in range(0, raw_grid.shape[0], na):
                pulse = i + pulse_begin
                nblock = min(na, rawdata.shape[0] - pulse, raw_mm.shape[0] - i)
                block_in = np.s_[pulse:pulse+nblock, :]
                block_out = np.s_[i:i+nblock, :]
                z = rawdata[block_in]
                # Remove NaNs.  TODO could incorporate into gap mask.
                z[np.isnan(z)] = 0.0
                if cfg.processing.zero_fill_gaps:
                    fill_gaps(z, swaths[:, pulse:pulse+nblock, :], 0.0)
                raw_mm[block_out] = z

            raw_clean, rfi_likelihood = process_rfi(cfg, raw_mm, temp)
            rfi_results[(frequency, pol)].append(
                (rfi_likelihood, raw_clean.shape[0]))
            del raw_mm, rawfd

            uniform_pri = not raw.isDithered(channel_in.freq_id)
            if uniform_pri:
                log.info("Uniform PRF, using raw data directly.")
                regridded, regridfd = raw_clean, None
            else:
                regridfd = temp("_regrid.c8")
                log.info(f"Resampling non-uniform raw data to {regridfd.name}.")
                regridded = resample(raw_clean, raw_times, raw_grid, swaths, orbit,
                                    dop[frequency], fn=regridfd,
                                    L=cfg.processing.nominal_antenna_size.azimuth)


            # Do range compression.
            rc, rc_grid, shift, deramp_rc = prep_rangecomp(cfg, raw, raw_grid,
                                        channel_in, channel_out, cal)

            # Precompute antenna patterns at downsampled spacing
            if cfg.processing.is_enabled.eap:
                antpat = AntennaPattern(raw, dem, antparser,
                                        instparser, orbit, attitude,
                                        el_lut=el_lut)

                log.info("Precomputing antenna patterns")
                i = np.arange(rc_grid.shape[0])
                ti = np.array(rc_grid.sensing_start + i / rc_grid.prf)

                spacing = cfg.processing.elevation_antenna_pattern.spacing
                span = rc_grid.slant_ranges[-1] - rc_grid.slant_ranges[0]
                nbins = math.ceil(span / spacing) + 1
                pat_ranges = isce3.core.Linspace(rc_grid.slant_ranges[0], spacing, nbins)
                patterns = antpat.form_pattern(
                    ti, pat_ranges, nearest=not uniform_pri, txrx_pols=[pol])

            fd = temp("_rc.c8")
            log.info(f"Writing range compressed data to {fd.name}")
            rcfile = Raster(fd.name, rc.output_size, rc_grid.shape[0], GDT_CFloat32)
            log.info(f"Range compressed data shape = {rcfile.data.shape}")

            # Compute NESZ if there exist noise-only range lines
            # get noise only range line indexes within processing interval
            cal_path_mask = raw.getCalType(
                channel_in.freq_id, pol[0])[pulse_begin:pulse_end]
            _, _, _, idx_noise = get_calib_range_line_idx(cal_path_mask)

            # form output slant range vector for all noise products
            if cfg.processing.noise_equivalent_backscatter.fill_nan_ends:
                nrgb_noise = cfg.processing.noise_equivalent_backscatter.num_range_block + 2
            else:
                log.warning('Noise powers will be non-uniform in range '
                            'with possible NaN values!')
                nrgb_noise = cfg.processing.noise_equivalent_backscatter.num_range_block
            sr_noise = np.linspace(
                    ogrid[frequency].slant_ranges.first,
                    ogrid[frequency].slant_ranges.last,
                    num=nrgb_noise
                    )
            if idx_noise.size == 0:
                log.warning(
                    'No noise-only range lines within the specified pulse '
                    'interval. Skip noise estimation and set noise equivalent '
                    'backscatter to zero.')
                pow_noise = np.zeros_like(sr_noise, dtype='f4')
            else: # there is at least one noise-only range line
                nrgl_noise = idx_noise.size
                log.info(f'Number of noise-only range lines is {nrgl_noise}')
                # create a dedicated memory map for noise data and processing.
                # set the number of range bins to rangecomp output size.
                fid_noise = temp("_noise.c8")
                data_noise = np.memmap(
                    fid_noise, mode='w+', shape=(nrgl_noise, rc.output_size),
                    dtype=np.complex64)
                rc.rangecompress(data_noise, raw_clean[idx_noise])
                # build and apply antenna pattern correction for noise
                # pulses if EAP is True
                if cfg.processing.is_enabled.eap:
                    log.info('Compensating dynamic antenna pattern for '
                             'noise-only range lines')
                    for pp in range(nrgl_noise):
                        tm = raw_times[idx_noise[pp]]
                        # get 2-way pattern for noise-only pulse time over
                        # a coarser slant range vector than that of echo.
                        # 2-way antenna pattern in EAP correction is slightly
                        # different for noise-only range lines than for
                        # re-grided echo range lines in dithering case
                        # w/ BLU interpolator. This is why the process
                        # is repeated here for noise estimation.
                        pat2w = antpat.form_pattern(
                            tm, pat_ranges, nearest=False, txrx_pols=[pol])
                        # interpolated pattern to a finer range spacing
                        pat_int = np.interp(
                            rc_grid.slant_ranges, pat_ranges, pat2w[pol])
                        # apply antenna pattern, replace bad value by NaN
                        mask_zero = np.isclose(abs(pat_int), 0)
                        mask_non_zero = ~mask_zero
                        data_noise[pp, mask_non_zero] /= pat_int[mask_non_zero]
                        data_noise[pp, mask_zero] = complex(np.nan, np.nan)

                if cfg.processing.is_enabled.range_cor:
                    log.info('Compensating range loss for noise range lines')
                    data_noise *= np.array(rc_grid.slant_ranges) / ref_range

                # correct for non-unity AZ Reference function given noise
                # data will not be run through AZ compression.
                # get SAR duration at the starting slant range and mid noise
                # range line.
                tm_mid_noise = raw_times[idx_noise[nrgl_noise // 2]]
                sar_dur_near = isce3.focus.get_sar_duration(
                    tm_mid_noise, rc_grid.starting_range, orbit,
                    isce3.core.Ellipsoid(), azres, rc_grid.wavelength
                    )
                n_pulse_sar =  sar_dur_near * rc_grid.prf
                # Multiply by the number of pulses within a SAR at the nearest
                # range and apply the encoding scalar from the config file
                data_noise *= scale * np.sqrt(
                    n_pulse_sar * np.array(rc_grid.slant_ranges) /
                    rc_grid.starting_range)
                # perform noise estimation
                # get valid subswath for noise-only range lines
                idx_noise_abs = pulse_begin + np.asarray(idx_noise)
                sbsw_noise = swaths[:, idx_noise_abs]
                pow_noise, sr_noise_rc = est_noise_power_in_focus(
                    data_noise, rc_grid.slant_ranges, sbsw_noise,
                    logger=log,
                    **vars(cfg.processing.noise_equivalent_backscatter)
                )
                # regrid noise power to match common output grid in range.
                # Use linear interpolation given small changes in slant range
                # and smoothed noise power if `fill_nan_end`.
                pow_noise[...] = np.interp(sr_noise, sr_noise_rc, pow_noise)
                del data_noise
            # store noise power and its AZ time for a particular Raw
            # over a particular AZ interval.
            # Use the very first and last AZ time stamp of the Raw.
            # Note that there is only one unique vector of noise power
            # as a function of slant range per a Raw file. Thus,
            # the noise power shall be repeated twice per L0B!
            azt_noise_all.extend(raw_times[::raw_times.size - 1])
            pow_noise_all.extend(2 * [pow_noise])

            del raw_clean

            # And do radiometric corrections at the same time.
            for pulse in range(0, rc_grid.shape[0], na):
                log.info(f"Range compressing block at pulse {pulse}")
                block = np.s_[pulse:pulse+na, :]
                rc.rangecompress(rcfile.data[block], regridded[block])
                if abs(shift) > 0.0:
                    log.info("Shifting mixed-mode data to baseband")
                    rcfile.data[block] *= deramp_rc[np.newaxis,:]
                if cfg.processing.is_enabled.eap:
                    log.info("Compensating dynamic antenna pattern")
                    for i in range(rc_grid[block].shape[0]):
                        interp_pattern = np.interp(rc_grid.slant_ranges, pat_ranges, patterns[pol][pulse + i])
                        rcfile.data[pulse + i, :] /= interp_pattern
                if cfg.processing.is_enabled.range_cor:
                    log.info("Compensating range loss")
                    # Two-way power drops with R^4, so amplitude drops with R^2.
                    # Synthetic aperture length and thus signal gain scales
                    # with R (not compensated in `backproject`).  So we just
                    # have to scale by R to compensate the range fading.
                    rcfile.data[block] *= np.array(rc_grid.slant_ranges) / ref_range


            del regridded, regridfd

            if dump_height:
                fd_hgt = temp(f"_height_{frequency}{pol}.f4")
                shape = ogrid[frequency].shape
                hgt_mm = np.memmap(fd_hgt, mode="w+", shape=shape, dtype='f4')
                log.debug(f"Dumping height to {fd_hgt.name} with shape {shape}")

            # Do azimuth compression.
            igeom = isce3.container.RadarGeometry(rc_grid, orbit, dop[frequency])

            for block, (t0, t1) in blocks_bounds[frequency]:
                description = f"(i, j) = ({block[0].start}, {block[1].start})"
                if not cfg.processing.is_enabled.azcomp:
                    continue
                if not is_overlapping(t0, t1,
                                    rc_grid.sensing_start, rc_grid.sensing_stop):
                    log.info(f"Skipping inactive azcomp block at {description}")
                    continue
                log.info(f"Azcomp block at {description}")
                bgrid = ogrid[frequency][block]
                ogeom = isce3.container.RadarGeometry(bgrid, orbit, zerodop)
                z = np.zeros(bgrid.shape, 'c8')
                hgt = hgt_mm[block] if dump_height else None
                err = backproject(z, ogeom, rcfile.data, igeom, dem,
                            channel_out.band.center, azres,
                            kernel, atmos, get_rdr2geo_params(cfg),
                            get_geo2rdr_params(cfg, orbit), height=hgt)
                if err:
                    log.warning("azcomp block contains some invalid pixels")
                writer.queue_write(z, block)

            # Raster/GDAL creates a .hdr file we have to clean up manually.
            hdr = fd.name.replace(".c8", ".hdr")
            if cfg.processing.delete_tempfiles:
                delete_safely(hdr)
            del rcfile

            if dump_height:
                del fd_hgt, hgt_mm

        writer.notify_finished()
        log.info(f"Image statistics {frequency} {pol} = {writer.stats}")
        slc.write_stats(frequency, pol, writer.stats)
        slc.set_rfi_results(rfi_results)

        # Dump the noise product for a certain band and pol over entire
        # AZ times covering all Raw files.
        noise_prod = NoiseEquivalentBackscatterProduct(
            np.asarray(pow_noise_all), sr_noise, np.asarray(azt_noise_all),
            grid_epoch, frequency, pol
            )
        # dump the noise product into RSLC product
        slc.set_noise(noise_prod)
        del pow_noise_all, azt_noise_all, sr_noise

    log.info("All done!")


def configure_logging():
    log_level = logging.DEBUG
    log.setLevel(log_level)
    # Format from L0B PGE Design Document, section 9.  Kludging error code.
    msgfmt = ('%(asctime)s.%(msecs)03d, %(levelname)s, RSLC, %(module)s, '
        '999999, %(pathname)s:%(lineno)d, "%(message)s"')
    fmt = logging.Formatter(msgfmt, "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    for friend in ("Raw", "SLCWriter", "nisar.antenna.pattern", "rslc_cal",
                   "isce3.focus.notch"):
        l = logging.getLogger(friend)
        l.setLevel(log_level)
        l.addHandler(sh)


def main(argv):
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    configure_logging()
    cfg = validate_config(load_config(args.run_config_path))
    echofile = cfg.runconfig.groups.product_path_group.sas_config_file
    if echofile:
        log.info(f"Logging configuration to file {echofile}.")
        with open(echofile, "w") as f:
            dump_config(cfg, f)
    focus(cfg, args.run_config_path)


if __name__ == '__main__':
    main(sys.argv[1:])
