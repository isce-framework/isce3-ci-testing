#!/usr/bin/env python3
"""
Generate El Null Range product from L0B (DM2) + Antenna HDF5 + [DEM raster]
data which will be used for pointing analyses by D&C team
"""
from __future__ import annotations
import os
import time
import argparse as argp
from datetime import datetime, timezone

import numpy as np

from nisar.products.readers.Raw import Raw
from nisar.products.readers.antenna import AntennaParser
from isce3.geometry import DEMInterpolator
from isce3.io import Raster
from nisar.pointing import el_null_range_from_raw_ant
from nisar.log import set_logger
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.products.readers.attitude import load_attitude_from_xml
from nisar.workflows.helpers import copols_or_desired_pols_from_raw


def cmd_line_parser():
    """Parse command line input arguments.

    Notes
    -----
    It also allows parsing arguments via an ASCII file
    by using prefix char "@".

    Returns
    -------
    argparse.Namespace

    """
    prs = argp.ArgumentParser(
        description=('Generate EL Null-Range product from L0B (DM2) + Antenna '
                     'HDF5 + [DEM raster] data'),
        formatter_class=argp.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@"
    )
    prs.add_argument('--l0b', type=str, required=True, dest='l0b_file',
                     help='Filename of HDF5 L0B product, Diagnostic Mode # 2')
    prs.add_argument('--ant', type=str, required=True, dest='antenna_file',
                     help='Filename of HDF5 Antenna product')
    prs.add_argument('-d', '--dem_file', type=str, dest='dem_file',
                     help='DEM raster file in (GDAL-compatible format such as '
                     'GeoTIFF) containing heights w.r.t. WGS-84 ellipsoid. '
                     'Default is no DEM!')
    prs.add_argument('-f', '--freq', type=str, choices=['A', 'B'],
                     dest='freq_band',
                     help='Frequency band such as "A". If set, the products '
                     'over desired `txrx_pol` or over all co-pols will be '
                     'processed. Otherwise, either all frequency bands with '
                     'desired `txrx_pol` or with all co-pols will be '
                     'processed (default)!')
    prs.add_argument('-p', '--pol', type=str, dest='txrx_pol',
                     choices=['HH', 'VV', 'HV', 'VH', 'RH', 'RV', 'LH', 'LV'],
                     help=('TxRx Polarization such as "HH". If set, the '
                           'products either over specified `freq_band` or over'
                           ' all available bands will be processed. Otherwise,'
                           ' the first co-pol of the specified `freq_band` or '
                           'all co-pols over all frequency bands will be '
                           'processed (default)!')
                     )
    prs.add_argument('--orbit', type=str, dest='orbit_file',
                     help='Filename of an external orbit XML file. The orbit '
                     'data will be used in place of those in L0B. Default is '
                     'orbit data stored in L0B.')
    prs.add_argument('--attitude', type=str, dest='attitude_file',
                     help='Filename of an external attitude XML '
                     'file. The attitude data will be used in place of those '
                     'in L0B. Default is attitude data stored in L0B.')
    prs.add_argument('-a', '--az_block_dur', type=float, dest='az_block_dur',
                     default=3.0, help='Duration of azimuth block in seconds.'
                     ' This value will be limited by total azimuth duration.'
                     ' The value must be equal or larger than the mean PRI.')
    prs.add_argument('-o', '--out', type=str, dest='out_path', default='.',
                     help='Output directory to dump EL null product. The '
                     'product is CSV file whose name conforms to JPL D-104976.'
                     )
    prs.add_argument('--caltone', action='store_true', dest='apply_caltone',
                     help=('Balance out RX channels by applying caltone '
                           'coefficients extracted from L0B.')
                     )
    prs.add_argument('-r', '--ref_height', type=float, dest='ref_height',
                     default=0.0,
                     help=('Reference height in (m) w.r.t WGS84 ellipsoid. It '
                           'will be simply used if "dem_file" is not provided')
                     )
    prs.add_argument('--plot', action='store_true', dest='plot',
                     help='Plot null power patterns (antenna v.s. echo) and '
                     'save them in *.png files at the specified output path')
    prs.add_argument('--deg', type=int, dest='polyfit_deg',
                     default=6, help='Degree of the polyfit used for'
                     ' smoothing and location estimation of echo null.')
    prs.add_argument('--exclude-nulls', type=int, nargs='*',
                     dest='exclude_nulls',
                     help=('List of excluded nulls, in the range [1, N-1], '
                           'where N is the number beams or RX channels. '
                           'The respective quality factors are simply '
                           'set to zero!')
                     )
    prs.add_argument('--sample-delays', type=int, nargs='*',
                     help=('Relative integer sample delays of right RX '
                           'channel wrt left one in ascending RX order for '
                           'all null pairs of either the selected frequency '
                           'band ("A" or "B") or the very first of two ("A") '
                           'if split spectrum. The number of delays shall be '
                           'equal to the total number of nulls.')
                     )
    prs.add_argument('--sample-delays2', type=int, nargs='*',
                     help=('Relative integer sample delays of right RX '
                           'channel wrt left one in ascending RX order for '
                           'all null pairs of the second band ("B") if split '
                           'spectrum and both bands are processed. The number '
                           'of delays shall be equal to the total number of '
                           'nulls.')
                     )
    prs.add_argument('--amp-ratio-imbalances', type=float, nargs='*',
                     help=('Amplitude ratio (linear) of right to left RX '
                           'channels of all null pairs in ascending RX order. '
                           'The size shall be equal to the number of nulls. '
                           'This will be applied to all processed frequency '
                           'bands. This is an external and separate '
                           'correction from that of caltone.')
                     )
    prs.add_argument('--phase-diff-imbalances', type=float, nargs='*',
                     help=('Phase difference (degrees) of right and left RX '
                           'channels of all null pairs in ascending order. '
                           'The size shall be equal to the number of nulls. '
                           'This will be applied to all processed frequency '
                           'bands. This is an external and separate '
                           'correction from that of caltone.')
                     )
    return prs.parse_args()


def gen_el_null_range_product(args):
    """Generate EL Null-Range Product.

    It generates Null locations in elevation (EL) direction as a function
    slant range at various azimuth/pulse times and dump them into a CSV file.

    Parameters
    ----------
    args : argparse.Namespace
        All input arguments parsed from a command line or an ASCII file.

    Notes
    -----
    The format of the file and output filename convention is defined
    in reference [1]_.

    References
    ----------
    .. [1] D. Kannapan, "D&C Radar Data Product SIS," JPL D-104976,
        December 3, 2020.

    """
    # Const
    prefix_name_csv = 'NISAR_ANC'
    # operation mode, whether DBF (single or a composite channel) or
    # 'DM2' (multi-channel). Null only possible via "DM2"!
    op_mode = 'DM2'

    tic = time.time()
    # set logging
    logger = set_logger("ELNullRangeProduct")

    # build Raw object
    raw_obj = Raw(hdf5file=args.l0b_file)
    # get the SAR band char
    sar_band_char = raw_obj.sarBand
    logger.info(f'SAR band char -> {sar_band_char}')

    # build antenna object from antenna file
    ant_obj = AntennaParser(args.antenna_file)

    # build dem interp object from DEM raster or ref height
    if args.dem_file is None:  # set to a fixed height
        dem_interp_obj = DEMInterpolator(args.ref_height)
    else:  # build from DEM Raster file
        dem_interp_obj = DEMInterpolator(Raster(args.dem_file))

    # build orbit and attitude object if external files are provided
    if args.orbit_file is None:
        orbit = None
    else:
        logger.info('Parsing external orbit XML file')
        orbit = load_orbit_from_xml(args.orbit_file)

    if args.attitude_file is None:
        attitude = None
    else:
        logger.info('Parsing external attitude XML file')
        attitude = load_attitude_from_xml(args.attitude_file)

    # get common keyword args for function "el_null_range_from_raw_ant"
    kwargs = {key: val for key, val in vars(args).items() if
              key in ['az_block_dur', 'apply_caltone', 'plot',
                      'out_path', 'polyfit_deg']}

    # logic for frequency band and TxRx polarization choices.
    # form a new dict "frq_pol" with key=freq_band and value=[txrx_pol]
    frq_pol = copols_or_desired_pols_from_raw(
        raw_obj, args.freq_band, args.txrx_pol)
    logger.info(f'List of selected frequency bands and TxRx Pols -> {frq_pol}')

    # check whether there are more than one frequency band
    # when "sample_delays2" is provided.
    if args.sample_delays2 is not None and len(frq_pol) == 1:
        logger.warning('Input "sample-delays2" will be ignored given '
                       'simply one frequency band will be processed!')

    exclude_nulls = args.exclude_nulls
    if exclude_nulls is not None:
        exclude_nulls = set(exclude_nulls)

    # build complex imbalance ratio only if either of
    # amp or phase is provided
    rx_imbalances = None
    if args.amp_ratio_imbalances is not None:
        rx_imbalances = np.asarray(args.amp_ratio_imbalances, dtype='c8')
    if args.phase_diff_imbalances is not None:
        imb_phs = np.exp(1j * np.deg2rad(args.phase_diff_imbalances))
        if rx_imbalances is None:
            rx_imbalances = imb_phs
        else:
            if imb_phs.size != rx_imbalances.size:
                raise ValueError(
                    f'Size mismatch between amplitude {rx_imbalances.size} '
                    f'and phase {imb_phs.size} imbalances.'
                )
            rx_imbalances *= imb_phs

    # loop over all desired frequency bands and their respective desired
    # polarizations
    sample_delays_all = [args.sample_delays, args.sample_delays2]
    for freq_band, sample_delays in zip(sorted(frq_pol), sample_delays_all):
        for txrx_pol in frq_pol[freq_band]:
            # check if the product is so-called noise-only (NO TX).
            # If no TX then skip that product.
            if raw_obj.is_tx_off(freq_band, txrx_pol):
                logger.warning(
                    f'Skip no-TX product ({freq_band},{txrx_pol})!')
                continue
            (null_num, sr_echo, el_ant, pow_ratio, az_datetime, null_flag,
             mask_valid, _, wavelength) = el_null_range_from_raw_ant(
                 raw_obj, ant_obj, dem_interp=dem_interp_obj, logger=logger,
                 orbit=orbit, attitude=attitude, freq_band=freq_band,
                 txrx_pol=txrx_pol, sample_delays_wrt_left=sample_delays,
                 imbalances_right2left=rx_imbalances, **kwargs
            )
            # check the excluded nulls whose quality factor will be zeroed out
            list_nulls = np.unique(null_num)
            logger.info(f'List of nulls -> {list_nulls}')
            if exclude_nulls is not None:
                logger.info('List of excluded nulls w/ zero quality '
                            f'factors -> {exclude_nulls}')
                if not exclude_nulls.issubset(list_nulls):
                    logger.warning(f'Excluded nulls {exclude_nulls} is out '
                                   f'of range of {list_nulls}.')
                    exclude_nulls.intersection_update(list_nulls)
                    logger.warning(
                        f'Updated list of excluded nulls -> {exclude_nulls}')
            # get the first and last utc azimuth time w/o fractional seconds
            # in "%Y%m%dT%H%M%S" format to be used as part of CSV product
            # filename.
            dt_utc_first = dt2str(az_datetime[0])
            dt_utc_last = dt2str(az_datetime[-1])
            # get current time w/o fractional seconds in "%Y%m%dT%H%M%S" format
            # used as part of CSV product filename
            dt_utc_cur = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')

            # naming convention of CSV file and product spec is defined in Doc:
            # See reference [1]
            name_csv = (
                f'{prefix_name_csv}_{sar_band_char}{freq_band}_{op_mode}_NULL_'
                f'{txrx_pol}_{dt_utc_cur}_{dt_utc_first}_{dt_utc_last}.csv'
            )
            file_csv = os.path.join(args.out_path, name_csv)
            logger.info(
                'Dump EL Null-Range product in "CSV" format to file ->\n '
                f'{file_csv}')
            # dump product into CSV file
            with open(file_csv, 'wt') as fid_csv:
                fid_csv.write('UTC Time,Band,Null Number,Range (m),Elevation'
                              ' (deg),Quality Factor\n')
                # report null-only product (null # > 0) w/ quality checking
                # afterwards
                for nn, null_val in enumerate(null_num):
                    quality_factor = 1 - pow_ratio[nn]
                    # Simply zero out quality factors for excluded nulls.
                    if exclude_nulls is not None:
                        if null_val in exclude_nulls:
                            quality_factor *= 0
                    fid_csv.write(
                        '{:s},{:1s},{:d},{:.3f},{:.3f},{:.3f}\n'.format(
                            az_datetime[nn].isoformat_usec(), sar_band_char,
                            null_num[nn], sr_echo[nn], el_ant[nn],
                            quality_factor)
                    )
                    # report possible invalid items/Rows
                    # add three for header line + null_zero + 0-based index to
                    # "nn"
                    if not mask_valid[nn]:
                        logger.warning(
                            f'Row # {nn + 2} may have invalid slant range '
                            'due to TX gap overlap!')
                    if not null_flag[nn]:
                        logger.warning(
                            f'Row # {nn + 3} may have invalid slant range due'
                            ' to failed convergence in null location '
                            'estimation!'
                        )

    # total elapsed time
    logger.info(f'Elapsed time -> {time.time() - tic:.1f} (sec)')


def dt2str(dt: 'isce3.core.DateTime', fmt: str = '%Y%m%dT%H%M%S') -> str:
    """isce3 DateTime to a desired string format."""
    return datetime.fromisoformat(dt.isoformat().split('.')[0]).strftime(fmt)


if __name__ == "__main__":
    """Main driver"""
    gen_el_null_range_product(cmd_line_parser())
