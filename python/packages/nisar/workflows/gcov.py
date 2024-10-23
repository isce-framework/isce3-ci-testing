#!/usr/bin/env python3

'''
collection of functions for NISAR GCOV workflow
'''

import datetime
import os
import tempfile
import time
from osgeo import gdal
import h5py
import journal
import numpy as np

import isce3
from isce3.core import crop_external_orbit
from isce3.core.types import complex32, read_complex_dataset
from nisar.products.readers import SLC
from nisar.workflows.h5_prep import add_radar_grid_cubes_to_hdf5
from isce3.atmosphere.tec_product import (tec_lut2d_from_json_srg,
                                          tec_lut2d_from_json_az)
from nisar.workflows.yaml_argparse import YamlArgparse
from nisar.workflows.gcov_runconfig import GCOVRunConfig
import nisar.workflows.helpers as helpers
from nisar.products.readers.orbit import load_orbit_from_xml
from nisar.products.writers.BaseL2WriterSingleInput import get_file_extension
from nisar.products.writers.GcovWriter import GcovWriter, run_geocode_cov


def read_rslc_backscatter(ds: h5py.Dataset, key, flag_rslc_is_complex32):
    """
    Read a complex HDF5 dataset and return the square of its magnitude

    Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32 conversions
    which are about 10x faster than HDF5 ones.

    Parameters
    ----------
    ds: h5py.Dataset
        Path to RSLC HDF5 dataset
    key: numpy.s_
        Numpy slice to subset input dataset
    flag_rslc_is_complex32: bool
        Flag indicating whether the RSLC should be read as CFloat16. Otherwise, the
        RSLC data will be read as CFloat32.

    Returns
    -------
    numpy.ndarray(numpy.float32)
        RSLC backscatter as a numpy.ndarray(numpy.float32)
    """
    if not flag_rslc_is_complex32:
        backscatter = np.absolute(ds[key][:]) ** 2
        return backscatter

    # This context manager handles h5py exception:
    # TypeError: data type '<c4' not understood
    data_block_c4 = ds.astype(complex32)[key]

    # backscatter = sqrt(r^2 + i^2) ^ 2
    #             = r^2 + i^2
    backscatter = data_block_c4['r'].astype(np.float32) ** 2
    backscatter += data_block_c4['i'].astype(np.float32) ** 2
    return backscatter


def prepare_rslc(in_file, freq, pol, out_file, lines_per_block,
                 flag_rslc_to_backscatter, pol_2=None, format="ENVI"):
    '''
    Copy RSLC dataset to GDAL format converting RSLC real and
    imaginary parts from float16 to float32. If the flag
    flag_rslc_to_backscatter is enabled, the
    RSLC complex values are converted to radar backscatter (square of
    the RSLC magnitude): out = abs(RSLC)**2

    Optionally, the output raster can be created from the
    symmetrization of two polarimetric channels
    (`pol` and `pol_2`).

    Parameters
    ----------
    in_file: str
        Path to RSLC HDF5
    freq: str
        RSLC frequency band to process ('A' or 'B')
    pol: str
        RSLC polarization to process
    out_file: str
        Output filename
    lines_per_block: int
        Number of lines per block
    flag_rslc_to_backscatter: bool
        Indicates if the RSLC complex values should be
        convered to radar backscatter (square of
        the RSLC magnitude)
    pol_2: str, optional
        Polarization associated with the second RSLC
    format: str, optional
        GDAL-friendly format

    Returns
    -------
    isce3.io.Raster
        output raster object
    '''
    info_channel = journal.info("prepare_rslc")

    # open RSLC HDF5 file dataset
    rslc = SLC(hdf5file=in_file)
    flag_rslc_is_complex32 = rslc.is_dataset_complex32(freq, pol)
    flag_symmetrize = pol_2 is not None

    pol_dataset_path = rslc.slcPath(freq, pol)
    info_channel.log(f'preparing dataset: {pol_dataset_path}')
    info_channel.log('    channel is complex32 (2 x 16bits):'
                     f' {flag_rslc_is_complex32}')
    info_channel.log('    requires polarimetric symmetrization:'
                     f' {flag_symmetrize}')

    pol_ref = f'HDF5:{in_file}:/{pol_dataset_path}'

    # If there's no need for pre-processing, return Raster of `pol_ref`
    if (not flag_rslc_is_complex32 and not flag_rslc_to_backscatter and
            not flag_symmetrize):
        return isce3.io.Raster(pol_ref)

    # If RSLC is already CFloat32 and the polarimetric symmetrization needs
    # to be applied coherently (i.e., using complex values), use ISCE3 C++
    # symmetrization for faster processing
    if (not flag_rslc_is_complex32 and not flag_rslc_to_backscatter and
            flag_symmetrize):
        info_channel.log('    symmetrizing cross-polarimetric channels'
                         ' coherently')
        pol_2_ref = f'HDF5:{in_file}:/{rslc.slcPath(freq, pol_2)}'

        hv_raster_obj = isce3.io.Raster(pol_ref)
        vh_raster_obj = isce3.io.Raster(pol_2_ref)

        # create output symmetrized HV object
        symmetrized_hv_obj = isce3.io.Raster(
            out_file,
            hv_raster_obj.width,
            hv_raster_obj.length,
            hv_raster_obj.num_bands,
            hv_raster_obj.datatype(),
            'GTiff')

        isce3.polsar.symmetrize_cross_pol_channels(
            hv_raster_obj,
            vh_raster_obj,
            symmetrized_hv_obj)

        return symmetrized_hv_obj

    hdf5_ds = rslc.getSlcDataset(freq, pol)

    # if provided, open RSLC HDF5 file dataset 2
    if flag_symmetrize:
        hdf5_ds_2 = rslc.getSlcDataset(freq, pol_2)

    # get RSLC dimension through GDAL
    gdal_ds = gdal.Open(pol_ref)
    rslc_length, rslc_width = gdal_ds.RasterYSize, gdal_ds.RasterXSize

    if flag_rslc_to_backscatter:
        gdal_dtype = gdal.GDT_Float32
    else:
        gdal_dtype = gdal.GDT_CFloat32

    # create output file
    driver = gdal.GetDriverByName(format)
    out_ds = driver.Create(out_file, rslc_width, rslc_length, 1, gdal_dtype)

    # start block processing
    lines_per_block = min(rslc_length, lines_per_block)
    num_blocks = int(np.ceil(rslc_length / lines_per_block))

    for block in range(num_blocks):
        line_start = block * lines_per_block
        if block == num_blocks - 1:
            block_length = rslc_length - line_start
        else:
            block_length = lines_per_block

        # read a block of data from the RSLC
        if flag_rslc_to_backscatter:
            data_block = read_rslc_backscatter(
                hdf5_ds, np.s_[line_start:line_start + block_length, :],
                flag_rslc_is_complex32)
        else:
            data_block = read_complex_dataset(
                hdf5_ds, np.s_[line_start:line_start + block_length, :])

        if flag_symmetrize:
            # compute average with a block of data from the second RSLC
            if flag_rslc_to_backscatter:
                data_block += read_rslc_backscatter(
                    hdf5_ds_2, np.s_[line_start:line_start + block_length, :],
                    flag_rslc_is_complex32)
            else:
                data_block += read_complex_dataset(
                    hdf5_ds_2, np.s_[line_start:line_start + block_length, :])
            data_block /= 2.0

        # write to GDAL raster
        out_ds.GetRasterBand(1).WriteArray(data_block, yoff=line_start, xoff=0)

    out_ds.FlushCache()
    return isce3.io.Raster(out_file)


def run(cfg):
    '''
    run GCOV

    Parameters
    ----------
    cfg : dict
        Dictionary containing processing parameters

    Returns
    -------
    output_files_list: list(str)
        List of output files

    '''
    scratch_path = cfg['product_path_group']['scratch_path']
    with tempfile.TemporaryDirectory(dir=scratch_path) \
            as raster_scratch_dir:
        output_files_list = _run(cfg, raster_scratch_dir)
        return output_files_list


def _run(cfg, raster_scratch_dir):
    '''
    run GCOV with a scratch directory

    Parameters
    ----------
    cfg : dict
        Dictionary containing processing parameters
    raster_scratch_dir: str
        Scratch directory

    Returns
    -------
    output_files_list: list(str)
        List of output files

    '''
    info_channel = journal.info("gcov.run")
    info_channel.log("Starting GCOV workflow")

    # pull parameters from cfg
    input_hdf5 = cfg['input_file_group']['input_file_path']
    output_hdf5 = cfg['product_path_group']['sas_output_file']
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']
    flag_fullcovariance = cfg['processing']['input_subset']['fullcovariance']
    flag_symmetrize_cross_pol_channels = \
        cfg['processing']['input_subset']['symmetrize_cross_pol_channels']

    # Retrieve file spacing params
    file_spacing_kwargs = cfg['output']
    fs_strategy = file_spacing_kwargs["fs_strategy"]
    fs_page_size = file_spacing_kwargs["fs_page_size"]

    # Initialize h5py open mode to 'write' to allow file spacing strategy to
    # be set - can only be done in write mode. After file created, open mode
    # will be changed to 'append' to allow h5py.File to work within the
    # frequency iteration loop it is nested in.
    h5_write_mode = 'w'

    output_gcov_terms_kwargs = cfg['output']['output_gcov_terms']
    output_secondary_layers_kwargs = cfg['output']['output_secondary_layers']

    # Sanity check page size and chunk size
    helpers.validate_fs_page_size(fs_page_size,
                                  output_gcov_terms_kwargs["chunk_size"])

    # Raster files format (output of GeocodeCov).
    # Cannot use HDF5 because we cannot save multiband HDF5 datasets
    # simultaneously, therefore we use "ENVI" instead
    output_gcov_terms_raster_files_format = \
        output_gcov_terms_kwargs['format']
    if output_gcov_terms_raster_files_format == 'HDF5':
        output_gcov_terms_raster_files_format = 'ENVI'
    secondary_layer_files_raster_files_format = \
        output_secondary_layers_kwargs['format']
    if secondary_layer_files_raster_files_format == 'HDF5':
        secondary_layer_files_raster_files_format = 'ENVI'

    # Set raster file extensions for GCOV terms and secondary layers
    gcov_terms_file_extension = get_file_extension(
        output_gcov_terms_raster_files_format)
    secondary_layers_file_extension = get_file_extension(
        secondary_layer_files_raster_files_format)

    radar_grid_cubes_geogrid = cfg['processing']['radar_grid_cubes']['geogrid']
    radar_grid_cubes_heights = cfg['processing']['radar_grid_cubes']['heights']

    # DEM parameters
    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']

    orbit_file = cfg["dynamic_ancillary_file_group"]['orbit_file']
    tec_file = cfg["dynamic_ancillary_file_group"]['tec_file']

    # unpack geocode run parameters
    geocode_dict = cfg['processing']['geocode']

    apply_range_ionospheric_delay_correction = \
        geocode_dict['apply_range_ionospheric_delay_correction']

    apply_azimuth_ionospheric_delay_correction = \
        geocode_dict['apply_azimuth_ionospheric_delay_correction']

    apply_valid_samples_sub_swath_masking = \
        geocode_dict['apply_valid_samples_sub_swath_masking']
    geogrid_upsampling = geocode_dict['geogrid_upsampling']
    abs_cal_factor = geocode_dict['abs_rad_cal']
    clip_max = geocode_dict['clip_max']
    clip_min = geocode_dict['clip_min']
    geogrids = geocode_dict['geogrids']
    flag_upsample_radar_grid = geocode_dict['upsample_radargrid']
    save_mask = geocode_dict['save_mask']
    min_block_size_mb = cfg["processing"]["geocode"]['min_block_size']
    max_block_size_mb = cfg["processing"]["geocode"]['max_block_size']

    # optional keyword arguments , i.e. arguments that may or may not be
    # included in the call to geocode()
    optional_geo_kwargs = {}

    # declare list of output and temporary files
    output_files_list = [output_hdf5]

    output_dir = os.path.dirname(output_hdf5)
    output_gcov_terms_kwargs['output_dir'] = output_dir
    output_secondary_layers_kwargs['output_dir'] = output_dir
    output_gcov_terms_kwargs['output_files_list'] = output_files_list
    output_secondary_layers_kwargs['output_files_list'] = output_files_list

    # read min/max block size converting MB to B
    if min_block_size_mb is not None:
        optional_geo_kwargs['min_block_size'] = min_block_size_mb * (2**20)
    if max_block_size_mb is not None:
        optional_geo_kwargs['max_block_size'] = max_block_size_mb * (2**20)

    # unpack geo2rdr parameters
    geo2rdr_dict = cfg['processing']['geo2rdr']
    threshold = geo2rdr_dict['threshold']
    maxiter = geo2rdr_dict['maxiter']

    # unpack pre-processing
    preprocess = cfg['processing']['pre_process']
    rg_look = preprocess['range_looks']
    az_look = preprocess['azimuth_looks']
    radar_grid_nlooks = rg_look * az_look

    # init parameters shared between frequencyA and frequencyB sub-bands
    slc = SLC(hdf5file=input_hdf5)
    zero_doppler = isce3.core.LUT2d()
    native_doppler = slc.getDopplerCentroid()

    # the variable `complex_type` will hold the complex data type
    # to be used for off-diagonal terms (if full covariance GCOV)
    complex_type = None

    for frequency, input_pol_list in freq_pols.items():

        # do no processing if no polarizations specified for current frequency
        if not input_pol_list:
            continue

        t_freq = time.time()

        # get sub_swaths metadata
        if apply_valid_samples_sub_swath_masking or save_mask:
            sub_swaths = slc.getSwathMetadata(frequency).sub_swaths()
        else:
            sub_swaths = None

        optional_geo_kwargs['sub_swaths'] = sub_swaths

        # unpack frequency dependent parameters
        radar_grid = slc.getRadarGrid(frequency)
        if radar_grid_nlooks > 1:
            radar_grid = radar_grid.multilook(az_look, rg_look)
        geogrid = geogrids[frequency]

        # set dict of input rasters
        input_raster_dict = {}

        # `input_pol_list` is the input list of polarizations that may include
        # HV and VH. `pol_list` is the actual list of polarizations to be
        # geocoded. It may include HV but it will not include VH if the
        # polarimetric symmetrization is performed
        pol_list = input_pol_list.copy()
        for pol in pol_list:
            temp_ref = f'HDF5:"{input_hdf5}":/{slc.slcPath(frequency, pol)}'
            input_raster_dict[pol] = temp_ref

        # symmetrize cross-polarimetric channels (if applicable)
        flag_symmetrization_required = (flag_symmetrize_cross_pol_channels and
                                        'HV' in input_pol_list and
                                        'VH' in input_pol_list)

        # Convert complex values to backscatter as a pre-processing step
        # only if full covariance was not requested and symmetrization was requested.
        # Note that the `GeocodeCov` module itself performs this conversion if necessary (and
        # it is faster than doing it as a pre-processing step if symmetrization is not required).
        flag_rslc_to_backscatter = (not flag_fullcovariance and
                                    flag_symmetrization_required)

        if flag_symmetrization_required:

            # temporary file for the symmetrized HV polarization
            symmetrized_hv_temp = tempfile.NamedTemporaryFile(
                dir=raster_scratch_dir, suffix=gcov_terms_file_extension)

            # call symmetrization function
            info_channel.log('Symmetrizing polarization channels HV and VH')
            input_raster = prepare_rslc(
                input_hdf5, frequency, 'HV',
                symmetrized_hv_temp.name, 2**11,  # 2**11 = 2048 lines
                flag_rslc_to_backscatter=flag_rslc_to_backscatter,
                pol_2='VH', format=output_gcov_terms_raster_files_format)

            # Since HV and VH were symmetrized into HV, remove VH from
            # `pol_list` and `from input_raster_dict`.
            pol_list.remove('VH')
            input_raster_dict.pop('VH')

            # Update `input_raster_dict` with the symmetrized polarization
            input_raster_dict['HV'] = input_raster

        # construct input rasters
        input_raster_list = []
        for pol, input_raster in input_raster_dict.items():

            # GDAL reference starting with HDF5 (RSLCs) are
            # converted to backscatter. This step is only
            # applied because h5py reads the NISAR C4 (4 bytes)
            # format faster than GDAL reads it. We take
            # advantage that we are reading the RLSC using
            # h5py to convert the data to backscatter (square)
            # and save it as float32.

            if isinstance(input_raster, str):
                info_channel.log(
                    f'Computing radar samples backscatter ({pol})')
                temp_pol_file = tempfile.NamedTemporaryFile(
                    dir=raster_scratch_dir,
                    suffix=gcov_terms_file_extension)
                input_raster = prepare_rslc(
                    input_hdf5, frequency, pol,
                    temp_pol_file.name, 2**12,  # 2**12 = 4096 lines
                    flag_rslc_to_backscatter=flag_rslc_to_backscatter,
                    format=output_gcov_terms_raster_files_format)

            input_raster_list.append(input_raster)

        info_channel.log('Preparing multi-band raster for geocoding')

        # if provided, load an external orbit from the runconfig file;
        # othewise, load the orbit from the RSLC metadata
        orbit = slc.getOrbit()
        if orbit_file is not None:
            external_orbit = load_orbit_from_xml(orbit_file,
                                                 radar_grid.ref_epoch)

            # Apply 2 mins of padding before / after sensing period when
            # cropping the external orbit.
            # 2 mins of margin is based on the number of IMAGEN TEC samples
            # required for TEC computation, with few more safety margins for
            # possible needs in the future.
            #
            # `7` in the line below is came from the default value for `npad`
            # in `crop_external_orbit()`. See:
            # .../isce3/python/isce3/core/crop_external_orbit.py
            npad = max(int(120.0 / external_orbit.spacing), 7)
            orbit = crop_external_orbit(external_orbit, orbit,
                                        npad=npad)

        # get azimuth ionospheric delay LUTs (if applicable)
        center_freq = \
            slc.getSwathMetadata(frequency).processed_center_frequency

        if apply_azimuth_ionospheric_delay_correction:
            az_correction = tec_lut2d_from_json_az(tec_file, center_freq,
                                                   orbit, radar_grid)
            optional_geo_kwargs['az_time_correction'] = az_correction

        # get slant-range ionospheric delay LUTs (if applicable)
        if apply_range_ionospheric_delay_correction:
            rg_correction = tec_lut2d_from_json_srg(tec_file, center_freq,
                                                    orbit, radar_grid,
                                                    zero_doppler, dem_file)
            optional_geo_kwargs['slant_range_correction'] = rg_correction

        root_ds = f'/science/LSAR/GCOV/grids/frequency{frequency}'

        optional_geo_kwargs['geogrid_upsampling'] = geogrid_upsampling
        optional_geo_kwargs['abs_cal_factor'] = abs_cal_factor
        optional_geo_kwargs['clip_min'] = clip_min
        optional_geo_kwargs['clip_max'] = clip_max
        optional_geo_kwargs['geogrid_upsampling'] = geogrid_upsampling
        optional_geo_kwargs['abs_cal_factor'] = abs_cal_factor
        optional_geo_kwargs['flag_upsample_radar_grid'] = \
            flag_upsample_radar_grid

        output_gcov_terms_kwargs['output_file_prefix'] = \
            f'frequency{frequency}_'
        output_secondary_layers_kwargs['output_file_prefix'] = \
            f'frequency{frequency}_'

        # non-None file spacing strategy can only be used in 'w', 'w-', or 'x'
        # mode. i.e. can not be used with an existing file. Otherwise ValueError
        # will be raised by h5py.
        if os.path.exists(output_hdf5):
            h5_write_mode = 'a'
            fs_strategy = None

        with h5py.File(output_hdf5, h5_write_mode,
                       fs_strategy=fs_strategy,
                       fs_page_size=fs_page_size) as hdf5_obj:

            run_geocode_cov(cfg, hdf5_obj, root_ds,
                            frequency, pol_list,
                            radar_grid, input_raster_list,
                            zero_doppler,
                            raster_scratch_dir,
                            geogrid, orbit, gcov_terms_file_extension,
                            output_gcov_terms_raster_files_format,
                            secondary_layers_file_extension,
                            secondary_layer_files_raster_files_format,
                            flag_fullcovariance, radar_grid_nlooks,
                            output_gcov_terms_kwargs,
                            output_secondary_layers_kwargs,
                            optional_geo_kwargs)

            t_freq_elapsed = time.time() - t_freq
            info_channel.log(f'frequency {frequency} ran in {t_freq_elapsed:.3f} seconds')

            if frequency.upper() == 'B' and 'A' in freq_pols:
                continue

            cube_geogrid = isce3.product.GeoGridParameters(
                start_x=radar_grid_cubes_geogrid.start_x,
                start_y=radar_grid_cubes_geogrid.start_y,
                spacing_x=radar_grid_cubes_geogrid.spacing_x,
                spacing_y=radar_grid_cubes_geogrid.spacing_y,
                width=int(radar_grid_cubes_geogrid.width),
                length=int(radar_grid_cubes_geogrid.length),
                epsg=radar_grid_cubes_geogrid.epsg)

            cube_group_name = '/science/LSAR/GCOV/metadata/radarGrid'
            native_doppler = slc.getDopplerCentroid()
            '''
            The native-Doppler LUT bounds error is turned off to
            computer cubes values outside radar-grid boundaries
            '''

            native_doppler.bounds_error = False
            # Get a 3D chunk size for metadata cubes from:
            # - One height layer (H = 1)
            # - The secondary layers chunk size (A X B):
            # chunk_size = [H, A, B] = [1, A, B]
            chunk_size_height_layers = [1]
            radar_grid_cubes_chunk_size = (
                chunk_size_height_layers +
                output_secondary_layers_kwargs['chunk_size'])
            radar_grid_cubes_compression_enabled = \
                output_secondary_layers_kwargs['compression_enabled']
            radar_grid_cubes_compression_type = \
                output_secondary_layers_kwargs['compression_type']
            radar_grid_cubes_compression_level = \
                output_secondary_layers_kwargs['compression_level']
            radar_grid_cubes_shuffle_filter = \
                output_secondary_layers_kwargs['shuffle_filtering_enabled']

            add_radar_grid_cubes_to_hdf5(
                hdf5_obj, cube_group_name,
                cube_geogrid,
                radar_grid_cubes_heights,
                radar_grid, orbit, native_doppler,
                zero_doppler, threshold, maxiter,
                chunk_size=radar_grid_cubes_chunk_size,
                compression_enabled=radar_grid_cubes_compression_enabled,
                compression_type=radar_grid_cubes_compression_type,
                compression_level=radar_grid_cubes_compression_level,
                shuffle_filter=radar_grid_cubes_shuffle_filter)

    return output_files_list


if __name__ == "__main__":

    t_all = time.time()
    info_channel = journal.info("gcov.run")

    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()
    gcov_runconfig = GCOVRunConfig(args)

    sas_output_file = gcov_runconfig.cfg[
        'product_path_group']['sas_output_file']

    if os.path.isfile(sas_output_file):
        os.remove(sas_output_file)

    output_files_list = run(gcov_runconfig.cfg)

    with GcovWriter(runconfig=gcov_runconfig) as gcov_obj:
        gcov_obj.populate_metadata()

    info_channel.log('output file(s):')
    for filename in output_files_list:
        info_channel.log(f'    {filename}')

    t_all_elapsed = time.time() - t_all
    hms_str = str(datetime.timedelta(seconds=int(t_all_elapsed)))
    t_all_elapsed_str = f'elapsed time: {hms_str}s ({t_all_elapsed:.3f}s)'

    info_channel.log(t_all_elapsed_str)
