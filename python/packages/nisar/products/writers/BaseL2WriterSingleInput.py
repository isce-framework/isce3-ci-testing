from osgeo import gdal, osr, ogr, gdal_array
import tempfile
import numpy as np
import warnings
import journal
import os

import isce3
from nisar.products.writers import BaseWriterSingleInput
from nisar.workflows.h5_prep import set_get_geo_info
from isce3.core.types import truncate_mantissa
from isce3.geometry import get_near_and_far_range_incidence_angles
from nisar.products.readers.orbit import load_orbit


LEXICOGRAPHIC_BASE_POLS = ['HH', 'HV', 'VH', 'VV']
COMPACT_POLS = ['RH', 'RV']
CROSSTALK_DATASETS = ['txHorizontalCrosspol',
                      'txVerticalCrosspol',
                      'rxHorizontalCrosspol',
                      'rxVerticalCrosspol']

# For now, `referenceTerrainHeight` is a 1-D Dataset, varying with
# azimuth time. In future, we plan to reimplement this as a 2-D Dataset,
# but in the meantime the code needs to be able to handle the 1-D case.
LUT_1D_AZ_DATASETS = ['referenceTerrainHeight']
LUT_1D_RG_DATASETS = CROSSTALK_DATASETS

# The constants below specify the number of rows or columns to be added
# in the direction where 1-D LUTs will be expanded into 2-D LUTs before
# the margin is added
LUT_1D_AZ_DATASETS_LUT_EXPANSION_ALONG_RG_N_POINTS = 21
LUT_1D_RG_DATASETS_LUT_EXPANSION_ALONG_AZ_N_POINTS = 21

# The number of additional rows or columns to be added in the
# direction where 1-D LUTs will be expanded into 2-D LUTs, beyond
# the extents of the input RSLC radar grid, on either side.
# This extra margin may be necessary for accommodating the width
# of the interpolator kernel when performing LUT lookups near the 
# boundary of the valid data.
LUT_1D_AZ_DATASETS_LUT_EXPANSION_ALONG_RG_MARGIN_IN_PIXELS = 5
LUT_1D_RG_DATASETS_LUT_EXPANSION_ALONG_AZ_MARGIN_IN_PIXELS = 5


def _get_attribute_dict(band,
                        standard_name=None,
                        long_name=None,
                        units=None,
                        fill_value=None,
                        valid_min=None,
                        valid_max=None,
                        stats_obj_list=None,
                        stats_real_imag_obj_list=None,
                        to_string_function=str):
    '''
    Get attribute dictionary for a raster layer

    Parameters
    ----------
    band : int
        Raster band
    standard_name : string, optional
        HDF5 dataset standard name
    long_name : string, optional
        HDF5 dataset long name
    units : str, optional
        Value to populate the HDF5 dataset attribute "units"
    fill_value : scalar, optional
        Value to populate the HDF5 dataset attribute "_FillValue"
    valid_min : scalar, optional
        Value to populate the HDF5 dataset attribute "valid_min"
    valid_max : scalar, optional
        Value to populate the HDF5 dataset attribute "valid_max"
    stats_obj_list : isce3.math.StatsFloat32 or isce3.math.StatsFloat64
        List of real-valued stats object
    stats_real_imag_obj_list : list(isce3.math.StatsRealImagFloat32) or
                               list(isce3.math.StatsRealImagFloat64), optional
        List of complex stats object
    to_string_function: function, optional
        Function to convert input data type to string

    Returns
    -------
    attr_dict: dict
        Dataset attributes represented as a dictionary
    '''

    attr_dict = {}
    if standard_name is not None:
        attr_dict['standard_name'] = to_string_function(standard_name)

    if long_name is not None:
        attr_dict['long_name'] = to_string_function(long_name)

    if units is not None:
        attr_dict['units'] = to_string_function(units)

    if fill_value is not None:
        attr_dict['_FillValue'] = fill_value

    if stats_obj_list is not None:
        stats_obj = stats_obj_list[band]
        attr_dict['min_value'] = stats_obj.min
        attr_dict['mean_value'] = stats_obj.mean
        attr_dict['max_value'] = stats_obj.max
        attr_dict['sample_stddev'] = stats_obj.sample_stddev

    elif stats_real_imag_obj_list is not None:

        stats_obj = stats_real_imag_obj_list[band]
        attr_dict['min_real_value'] = stats_obj.real.min
        attr_dict['mean_real_value'] = stats_obj.real.mean
        attr_dict['max_real_value'] = stats_obj.real.max
        attr_dict['sample_stddev_real'] = stats_obj.real.sample_stddev

        attr_dict['min_imag_value'] = stats_obj.imag.min
        attr_dict['mean_imag_value'] = stats_obj.imag.mean
        attr_dict['max_imag_value'] = stats_obj.imag.max
        attr_dict['sample_stddev_imag'] = stats_obj.imag.sample_stddev

    if valid_min is not None:
        attr_dict['valid_min'] = valid_min

    if valid_max is not None:
        attr_dict['valid_max'] = valid_max

    return attr_dict


def _get_stats_obj_list(raster, compute_stats):
    '''
    Get vector of statistic objects from a raster (isce3.io.Raster object)

    Parameters
    ----------
    raster: isce3.io.Raster
        Input raster
    compute_stats: bool
        Flag that indicates if statistics should be computed for the
        raster layer

    Returns
    -------
    stats_obj_list : list(isce3.math.StatsFloat32) or
                     list(isce3.math.StatsFloat64) or None
        List of real-valued stats object or None if compute_stats is False
    stats_real_imag_obj_list : list(isce3.math.StatsRealImagFloat32) or
                             list(isce3.math.StatsRealImagFloat64) or None
        List of complex-valued stats object or None if compute_stats is False
    '''
    stats_real_imag_obj_list = None
    stats_obj_list = None

    if not compute_stats:
        return stats_obj_list, stats_real_imag_obj_list

    if (raster.datatype() == gdal.GDT_CFloat32 or
            raster.datatype() == gdal.GDT_CFloat64):
        stats_real_imag_obj_list = \
            isce3.math.compute_raster_stats_real_imag(raster)
    elif (raster.datatype() == gdal.GDT_Float32 or
            raster.datatype() == gdal.GDT_Byte or
            raster.datatype() == gdal.GDT_Int16 or
            raster.datatype() == gdal.GDT_UInt16):
        stats_obj_list = isce3.math.compute_raster_stats_float32(raster)
    else:
        # Handle integer datatypes as float64.
        stats_obj_list = isce3.math.compute_raster_stats_float64(raster)
    return stats_obj_list, stats_real_imag_obj_list


def save_dataset(ds_filename, h5py_obj, root_path,
                 yds, xds, output_ds_name,
                 format='HDF5',
                 output_dir='',
                 output_file_prefix='',
                 compression_enabled=True,
                 compression_type='gzip',
                 compression_level=9,
                 chunking_enabled=True,
                 chunk_size=[512, 512],
                 shuffle_filtering_enabled=False,
                 mantissa_nbits=None,
                 output_files_list=None,
                 standard_name=None,
                 long_name=None,
                 units=None,
                 fill_value=None,
                 valid_min=None,
                 valid_max=None,
                 compute_stats=True,
                 hdf5_data_type=None):
    '''
    Write a temporary multi-band raster file as an output HDF5 file
    or a set of single-band files.

    Parameters
    ----------
    ds_filename : str
        Source raster file
    h5py_obj : h5py.File
        h5py object of destination HDF5
    root_path : str
        Path of the group within the HDF5 file to store the output raster
        data in (only applicable for "format" equal to "HDF5")
    yds : h5py.Dataset
        Y-axis dataset
    xds : h5py.Dataset
        X-axis dataset
    output_ds_name : str or list of str
        Dataset name or list of names to be used in the output,
        for each band of the source raster file. The number of
        dataset names must match the number of bands in the source raster
        file
    format : str
        File format. Options: "HDF5", "GTIFF", and "ENVI"
    output_dir : str
        Output directory (only applicable for "format"
        different than "HDF5)
        The output file name will be:
        {output_dir}/{output_file_prefix}{output_ds_name_band}{file_extension}'
    output_file_prefix : str
        Output file prefix (only applicable for "format"
        different than "HDF5).
        The output file name will be:
        {output_dir}/{output_file_prefix}{output_ds_name_band}{file_extension}'
    compression_enabled : bool, optional
        Enable/disable compression of output raster
        (only applicable for "format" equal to "HDF5")
    compression_type : str or None, optional
        Output data compression
        (only applicable for "format" equal to "HDF5")
    compression_level : int or None, optional
        Output data compression level
        (only applicable for "format" equal to "HDF5")
    chunking_enabled: bool, optional
        Enabled/disable chunk storage. Defaults to True
        (only applicable for "format" equal to "HDF5")
    chunk_size: list or None, optional
        Chunk size as a list (Y, X). If `chunk_size` is not provided or
        it is set to `None`, and `chunking_enabled` is set to `True`,
        `chunk_size` defaults to `[512, 512]`
        or a smaller size, constrained by the dimensions of the image
        (only applicable for "format" equal to "HDF5")
    shuffle_filtering_enabled: bool or None, optional
        Apply shuffle filter for block-oriented compressors
        (e.g., GZIP or LZF) to improve compression ratio.
        If None, the default option is used.
        (only applicable for "format" equal to "HDF5")
    mantissa_nbits: int or None, optional
        Number bits retained in the mantissa of the floating point
        representation of each component real and imaginary (if applicable)
        of each output sample.
        If None or "0", the mantissa bits truncation is not applied
    output_files_list : list or None, optional
        List of output files
        (only applicable for "format" different than "HDF5")
    standard_name : string, optional
        HDF5 dataset standard name
    long_name : string, optional
        HDF5 dataset long name
    units : str, optional
        Value to populate the HDF5 dataset attribute "units"
    fill_value : float, optional
        Value to populate the HDF5 dataset attribute "_FillValue".
        Defaults to "nan" or "(nan+nanj)" if the raster layer
        is real- or complex-valued, respectively. If the layer data
        type is integer and `fill_value` is None, the attribute "_FillValue"
        will not be populated as an attribute of the HDF5 dataset.
    valid_min : float, optional
        Value to populate the HDF5 dataset attribute "valid_min"
    valid_max : float, optional
        Value to populate the HDF5 dataset attribute "valid_max"
    compute_stats: bool, optional
        Flag that indicates if statistics should be compute for the
        raster layer. Defaults to True
    hdf5_data_type: Union[h5py.h5t.TypeCompoundID, numpy.dtype], optional
        Data type of the output H5 datasets
    '''

    raster = isce3.io.Raster(ds_filename)

    if (fill_value is None and raster.datatype() == gdal.GDT_CFloat32):
        fill_value = np.complex64(np.nan + 1j * np.nan)
    elif (fill_value is None and raster.datatype() == gdal.GDT_CFloat64):
        fill_value = np.complex128(np.nan + 1j * np.nan)
    elif (fill_value is None and raster.datatype() == gdal.GDT_Float32):
        fill_value = np.float32(np.nan)
    elif (fill_value is None and raster.datatype() == gdal.GDT_Float64):
        fill_value = np.float64(np.nan)

    if format == 'HDF5':
        save_hdf5_dataset(ds_filename, h5py_obj, root_path,
                          yds, xds, output_ds_name,
                          compression_enabled=compression_enabled,
                          compression_type=compression_type,
                          compression_level=compression_level,
                          chunking_enabled=chunking_enabled,
                          chunk_size=chunk_size,
                          shuffle_filtering_enabled=shuffle_filtering_enabled,
                          mantissa_nbits=mantissa_nbits,
                          standard_name=standard_name,
                          long_name=long_name,
                          units=units,
                          fill_value=fill_value,
                          valid_min=valid_min,
                          valid_max=valid_max,
                          compute_stats=compute_stats,
                          hdf5_data_type=hdf5_data_type)

    else:
        save_raster(ds_filename, output_ds_name,
                    format=format,
                    output_dir=output_dir,
                    output_file_prefix=output_file_prefix,
                    mantissa_nbits=mantissa_nbits,
                    output_files_list=output_files_list,
                    standard_name=standard_name,
                    long_name=long_name,
                    units=units,
                    fill_value=fill_value,
                    valid_min=valid_min,
                    valid_max=valid_max,
                    compute_stats=compute_stats)


def save_hdf5_dataset(ds_filename, h5py_obj, root_path,
                      yds, xds, output_ds_name,
                      compression_enabled=True,
                      compression_type='gzip',
                      compression_level=9,
                      chunking_enabled=True,
                      chunk_size=[512, 512],
                      shuffle_filtering_enabled=False,
                      mantissa_nbits=None,
                      standard_name=None,
                      long_name=None,
                      units=None,
                      fill_value=None,
                      valid_min=None,
                      valid_max=None,
                      compute_stats=True,
                      hdf5_data_type=None):
    '''
    Write a raster files as a set of HDF5 datasets

    Parameters
    ----------
    ds_filename : str
        Source raster file
    h5py_obj : h5py.File
        h5py object of destination HDF5
    root_path : str
        Path of the group within the HDF5 file to store the output raster
    yds : h5py.Dataset
        Y-axis dataset
    xds : h5py.Dataset
        X-axis dataset
    output_ds_name : str or list of str
        Dataset name or list of names to be added to `root_path`
        for each band of the source raster file. The number of
        dataset names must match the number of bands in the source raster
        file
    compression_enabled : bool, optional
        Enable/disable compression of output raster
    compression_type : str or None, optional
        Output data compression
    compression_level : int or None, optional
        Output data compression level
    chunking_enabled: bool, optional
        Enabled/disable chunk storage. Defaults to True
    chunk_size: list or None, optional
        Chunk size as a list (Y, X). If `chunk_size` is not provided or
        it is set to `None`, and `chunking_enabled` is set to `True`,
        `chunk_size` defaults to `[512, 512]`
        or a smaller size, constrained by the dimensions of the image
    shuffle_filtering_enabled: bool or None, optional
        Apply shuffle filter for block-oriented compressors
        (e.g., GZIP or LZF) to improve compression ratio.
        If None, the default option is used
    mantissa_nbits: int or None, optional
        Number bits retained in the mantissa of the floating point
        representation of each component real and imaginary (if applicable)
        of each output sample.
        If None or "0", the mantissa bits truncation is not applied
    standard_name : string, optional
        HDF5 dataset standard name
    long_name : string, optional
        HDF5 dataset long name
    units : str, optional
        Value to populate the HDF5 dataset attribute "units"
    fill_value : float, optional
        Value to populate the HDF5 dataset attribute "_FillValue".
        Defaults to "nan" or "(nan+nanj)" if the raster layer
        is real- or complex-valued, respectively. If the layer data
        type is integer and `fill_value` is None, the attribute "_FillValue"
        will not be populated as an attribute of the HDF5 dataset.
    valid_min : float, optional
        Value to populate the HDF5 dataset attribute "valid_min"
    valid_max : float, optional
        Value to populate the HDF5 dataset attribute "valid_max"
    compute_stats: bool, optional
        Flag that indicates if statistics should be compute for the
        raster layer. Defaults to True
    hdf5_data_type: Union[h5py.h5t.TypeCompoundID, numpy.dtype], optional
        Data type of the output H5 datasets
    '''

    info_channel = journal.info("save_hdf5_dataset")

    gdal_ds = gdal.Open(ds_filename, gdal.GA_ReadOnly)
    nbands = gdal_ds.RasterCount
    length = gdal_ds.RasterYSize
    width = gdal_ds.RasterXSize

    if isinstance(output_ds_name, str):
        num_names = 1
    else:
        num_names = len(output_ds_name)

    if nbands != num_names:
        raise ValueError('The number of output dataset names must match number'
                         ' of input raster bands.'
                         f' Number of output file names: {num_names}'
                         f' ({output_ds_name}). Number of bands in raster'
                         f' "{ds_filename}": {nbands}')

    raster = isce3.io.Raster(ds_filename)

    create_dataset_kwargs = {}

    stats_obj_list, stats_real_imag_obj_list = \
        _get_stats_obj_list(raster, compute_stats)

    if chunking_enabled:
        if chunk_size is None:
            chunk_size = [512, 512]

        effective_chunk_size = (min(chunk_size[0], length),
                                min(chunk_size[1], width))

        create_dataset_kwargs['chunks'] = effective_chunk_size

    if compression_enabled:
        if compression_type is not None:
            create_dataset_kwargs['compression'] = compression_type

        if compression_level is not None:
            create_dataset_kwargs['compression_opts'] = \
                compression_level

        if shuffle_filtering_enabled is not None:
            create_dataset_kwargs['shuffle'] = shuffle_filtering_enabled

    for band in range(nbands):
        gdal_band = gdal_ds.GetRasterBand(band+1)

        attr_dict = _get_attribute_dict(
            band,
            standard_name=standard_name,
            long_name=long_name,
            units=units,
            fill_value=fill_value,
            valid_min=valid_min,
            valid_max=valid_max,
            stats_obj_list=stats_obj_list,
            stats_real_imag_obj_list=stats_real_imag_obj_list,
            to_string_function=np.bytes_)

        if isinstance(output_ds_name, str):
            output_ds_name_band = output_ds_name
        else:
            output_ds_name_band = output_ds_name[band]

        h5_ds = f'{root_path}/{output_ds_name_band}'

        if hdf5_data_type is not None:
            band_data_type = hdf5_data_type
        else:
            band_data_type = \
                gdal_array.GDALTypeCodeToNumericTypeCode(gdal_band.DataType)

        width = int(gdal_ds.RasterXSize)
        length = int(gdal_ds.RasterYSize)

        block_nbands = 1
        n_threads = 1
        # Convert type size from bits to bytes (assuming word size is 8 bits)
        data_type_size = int(gdal.GetDataTypeSize(gdal_band.DataType)) // 8
        min_block_size = int(isce3.core.default_min_block_size)
        max_block_size = int(isce3.core.default_max_block_size)

        ret = isce3.core.get_block_processing_parameters(
            array_length=length,
            array_width=width,
            nbands=block_nbands,
            type_size=data_type_size,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            snap=effective_chunk_size[0],
            n_threads=n_threads)

        n_blocks_y = ret['n_blocks_y']
        n_blocks_x = ret['n_blocks_x']
        block_length = ret['block_length']
        block_width = ret['block_width']

        info_channel.log(f'saving dataset: {h5_ds}')
        info_channel.log(f'    length: {length}')
        info_channel.log(f'    width: {width}')
        info_channel.log(f'    block processing (Y-direction): {n_blocks_y}'
                         f' blocks of {block_length} lines')
        info_channel.log(f'    block processing (X-direction): {n_blocks_x}'
                         f' blocks of {block_width} columns')
        info_channel.log('    h5 dataset chunk length:'
                         f' {effective_chunk_size[0]}')
        info_channel.log('    h5 dataset chunk width:'
                         f' {effective_chunk_size[1]}')

        # create the output H5 dataset
        dset = h5py_obj.create_dataset(h5_ds,
                                       shape=(length, width),
                                       dtype=band_data_type,
                                       **create_dataset_kwargs)

        for block_y in range(n_blocks_y):

            offset_y = block_y * block_length
            this_block_length = min(block_length, length - offset_y)

            for block_x in range(n_blocks_x):

                offset_x = block_x * block_width
                this_block_width = min(block_width, width - offset_x)

                # read data block from raster
                data_block = gdal_band.ReadAsArray(
                    xoff=int(offset_x),
                    yoff=int(offset_y),
                    win_xsize=int(this_block_width),
                    win_ysize=int(this_block_length)
                )

                # truncate array (if applicable)
                if mantissa_nbits is not None:
                    truncate_mantissa(data_block, mantissa_nbits)

                # write data block to HDF5 file
                block_slice = np.index_exp[
                    offset_y:offset_y + this_block_length,
                    offset_x:offset_x + this_block_width
                ]
                dset.write_direct(data_block, dest_sel=block_slice)

        dset.dims[0].attach_scale(yds)
        dset.dims[1].attach_scale(xds)
        dset.attrs['grid_mapping'] = np.bytes_("projection")

        for attr_name, attr_value in attr_dict.items():
            dset.attrs.create(attr_name, data=attr_value)


def get_file_extension(format):
    '''
    Get file extension for a supported file formats "GTiff" and
    "ENVI"

    Parameters
    ----------
    format: str
        File format: "GTiff" or "ENVI"

    Returns
    -------
    file_extension: str
        File extension
    '''
    if format == 'GTiff':
        file_extension = '.tif'
    elif format == 'ENVI':
        file_extension = '.bin'
    else:
        error_message = f"Unsupported file format: {format}"
        error_channel = journal.error('get_file_extension')
        error_channel.log(error_message)
        raise NotImplementedError(error_message)
    return file_extension


def save_raster(ds_filename, output_ds_name,
                format='GTiff',
                output_dir='',
                output_file_prefix='',
                mantissa_nbits=None,
                output_files_list=None,
                standard_name=None,
                long_name=None,
                units=None,
                fill_value=None,
                valid_min=None,
                valid_max=None,
                compute_stats=True):
    '''
    Write a raster layer from a multi-band file into individual
    single-band raster files

    Parameters
    ----------
    ds_filename : str
        Source raster file
    output_ds_name : str
        Dataset name or list of names to be used as file suffixes
        for each band of the source raster file. The number of
        dataset names must match the number of bands in the source raster
        file.
        The output file name(s) will be:
        {output_dir}/{output_file_prefix}{output_ds_name_band}{file_extension}'
    format : str
        File format. Options: "GTIFF", and "ENVI"
    output_dir : str
        Output directory (only applicable for "format"
        different than "HDF5)
        The output file name will be:
        {output_dir}/{output_file_prefix}{output_ds_name_band}{file_extension}'
    output_file_prefix : str
        Output file prefix (only applicable for "format"
        different than "HDF5).
        The output file name will be:
        {output_dir}/{output_file_prefix}{output_ds_name_band}{file_extension}'
    mantissa_nbits: int or None, optional
        Number bits retained in the mantissa of the floating point
        representation of each component real and imaginary (if applicable)
        of each output sample.
        If None or "0", the mantissa bits truncation is not applied
    output_files_list : list or None, optional
        List of output files
    standard_name : string, optional
        Dataset standard name
    long_name : string, optional
        Dataset long name
    units : str, optional
        Value to populate the HDF5 dataset attribute "units"
    fill_value : float, optional
        Value to populate the HDF5 dataset attribute "_FillValue".
        If None, the "nodata" attribute of the dataset will not be populated.
    valid_min : float, optional
        Value to populate the HDF5 dataset attribute "valid_min"
    valid_max : float, optional
        Value to populate the HDF5 dataset attribute "valid_max"
    '''
    raster = isce3.io.Raster(ds_filename)

    gdal_ds = gdal.Open(ds_filename, gdal.GA_ReadOnly)
    nbands = gdal_ds.RasterCount
    if isinstance(output_ds_name, str):
        num_names = 1
    else:
        num_names = len(output_ds_name)

    if nbands != num_names:
        raise ValueError('The number of output dataset names must match number'
                         ' of input raster bands.'
                         f' Number of output file names: {num_names}'
                         f' ({output_ds_name}).'
                         f' Number of bands in raster "{ds_filename}":'
                         f' {nbands}')
    stats_obj_list, stats_real_imag_obj_list = \
        _get_stats_obj_list(raster, compute_stats)

    for band in range(nbands):
        gdal_band = gdal_ds.GetRasterBand(band+1)
        data = gdal_band.ReadAsArray()

        if mantissa_nbits is not None:
            truncate_mantissa(data, mantissa_nbits)

        attr_dict = _get_attribute_dict(
            band,
            standard_name=standard_name,
            long_name=long_name,
            units=units,
            fill_value=fill_value,
            valid_min=valid_min,
            valid_max=valid_max,
            stats_obj_list=stats_obj_list,
            stats_real_imag_obj_list=stats_real_imag_obj_list)

        if isinstance(output_ds_name, str):
            output_ds_name_band = output_ds_name
        else:
            output_ds_name_band = output_ds_name[band]

        # Save non-HDF5 output file
        gdal_dtype = gdal_band.DataType
        projection = gdal_ds.GetProjectionRef()
        geotransform = gdal_ds.GetGeoTransform()

        # get file extension
        file_extension = get_file_extension(format)

        # get output filename
        output_file = \
            f'{output_file_prefix}{output_ds_name_band}{file_extension}'
        if output_dir:
            output_file = os.path.join(output_dir, output_file)

        if output_files_list is not None:
            output_files_list.append(output_file)

        driver_out = gdal.GetDriverByName(format)
        raster_out = driver_out.Create(
            output_file, data.shape[1],
            data.shape[0], 1, gdal_dtype)
        raster_out.SetProjection(projection)
        raster_out.SetGeoTransform(geotransform)
        raster_out.SetMetadata(attr_dict)
        raster_out.SetDescription(long_name)

        band_out = raster_out.GetRasterBand(1)
        if fill_value is not None:
            band_out.SetNoDataValue(fill_value)
        band_out.WriteArray(data)
        band_out.FlushCache()

        # As a precaution, delete the raster objects in order to
        # free memory inside this loop.
        del band_out
        del raster_out


class BaseL2WriterSingleInput(BaseWriterSingleInput):
    """
    Base L2 writer class that can be used for NISAR L2 products
    """

    def __init__(self, runconfig, *args, orbit=None, **kwargs):

        super().__init__(runconfig, *args, **kwargs)

        self.freq_pols_dict = self.cfg['processing']['input_subset'][
            'list_of_frequencies']

        # This orbit file will be saved in the product metadata. It is assumed that
        # the same orbit file was used in the SAS workflow processing.
        self.orbit_file = \
            self.cfg["dynamic_ancillary_file_group"]['orbit_file']

        # if the orbit has not been provided, i.e., `orbit is None`,
        # load it from the input RSLC or from the external file (if provided)
        # using load_orbit()
        if orbit is not None:
            self.orbit = orbit
        else:
            ref_epoch = self.input_product_obj.getRadarGrid().ref_epoch
            self.orbit = load_orbit(self.input_product_obj, self.orbit_file,
                                    ref_epoch)

    @property
    def frequencies(self) -> list[str]:
        """
        The frequency subbands of the output product in the same order as the
        runconfig
        """
        return list(self.freq_pols_dict.keys())

    @property
    def first_sorted_frequency(self) -> str:
        """
        The first frequency of the output product in alphabetical order
        """
        return sorted(self.frequencies)[0]

    def populate_identification_l2_specific(self):
        """
        Populate L2 product specific parameters in the
        identification group
        """

        self.copy_from_input(
            'identification/zeroDopplerStartTime')

        self.copy_from_input(
            'identification/zeroDopplerEndTime')

        self.set_value(
            'identification/productLevel',
            'L2')

        is_geocoded = True
        self.set_value(
            'identification/isGeocoded',
            is_geocoded)

        self.copy_from_input(
            'identification/boundingPolygon')

        list_of_frequencies = list(self.freq_pols_dict.keys())

        self.set_value(
            'identification/listOfFrequencies',
            list_of_frequencies)

        self.copy_from_input(
            'identification/platformName',
            default='(NOT SPECIFIED)')

    def populate_ceos_analysis_ready_data_parameters_l2_common(self):

        self.copy_from_runconfig(
            '{PRODUCT}/metadata/ceosAnalysisReadyData/staticLayersDataAccess',
            'ceos_analysis_ready_data/static_layers_data_access',
            default='(NOT SPECIFIED)')

        ceos_ard_document_identifier = \
            ('https://ceos.org/ard/files/PFS/SAR/v1.0/CEOS-ARD_PFS'
             '_Synthetic_Aperture_Radar_v1.0.pdf')
        self.set_value(
            '{PRODUCT}/metadata/ceosAnalysisReadyData/'
            'ceosAnalysisReadyDataDocumentIdentifier',
            ceos_ard_document_identifier)

        # Iterate over the geogrids for each frequency and
        # store the maximum extents. Y extents are reversed
        # because the step size in the Y direction is negative
        geogrids = self.cfg['processing']['geocode']['geogrids']

        list_of_frequencies = list(self.freq_pols_dict.keys())

        start_x = +np.inf
        end_x = -np.inf
        start_y = -np.inf
        end_y = +np.inf

        epsg_code = None

        for frequency in list_of_frequencies:
            geogrid = geogrids[frequency]

            if epsg_code is None:
                epsg_code = geogrid.epsg
            elif epsg_code != geogrid.epsg:
                error_msg = ('EPSG code does not match for frequencies:'
                             f'{list_of_frequencies}')
                error_channel = journal.error(
                    'populate_identification_l2_specific')
                error_channel.log(error_msg)
                raise NotImplementedError(error_msg)

            start_x = min(start_x, geogrid.start_x)
            end_x = max(end_x, geogrid.end_x)

            start_y = max(start_y, geogrid.start_y)
            end_y = min(end_y, geogrid.end_y)

        # Notes about anti-meridian crossing:
        #
        # Estimating minimum and maximum longitudes values
        # from a grid in geographic coordinates (EPSG 4326)
        # can be challenging because longitude values can be
        # "wrapped" around the antimeridian, i.e., +181 becomes
        # -179. However, since in GeoGridParameters, `end_x`
        # is obtained as `start_x + width * step_x`,
        # the longitude coordinates will not be "wrapped" if they
        # cross the antimeridian. This means that `end_x` will not be
        # less than `start_x` and the minimum and maximum values
        # can be correctly estimated

        # Create geogrids' bbox ring
        # We choose the counter-clockwise orientation because this
        # defines the exterior of the polygon according to the OpenGIS
        # Standards:
        # OpenGIS Implementation Standard for Geographic
        # information - Simple feature access - Part 1: Common
        # architecture, v1.2.1, 2011-05-28 (page 26)
        geogrids_bbox_ring = ogr.Geometry(ogr.wkbLinearRing)
        geogrids_bbox_ring.AddPoint(start_x, start_y)
        geogrids_bbox_ring.AddPoint(start_x, end_y)
        geogrids_bbox_ring.AddPoint(end_x, end_y)
        geogrids_bbox_ring.AddPoint(end_x, start_y)
        geogrids_bbox_ring.AddPoint(start_x, start_y)

        # Create geogrids' bbox polygon
        geogrids_bbox_polygon = ogr.Geometry(ogr.wkbPolygon)
        geogrids_bbox_polygon.AddGeometry(geogrids_bbox_ring)

        # Assign georeference to the geogrids' bbox polygon
        geogrid_srs = osr.SpatialReference()
        geogrid_srs.ImportFromEPSG(epsg_code)
        geogrids_bbox_polygon.AssignSpatialReference(geogrid_srs)
        assert geogrids_bbox_polygon.IsValid()

        # Create a well-known text (WKT) representation of the geogrids'
        # bbox polygon and save it as the H5 Dataset `boundingBox``
        bounding_box_wkt = geogrids_bbox_polygon.ExportToWkt()

        self.set_value(
            '{PRODUCT}/metadata/ceosAnalysisReadyData/boundingBox',
            bounding_box_wkt)

        bounding_box_path = \
            (f'{self.output_product_path}/metadata/ceosAnalysisReadyData/'
             'boundingBox')

        self.output_hdf5_obj[bounding_box_path].attrs['epsg'] = epsg_code

        for xy in ['x', 'y']:
            h5_grp_path = ('{PRODUCT}/metadata/ceosAnalysisReadyData/'
                           'geometricAccuracy')
            runcfg_prefix = ('ceos_analysis_ready_data/'
                             'estimated_geometric_accuracy')

            bias_dataset = self.copy_from_runconfig(
                f'{h5_grp_path}/bias/{xy}',
                f'{runcfg_prefix}_bias_{xy}',
                format_function=np.float32,
                default=np.nan)
            bias_dataset.attrs['epsg'] = epsg_code

            standard_deviation_dataset = self.copy_from_runconfig(
                f'{h5_grp_path}/standardDeviation/{xy}',
                f'{runcfg_prefix}_standard_deviation_{xy}',
                format_function=np.float32,
                default=np.nan)
            standard_deviation_dataset.attrs['epsg'] = epsg_code

    def populate_calibration_information(self):

        # calibration parameters to be copied from the RSLC
        # common to all polarizations
        calibration_freq_parameter_list = ['commonDelay',
                                           'faradayRotation']

        # calibration metadata to be copied from the RSLC which are specific to
        # each polarization channel, but which only exist for
        # polarizations with a corresponding imagery layer in the RSLC
        calibration_existing_pols_parameter_list = \
            ['rfiLikelihood']

        # calibration metadata to be copied from the RSLC
        # that are specific to each polarization channel
        # but exist for to all possible polarizations within a given
        # polarimetric basis
        # including polarizations without a corresponding image layer
        # See: `isce3.python.packages.nisar.products.writers.SLC.set_calibration()`
        calibration_all_pols_parameter_list = \
            ['differentialDelay',
             'differentialPhase',
             'scaleFactor',
             'scaleFactorSlope']

        for frequency, pol_list in self.freq_pols_dict.items():

            for parameter in calibration_freq_parameter_list:
                cal_freq_path = (
                    '{PRODUCT}/metadata/calibrationInformation/'
                    f'frequency{frequency}')

                self.copy_from_input(f'{cal_freq_path}/{parameter}',
                                     default=np.nan)

            for pol in pol_list:
                for parameter in calibration_existing_pols_parameter_list:
                    self.copy_from_input(
                        f'{cal_freq_path}/{pol}/{parameter}',
                        default=np.nan)

            # Copy polarimetric calibration parameters associated with all
            # polarizations for given basis
            product_pols = []
            for pol_list in self.freq_pols_dict.values():
                product_pols.extend(pol_list)

            if all(pol in LEXICOGRAPHIC_BASE_POLS for pol in product_pols):
                list_of_all_pols = LEXICOGRAPHIC_BASE_POLS
            elif all(pol in COMPACT_POLS for pol in product_pols):
                list_of_all_pols = COMPACT_POLS
            else:
                error_channel = journal.error(
                    'BaseL2WriterSingleInput.populate_calibration_information')
                error_msg = ('Unsupported polarimetric channels:'
                             f' {self.freq_pols_dict}')
                error_channel.log(error_msg)
                raise KeyError(error_msg)

            for pol in list_of_all_pols:
                for parameter in calibration_all_pols_parameter_list:
                    self.copy_from_input(f'{cal_freq_path}/{pol}/{parameter}',
                                         default=np.nan)

        self.geocode_lut('{PRODUCT}/metadata/calibrationInformation/crosstalk',
                         output_ds_name_list=CROSSTALK_DATASETS,
                         skip_if_not_present=True,
                         compute_stats=True)

        for abstract_lut in ("eap", "noise"):

            if abstract_lut == "eap":
                old_name = new_name = "elevationAntennaPattern"
                compute_stats = False
            elif abstract_lut == "noise":
                # Noise dataset was renamed in product spec v1.2.0.
                old_name = "nes0"
                new_name = "noiseEquivalentBackscatter"
                compute_stats = True
            else:
                assert False, f"unexpected {abstract_lut=}"

            # geocode frequency dependent LUTs
            for frequency, pol_list in self.freq_pols_dict.items():

                # The path below is only valid for RSLC products
                # with product specification version 1.2.0 or above
                success = self.geocode_lut(
                    '{PRODUCT}/metadata/calibrationInformation/'
                    f'frequency{frequency}/{new_name}',
                    frequency=frequency,
                    output_ds_name_list=pol_list,
                    skip_if_not_present=True,
                    compute_stats=compute_stats)

                # Try reading with the old name.
                if not success and new_name != old_name:
                    root = ('{PRODUCT}/metadata/calibrationInformation/'
                            f'frequency{frequency}')
                    success = self.geocode_lut(
                        output_h5_group=f"{root}/{new_name}",
                        input_h5_group=f"{root}/{old_name}",
                        frequency=frequency,
                        output_ds_name_list=pol_list,
                        skip_if_not_present=True,
                        compute_stats=compute_stats)

                # Failed with these path schema, don't bother with other pols.
                if not success:
                    break

            # Succeeded with these path schema, don't bother trying the old one.
            if success:
                continue

            # The code below handles RSLC products with
            # product specification version prior to 1.1.0
            for frequency, pol_list in self.freq_pols_dict.items():
                for pol in pol_list:

                    zero_doppler_time_path = (
                        f'{self.input_product_path}/metadata/'
                        f'calibrationInformation/frequency{frequency}/{pol}/'
                        'zeroDopplerTime')

                    if zero_doppler_time_path in self.input_hdf5_obj:

                        self.geocode_lut(
                            output_h5_group=('{PRODUCT}/metadata/'
                                             'calibrationInformation'
                                             f'/frequency{frequency}/{new_name}'),
                            input_h5_group=('{PRODUCT}/metadata/'
                                            'calibrationInformation'
                                            f'/frequency{frequency}/{pol}'),
                            frequency=list(self.freq_pols_dict.keys())[0],
                            input_ds_name_list=[old_name],
                            output_ds_name_list=pol,
                            skip_if_not_present=True,
                            compute_stats=compute_stats)

                        continue

                    input_ds_name_list = [f'frequency{frequency}/{pol}/{old_name}']

                    self.geocode_lut(
                        output_h5_group=('{PRODUCT}/metadata/'
                                         'calibrationInformation'
                                         f'/frequency{frequency}/{new_name}'),
                        input_h5_group=('{PRODUCT}/metadata/'
                                        'calibrationInformation'),
                        frequency=list(self.freq_pols_dict.keys())[0],
                        input_ds_name_list=input_ds_name_list,
                        output_ds_name_list=pol,
                        skip_if_not_present=True,
                        compute_stats=compute_stats)

    def populate_orbit(self):

        # Save the orbit ephemeris
        orbit_hdf5_group = self.output_hdf5_obj[
            f'{self.output_product_path}/metadata'].require_group(
                "orbit")

        # Save the orbit into the HDF5 file
        # NISAR products' specifications require that the orbit reference
        # epoch has integer precision, i.e., without fractional part.
        # However, products generated from other software (e.g.,
        # JPL SimulationTools) may contain orbit reference epoch with
        # fractional precision. If that happens, the function below
        # will raise a RuntimeError
        try:
            self.orbit.save_to_h5(orbit_hdf5_group)
        except RuntimeError:

            # If a RuntimeError is raised, try again disabling requirement
            # for orbit reference epoch precision in seconds
            self.orbit.save_to_h5(orbit_hdf5_group,
                                  ensure_epoch_integer_seconds=False)

            # If this time, the orbit serialization is successful, print a
            # warning message to the user explaining that the orbit reference
            # epoch is saved with a fractional part, which is inconsistent
            # with NISAR product specifications.
            warning_channel = journal.warning(
                "BaseL2WriterSingleInput.populate_orbit")
            warning_channel.log(
                'The orbit reference epoch is not an integer, which is'
                ' inconsistent with NISAR product specifications. Conformance'
                ' will be disregarded, and the orbit reference epoch will be'
                ' saved with its fractional part.')

    def populate_attitude(self):
        # RSLC products with product specification version prior to v1.1.0 may
        # include the H5 group "angularVelocity", that should not be copied to
        # L2 products
        excludes_list = ['angularVelocity']

        # copy attitude information group
        self._copy_group_from_input('{PRODUCT}/metadata/attitude',
                                    excludes=excludes_list)

    def populate_source_data(self):
        """
        Populate the `sourceData` group
        """

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/productDoi',
            'identification/productDoi',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/productVersion',
            'identification/productVersion',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/lookDirection',
            'identification/lookDirection',
            format_function=str.title)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/productLevel',
            'identification/productLevel',
            default='L1')

        # CEOS ARD convention is 'Slant range' or 'Ground range'.
        self.set_value(
            '{PRODUCT}/metadata/sourceData/productGeometry',
            'Slant range')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingDateTime',
            'identification/processingDateTime',
            skip_if_not_present=True)

        # this parameter should be copied from the input RSLC product
        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingCenter',
            'identification/processingCenter',
            default='(NOT SPECIFIED)')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/productType',
            'identification/productType',
            default='(NOT SPECIFIED)')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/granuleId',
            'identification/granuleId',
            default='(NOT SPECIFIED)')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'parameters/runConfigurationContents',
            '{PRODUCT}/metadata/processingInformation/parameters/'
            'runConfigurationContents',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'parameters/referenceTerrainHeight',
            '{PRODUCT}/metadata/processingInformation/parameters/'
            'referenceTerrainHeight',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'parameters/zeroDopplerTime',
            '{PRODUCT}/metadata/processingInformation/parameters/'
            'zeroDopplerTime',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'parameters/slantRange',
            '{PRODUCT}/metadata/processingInformation/parameters/'
            'slantRange',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rfiDetection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rfiDetection',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rfiMitigation',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rfiMitigation',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rangeCompression',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rangeCompression',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/elevationAntennaPatternCorrection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'elevationAntennaPatternCorrection',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/rangeSpreadingLossCorrection',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'rangeSpreadingLossCorrection',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/dopplerCentroidEstimation',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'dopplerCentroidEstimation',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/azimuthPresumming',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'azimuthPresumming',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/azimuthCompression',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'azimuthCompression',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/processingInformation/'
            'algorithms/softwareVersion',
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'softwareVersion',
            skip_if_not_present=True)

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/swaths/zeroDopplerStartTime',
            'identification/zeroDopplerStartTime')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/swaths/zeroDopplerEndTime',
            'identification/zeroDopplerEndTime')

        self.copy_from_input(
            '{PRODUCT}/metadata/sourceData/swaths/zeroDopplerTimeSpacing',
            '{PRODUCT}/swaths/zeroDopplerTimeSpacing')

        for i, (frequency, _) in enumerate(self.freq_pols_dict.items()):
            radar_grid_obj = self.input_product_obj.getRadarGrid(frequency)

            output_swaths_freq_path = ('{PRODUCT}/metadata/sourceData/'
                                       f'swaths/frequency{frequency}')
            input_swaths_freq_path = ('{PRODUCT}/swaths/'
                                      f'frequency{frequency}')

            near_range_inc_angle_rad, far_range_inc_angle_rad = \
                get_near_and_far_range_incidence_angles(radar_grid_obj,
                                                        self.orbit)

            near_range_inc_angle_deg = np.rad2deg(near_range_inc_angle_rad)
            far_range_inc_angle_deg = np.rad2deg(far_range_inc_angle_rad)

            self.set_value(
                f'{output_swaths_freq_path}/nearRangeIncidenceAngle',
                near_range_inc_angle_deg)

            self.set_value(
                f'{output_swaths_freq_path}/farRangeIncidenceAngle',
                far_range_inc_angle_deg)

            self.copy_from_input(
                f'{output_swaths_freq_path}/listOfPolarizations',
                f'{input_swaths_freq_path}/listOfPolarizations')

            if i == 0:
                self.set_value(
                    '{PRODUCT}/metadata/sourceData/swaths/'
                    'numberOfAzimuthLines',
                    radar_grid_obj.length,
                    format_function=np.uint64)

            # compute azimuth resolution
            azimuth_bandwidth_path = (
                f'{self.input_product_path}'
                f'/swaths/frequency{frequency}/processedAzimuthBandwidth')
            azimuth_bandwidth = self.input_hdf5_obj[azimuth_bandwidth_path][()]

            along_track_spacing_path = (
                f'{self.input_product_path}'
                f'/swaths/frequency{frequency}/sceneCenterAlongTrackSpacing')
            along_track_spacing = self.input_hdf5_obj[
                along_track_spacing_path][()]

            zero_doppler_time_spacing_path = (
                f'{self.input_product_path}'
                f'/swaths/zeroDopplerTimeSpacing')
            zero_doppler_time_spacing = self.input_hdf5_obj[
                zero_doppler_time_spacing_path][()]

            # ground velocity and azimuth resolution computed at the scene center
            ground_velocity = along_track_spacing / zero_doppler_time_spacing
            azimuth_resolution = ground_velocity / azimuth_bandwidth

            self.set_value(
                f'{output_swaths_freq_path}/sceneCenterAlongTrackResolution',
                azimuth_resolution)

            # compute range resolution
            range_bandwidth_path = (
                f'{self.input_product_path}'
                f'/swaths/frequency{frequency}/processedRangeBandwidth')
            range_bandwidth = self.input_hdf5_obj[range_bandwidth_path][()]
            range_resolution = (isce3.core.speed_of_light /
                                (2 * range_bandwidth))
            self.set_value(
                f'{output_swaths_freq_path}/rangeResolution',
                range_resolution)

            self.copy_from_input(
                f'{output_swaths_freq_path}/sceneCenterAlongTrackSpacing',
                f'{input_swaths_freq_path}/sceneCenterAlongTrackSpacing')

            self.copy_from_input(
                f'{output_swaths_freq_path}/sceneCenterGroundRangeSpacing',
                f'{input_swaths_freq_path}/sceneCenterGroundRangeSpacing')

            self.copy_from_input(
                f'{output_swaths_freq_path}/acquiredRangeBandwidth',
                f'{input_swaths_freq_path}/acquiredRangeBandwidth')

            self.copy_from_input(
                f'{output_swaths_freq_path}/processedRangeBandwidth',
                f'{input_swaths_freq_path}/processedRangeBandwidth')

            self.copy_from_input(
                f'{output_swaths_freq_path}/processedAzimuthBandwidth',
                f'{input_swaths_freq_path}/processedAzimuthBandwidth')

            self.copy_from_input(
                f'{output_swaths_freq_path}/centerFrequency',
                f'{input_swaths_freq_path}/processedCenterFrequency')

            self.set_value(
                f'{output_swaths_freq_path}/slantRangeStart',
                radar_grid_obj.starting_range)

            self.copy_from_input(
                f'{output_swaths_freq_path}/slantRangeSpacing',
                f'{input_swaths_freq_path}/slantRangeSpacing')

            self.set_value(
                f'{output_swaths_freq_path}/numberOfRangeSamples',
                radar_grid_obj.width,
                format_function=np.uint64)

            self.copy_from_input(
                '{PRODUCT}/metadata/sourceData/processingInformation/'
                f'parameters/frequency{frequency}/dopplerCentroid',
                '{PRODUCT}/metadata/processingInformation/parameters/'
                f'frequency{frequency}/dopplerCentroid',
                skip_if_not_present=True)

            # Copy range-Doppler Doppler Centroid LUT into the sourceData
            # group.
            # First, we look for the coordinate vectors `zeroDopplerTime`
            # and `slantRange` in the same level of the `dopplerCentroid` LUT.
            # If these vectors are not found, a `KeyError` exception will be
            # raised. We catch that exception, and look for the coordinate
            # vectors two levels below, following old RSLC specs.
            try:
                self.copy_from_input(
                    '{PRODUCT}/metadata/sourceData/processingInformation/'
                    f'parameters/frequency{frequency}/zeroDopplerTime',
                    '{PRODUCT}/metadata/processingInformation/parameters/'
                    f'frequency{frequency}/zeroDopplerTime')
            except KeyError:
                self.copy_from_input(
                    '{PRODUCT}/metadata/sourceData/processingInformation/'
                    f'parameters/frequency{frequency}/zeroDopplerTime',
                    '{PRODUCT}/metadata/processingInformation/parameters/'
                    'zeroDopplerTime')
            try:
                self.copy_from_input(
                    '{PRODUCT}/metadata/sourceData/processingInformation/'
                    f'parameters/frequency{frequency}/slantRange',
                    '{PRODUCT}/metadata/processingInformation/parameters/'
                    f'frequency{frequency}/slantRange')
            except KeyError:
                self.copy_from_input(
                    '{PRODUCT}/metadata/sourceData/processingInformation/'
                    f'parameters/frequency{frequency}/slantRange',
                    '{PRODUCT}/metadata/processingInformation/parameters/'
                    'slantRange')

    def populate_processing_information_l2_common(self):

        # Since the flag "rfiMitigationApplied" is not present in the RSLC
        # metadata, we populate it by reading the name of the
        # RFI mitigation algorithm from the RSLC metadata. The flag
        # is only True if the name of the algorithm is present in the metadata,
        # if it's not empty, and if the substring "disabled" is not part of the
        # algorithm name
        rfi_mitigation_path = (
            f'{self.output_product_path}/metadata/processingInformation/'
            'algorithms/rfiMitigation')
        flag_rfi_mitigation_applied = (
            rfi_mitigation_path in self.output_hdf5_obj and
            rfi_mitigation_path != '' and
            'disabled' not in rfi_mitigation_path.lower())

        parameters_group = \
            '{PRODUCT}/metadata/processingInformation/parameters'

        self.set_value(
            f'{parameters_group}/rfiMitigationApplied',
            flag_rfi_mitigation_applied)

        self.set_value(
            '{PRODUCT}/metadata/processingInformation/algorithms/'
            'softwareVersion',
            isce3.__version__)

        # populate processing information inputs parameters
        inputs_group = \
            '{PRODUCT}/metadata/processingInformation/inputs'

        self.set_value(
            f'{inputs_group}/l1SlcGranules',
            [os.path.basename(self.input_file)])

        # populate input orbit and TEC files
        for ancillary_type in ['orbit', 'tec']:
            ancillary_file = self.cfg[
                'dynamic_ancillary_file_group'][f'{ancillary_type}_file']
            if ancillary_file is None:
                ancillary_file = '(NOT SPECIFIED)'
            self.set_value(
                f'{inputs_group}/{ancillary_type}Files',
                [ancillary_file])

        # `run_config_path` can be either a file name or a string
        # representing the contents of the runconfig file (identified
        # by the presence of a "/n" in the `run_config_path`)
        if '\n' not in self.runconfig.args.run_config_path:
            self.set_value(
                f'{inputs_group}/configFiles',
                [self.runconfig.args.run_config_path])

        self.copy_from_runconfig(
            f'{inputs_group}/demSource',
            'dynamic_ancillary_file_group/dem_file_description',
            default='(NOT SPECIFIED)')

        # geocode referenceTerrainHeight
        ds_name_list = ['referenceTerrainHeight']
        self.geocode_lut(
            '{PRODUCT}/metadata/processingInformation/parameters',
            output_ds_name_list=ds_name_list,
            skip_if_not_present=True)

        # geocode dopplerCentroid LUT
        ds_name_list = ['dopplerCentroid']
        for frequency in self.freq_pols_dict.keys():

            self.geocode_lut(
                '{PRODUCT}/metadata/processingInformation/parameters/'
                f'frequency{frequency}',
                frequency=frequency,
                output_ds_name_list=ds_name_list,
                skip_if_not_present=True)

    def geocode_lut(self, output_h5_group, input_h5_group=None,
                    frequency=None, output_ds_name_list=None,
                    input_ds_name_list=None,
                    skip_if_not_present=False,
                    compute_stats=False):
        """
        Geocode a look-up table (LUT) from the input product in
        radar coordinates to the output product in map coordinates

        Parameters
        ----------
        output_h5_group: str
            Path to the output HDF5 LUT group
        input_h5_group: str, optional
            Path to the input HDF5 LUT group. If not provided, the
            same path of the output dataset `output_h5_group` will
            be used
        frequency: str, optional
            Frequency sub-band, used to read the sub-band radar grid and/or
            wavelength.
            The sub-band wavelength is only used in geocoding 
            (during geo2rdr) if the dataset (LUT) is not in the
            zero-Doppler geometry. If not specified, the first
            available frequency in the sorted list will be used.
        output_ds_name_list: str, list
            List of LUT datasets to geocode
        input_ds_name_list: list
            List of LUT datasets to geocode. If not provided, the
            same list as the `output_ds_name_list` will be used
        skip_if_not_present: bool, optional
            Flag to prevent the execution to stop if the dataset
            is not present from input
        compute_stats: bool, optional
            Flag that indicates if statistics should be computed for the
            output raster layer. Defaults to False.

        Returns
        -------
        success: bool
           Flag that indicates if the geocoding the LUT group was successful
        """
        if not output_ds_name_list:
            return False

        if input_h5_group is None:
            input_h5_group = output_h5_group

        if input_ds_name_list is None and isinstance(output_ds_name_list, str):
            input_ds_name_list = [output_ds_name_list]
        elif input_ds_name_list is None:
            input_ds_name_list = output_ds_name_list

        input_h5_group_path = self.root_path + '/' + input_h5_group
        input_h5_group_path = \
            input_h5_group_path.replace(
                '{PRODUCT}', self.input_product_hdf5_group_type)

        error_channel = journal.error('geocode_lut')

        # check if group exists within the input product
        if input_h5_group_path not in self.input_hdf5_obj:
            not_found_msg = ('Metadata entry not found in the input'
                             ' product: ' + input_h5_group_path)
            if skip_if_not_present:
                warnings.warn(not_found_msg)
                return False
            else:
                error_channel.log(not_found_msg)
                raise KeyError(not_found_msg)

        output_h5_group_path = self.root_path + '/' + output_h5_group
        output_h5_group_path = \
            output_h5_group_path.replace('{PRODUCT}', self.product_type)

        is_calibration_information_group = \
            'calibrationInformation' in output_h5_group
        is_processing_information_group = \
            'processingInformation' in output_h5_group

        if (is_calibration_information_group and
                is_processing_information_group):
            error_msg = ('Malformed input product group:'
                         f' {output_h5_group}. Should'
                         ' contain either "calibrationInformation"'
                         ' or "processingInformation"')
            error_channel.log(error_msg)
            raise ValueError(error_msg)
        if is_calibration_information_group:
            metadata_group = 'calibrationInformation'
        elif is_processing_information_group:
            metadata_group = 'processingInformation'
        else:
            error_msg = f'Could not determine LUT group for {output_h5_group}'
            error_channel.log(error_msg)
            raise NotImplementedError(error_msg)

        if frequency is None:
            frequency = self.first_sorted_frequency

        return self.geocode_metadata_group(
            frequency,
            input_ds_name_list,
            output_ds_name_list,
            metadata_group,
            input_h5_group_path,
            output_h5_group_path,
            skip_if_not_present,
            compute_stats)

    def geocode_metadata_group(self,
                               frequency,
                               input_ds_name_list,
                               output_ds_name_list,
                               metadata_group,
                               input_h5_group_path,
                               output_h5_group_path,
                               skip_if_not_present,
                               compute_stats):
        """
        Geocode look-up tables (LUTs) from the input product in
        radar coordinates to the output product in map coordinates
        using runconfig parameters associated with that
        metadata group, either 'calibrationInformation'
        or 'processingInformation'

        Parameters
        ----------
        frequency: str, optional
            Frequency sub-band, used to read the sub-band radar grid and/or
            wavelength.
            The sub-band wavelength is only used in geocoding 
            (during geo2rdr) if the dataset (LUT) is not in the 
            zero-Doppler geometry
        input_ds_name_list: list
            List of LUT datasets to geocode
        output_ds_name_list: str, list
            List of output LUT datasets
        metadata_group: str
            Metadata group, either 'calibrationInformation'
            or 'processingInformation'
        input_h5_group_path: str
            Path of the input group
        output_h5_group_path: str
            Path of the output group
        skip_if_not_present: bool, optional
            Flag to prevent the execution to stop if the dataset
            is not present from input
        compute_stats: bool, optional
            Flag that indicates if statistics should be computed for the
            output raster layer. Defaults to False.

        Returns
        -------
        success: bool
           Flag that indicates if the geocoding the LUT group was successful
        """

        error_channel = journal.error('geocode_metadata_group')

        scratch_path = self.cfg['product_path_group']['scratch_path']

        if metadata_group == 'calibrationInformation':
            metadata_geogrid = self.cfg['processing'][
                'calibration_information']['geogrid']
        elif metadata_group == 'processingInformation':
            metadata_geogrid = self.cfg['processing'][
                'processing_information']['geogrid']
        else:
            error_msg = f'Invalid metadata group {metadata_group}'
            error_channel.log(error_msg)
            raise NotImplementedError(error_msg)

        dem_file = self.cfg['dynamic_ancillary_file_group']['dem_file']

        # unpack geo2rdr parameters
        geo2rdr_dict = self.cfg['processing']['geo2rdr']
        threshold = geo2rdr_dict['threshold']
        maxiter = geo2rdr_dict['maxiter']

        # init parameters shared between frequencyA and frequencyB sub-bands
        dem_raster = isce3.io.Raster(dem_file)
        zero_doppler = isce3.core.LUT2d()

        epsg = dem_raster.get_epsg()
        proj = isce3.core.make_projection(epsg)
        ellipsoid = proj.ellipsoid

        # do not apply any exponentiation to the samples to geocode
        exponent = 1

        geocode_mode = isce3.geocode.GeocodeOutputMode.INTERP

        radar_grid_slc = self.input_product_obj.getRadarGrid(frequency)

        # If some -- but not all -- input datasets are 1-D, an error will be
        # raised later on.
        flag_luts_are_1d_rg = all([var in LUT_1D_RG_DATASETS
                                   for var in input_ds_name_list])

        if not flag_luts_are_1d_rg:
            zero_doppler_path = f'{input_h5_group_path}/zeroDopplerTime'
            try:
                zero_doppler_h5_dataset = self.input_hdf5_obj[
                    zero_doppler_path]
            except KeyError:
                not_found_msg = ('Metadata entry not found in the input'
                                 ' product: ' + zero_doppler_path)
                if skip_if_not_present:
                    warnings.warn(not_found_msg)
                    return False
                else:
                    error_channel.log(not_found_msg)
                    raise KeyError(not_found_msg)
            lines = zero_doppler_h5_dataset.size

            if lines >= 2:
                time_spacing = np.average(np.diff(zero_doppler_h5_dataset))
            else:
                time_spacing = None

            if time_spacing is None or time_spacing <= 0:
                error_msg = ('Invalid zero-Doppler time array under'
                             f' {zero_doppler_path}:'
                             f' {zero_doppler_h5_dataset[()]}')
                error_channel.log(error_msg)
                raise RuntimeError(error_msg)

            ref_epoch = isce3.io.get_ref_epoch(zero_doppler_h5_dataset.parent,
                                               zero_doppler_h5_dataset.name)

            prf = 1.0 / time_spacing

            sensing_start = zero_doppler_h5_dataset[0]
        else:
            # read starting and ending sensing time from the RSLC radar grid
            sensing_start = radar_grid_slc.sensing_start
            sensing_end = radar_grid_slc.sensing_stop

            # determine number of lines, pulse-repetition interval (PRI)
            # pulse-repetitition frequency (PRF) and reference epoch
            lines = LUT_1D_RG_DATASETS_LUT_EXPANSION_ALONG_AZ_N_POINTS
            pri = (sensing_end - sensing_start) / (lines - 1)
            prf = 1.0 / pri
            ref_epoch = radar_grid_slc.ref_epoch

            # Add margins in both sides
            margin_in_pixels = \
                LUT_1D_RG_DATASETS_LUT_EXPANSION_ALONG_AZ_MARGIN_IN_PIXELS
            margin_in_seconds = margin_in_pixels * pri
            sensing_start -= margin_in_seconds
            sensing_end += margin_in_seconds
            lines += 2 * margin_in_pixels

        # The `referenceTerrainHeight` LUT can be either a 1-D LUT (along
        # azimuth) or a 2-D LUT. So, to determine the type of the LUT to
        # geocode, check the constant `LUT_1D_AZ_DATASETS`, but also verify if
        # `slantRange` is present within the LUT group to confirm its
        # dimensions.
        slant_range_path = f'{input_h5_group_path}/slantRange'
        flag_luts_are_1d_az = (all([var in LUT_1D_AZ_DATASETS
                                   for var in input_ds_name_list]) and
                               slant_range_path not in self.input_hdf5_obj)

        if not flag_luts_are_1d_az:
            slant_range_path = f'{input_h5_group_path}/slantRange'
            try:
                slant_range_h5_dataset = self.input_hdf5_obj[slant_range_path]
            except KeyError:
                not_found_msg = ('Metadata entry not found in the input'
                                 ' product: ' + slant_range_path)
                if skip_if_not_present:
                    warnings.warn(not_found_msg)
                    return False
                else:
                    error_channel.log(not_found_msg)
                    raise KeyError(not_found_msg)
            samples = slant_range_h5_dataset.size

            if samples >= 2:
                range_spacing = np.average(np.diff(slant_range_h5_dataset))
            else:
                range_spacing = None

            if range_spacing is None or range_spacing <= 0:
                error_msg = ('Invalid range spacing array under'
                             f' {slant_range_path}: {slant_range_h5_dataset[()]}')
                error_channel.log(error_msg)
                raise RuntimeError(error_msg)

            starting_range = slant_range_h5_dataset[0]

        else:
            # read starting and ending range from the RSLC radar grid
            starting_range = radar_grid_slc.starting_range
            ending_range = radar_grid_slc.end_range

            # determine number of samples and range spacing
            samples = LUT_1D_AZ_DATASETS_LUT_EXPANSION_ALONG_RG_N_POINTS
            range_spacing = ((ending_range - starting_range) /
                             (samples - 1))

            # Add margins in both sides
            margin_in_pixels = \
                LUT_1D_AZ_DATASETS_LUT_EXPANSION_ALONG_RG_MARGIN_IN_PIXELS
            margin_in_meters = margin_in_pixels * range_spacing
            starting_range -= margin_in_meters
            ending_range += margin_in_meters
            samples += 2 * margin_in_pixels

        radar_grid = isce3.product.RadarGridParameters(
                sensing_start,
                radar_grid_slc.wavelength,
                prf,
                starting_range,
                range_spacing,
                radar_grid_slc.lookside,
                lines, samples, ref_epoch)

        # construct input rasters
        input_raster_list = []

        '''
        Create list of input Raster objects. The list will be used to
        create the input VRT file and raster (input_raster_obj).
        input_ds_name_list defines the target HDF5 dataset that will contain
        the geocoded metadata
        '''
        flag_all_succeeded = True
        for count, var in enumerate(input_ds_name_list):
            var_h5_path = f'{input_h5_group_path}/{var}'

            if var_h5_path not in self.input_hdf5_obj:
                not_found_msg = ('Metadata entry not found in the input'
                                 ' product: ' + var_h5_path)
                if skip_if_not_present:
                    warnings.warn(not_found_msg)
                    flag_all_succeeded = False

                    # remove corresponding dataset in `output_ds_name_list`
                    output_ds_name_list.pop(count)
                    continue
                else:
                    error_channel.log(not_found_msg)
                    raise KeyError(not_found_msg)

            # Some LUTs may be 1-dimensional. These datasets need to handled
            # differently.
            # If the dataset does not have one dimension, create an ISCE3
            # Raster object and continue to the next for-loop iteration
            if not flag_luts_are_1d_rg and not flag_luts_are_1d_az:
                raster_ref = f'HDF5:"{self.input_file}":/{var_h5_path}'

                # Read `raster_ref` catching/handling potential problems
                temp_raster = isce3.io.Raster(raster_ref)
                input_raster_list.append(temp_raster)
                continue

            # Handle 1-D LUTs
            var_h5_dataset = self.input_hdf5_obj[var_h5_path]
            var_array = var_h5_dataset[()]

            # If LUT is a vector along azimuth
            if flag_luts_are_1d_az:
                warning_msg = ('Geolocating one dimensional dataset:'
                               f' {var_h5_path} in {self.input_file}'
                               ' (az. vector)')
                warnings.warn(warning_msg)
                new_var_array = np.repeat(np.transpose([var_array]),
                                          samples, axis=1)

            # If LUT is a vector along range
            elif flag_luts_are_1d_rg:
                warning_msg = ('Geolocating one dimensional dataset:'
                               f' {var_h5_path} in {self.input_file}'
                               ' (rg. vector)')
                warnings.warn(warning_msg)

                new_var_array = np.repeat([var_array], lines, axis=0)

            else:
                not_found_msg = ('Failed to create GDAL dataset from'
                                 f' reference: {raster_ref}')
                if skip_if_not_present:
                    warnings.warn(not_found_msg)
                    return False

                error_channel.log(not_found_msg)
                raise KeyError(not_found_msg)

            temp_file = tempfile.NamedTemporaryFile(dir=scratch_path,
                                                    suffix='.bin')
            length, width = new_var_array.shape
            dtype = gdal_array.NumericTypeCodeToGDALTypeCode(
                new_var_array.dtype)
            temp_raster = isce3.io.Raster(path=temp_file.name,
                                          width=width,
                                          length=length,
                                          num_bands=1,
                                          dtype=dtype,
                                          driver_name="ENVI")
            temp_raster[:, :] = new_var_array

            input_raster_list.append(temp_raster)

        if len(input_raster_list) == 0:
            warnings.warn('No dataset to geocode from input list: '
                          f'{input_ds_name_list}')
            return False

        # create a temporary file that will point to the input layers
        input_temp = tempfile.NamedTemporaryFile(
            dir=scratch_path, suffix='.vrt')
        input_raster_obj = isce3.io.Raster(
            input_temp.name, raster_list=input_raster_list)

        # init Geocode object depending on raster type
        if input_raster_obj.datatype() == gdal.GDT_Float32:
            geo = isce3.geocode.GeocodeFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_Float64:
            geo = isce3.geocode.GeocodeFloat64()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat32:
            geo = isce3.geocode.GeocodeCFloat32()
        elif input_raster_obj.datatype() == gdal.GDT_CFloat64:
            geo = isce3.geocode.GeocodeCFloat64()
        else:
            err_str = 'Unsupported raster type for geocoding'
            error_channel.log(err_str)
            raise NotImplementedError(err_str)

        # init geocode members
        geo.orbit = self.orbit
        geo.ellipsoid = ellipsoid
        geo.doppler = zero_doppler
        geo.threshold_geo2rdr = threshold
        geo.numiter_geo2rdr = maxiter

        geo.geogrid(metadata_geogrid.start_x,
                    metadata_geogrid.start_y,
                    metadata_geogrid.spacing_x,
                    metadata_geogrid.spacing_y,
                    metadata_geogrid.width,
                    metadata_geogrid.length,
                    metadata_geogrid.epsg)

        # create output raster
        temp_output = tempfile.NamedTemporaryFile(
            dir=scratch_path, suffix='.tif')

        dtype = input_raster_obj.datatype()

        output_raster_obj = isce3.io.Raster(
            temp_output.name, metadata_geogrid.width, metadata_geogrid.length,
            input_raster_obj.num_bands, dtype, 'GTiff')

        # geocode rasters
        geo.geocode(radar_grid=radar_grid,
                    input_raster=input_raster_obj,
                    output_raster=output_raster_obj,
                    output_mode=geocode_mode,
                    dem_raster=dem_raster,
                    exponent=exponent)

        output_raster_obj.close_dataset()
        del output_raster_obj

        x_coord_path = f'{output_h5_group_path}/xCoordinates'
        y_coord_path = f'{output_h5_group_path}/yCoordinates'

        if (x_coord_path not in self.output_hdf5_obj or
                y_coord_path not in self.output_hdf5_obj):
            yds, xds, *_ = set_get_geo_info(
                self.output_hdf5_obj, output_h5_group_path, metadata_geogrid,
                flag_save_coordinate_spacing=False)
        else:
            xds = self.output_hdf5_obj[x_coord_path]
            yds = self.output_hdf5_obj[y_coord_path]

        save_dataset(temp_output.name, self.output_hdf5_obj,
                     output_h5_group_path,
                     yds, xds, output_ds_name_list,
                     compute_stats=compute_stats)

        input_temp.close()
        temp_output.close()

        return flag_all_succeeded
