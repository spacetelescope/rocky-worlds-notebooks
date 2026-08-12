# -*- coding: utf-8 -*-
"""
Build Rocky Worlds DDT eclipse catalog and light-curve products.

This module creates single-eclipse and checkpoint HLSP products from Eureka
outputs, including HDF5/netCDF catalog products, FITS catalog summaries, and
grouped light-curve products with one native time axis per visit.
"""

from pathlib import Path
import numpy as np
import xarray as xr
from astropy.io import fits
from jwst.datamodels import JwstDataModel
import unicodedata
import re
import pandas as pd

# Setting default metadata that won't ever change
DOI = '10.17909/qsyr-ny68'
HLSPID = 'Rocky-Worlds'
HLSPNAME = 'Rocky Worlds DDT'
HLSP_PI = 'Néstor Espinoza, Hannah Diamond-Lowe'
HLSPLEAD = 'Taylor J. Bell'
LICENSE = 'CC BY 4.0'
LICENURL = 'https://creativecommons.org/licenses/by/4.0/'
RADESYS = 'ICRS'
TUNIT = 'mJy'
TIMESYS = 'TDB'

# Define constants for data product formatting
upperErrorPercentile = 84.135
lowerErrorPercentile = 15.865
medianPercentile = 50
eclipseDepthDecimals = 1
eclipseDepth_units = 'ppm'
eclipseTimeDecimals = 5
timeOffset = 2400000.5
time_units = 'BJD_TDB'
absoluteFluxDecimals = 5
lightcurveDecimals = 7
timeDecimals = 7
lightcurve_units = 'None (normalized)'


def _da1(value, visit_idx, units=None):
    """
    Create a 1-element DataArray along the ``visit`` dimension.

    Parameters
    ----------
    value : Any
        Value to place in the single-element array.
    visit_idx : int
        Visit index to use as the coordinate value.
    units : str or None, optional
        Units to attach to the DataArray (stored in ``attrs['units']``).

    Returns
    -------
    xarray.DataArray
        DataArray with shape ``(visit: 1)`` and optional units attribute.
    """
    da = xr.DataArray(
        np.array([value]),
        coords={'visit': [visit_idx]},
        dims=['visit'],
    )
    if units:
        da.attrs['units'] = units
    return da


def _derive_visit_id_string(file_ids):
    """
    Parse JWST segment names to collect one or more VISIT_ID strings.

    Parameters
    ----------
    file_ids : list of str
        Base names like
        ``jw09235001001_03101_00001-seg001_mirimage`` (no extension).

    Returns
    -------
    str
        Comma-separated, numerically sorted unique VISIT_IDs, each an
        11-digit string (e.g., ``'09235001001,09235002001'``). Returns
        an empty string if none are found.

    Notes
    -----
    The parser looks for ``^jw(\\d{11})_`` at the start of each base name.
    """
    pat = re.compile(r'^jw(\d{11})_')
    ids = []
    for fid in file_ids:
        base = Path(fid).name
        m = pat.match(base)
        if m:
            ids.append(m.group(1))
    uniq = sorted(set(ids), key=int)
    return ','.join(uniq)


def read_stage2_r_s_meta(stage2_fits_path):
    """
    Read Stage-2 FITS header and extract provenance and basic context.

    Parameters
    ----------
    stage2_fits_path : str
        Path to the Stage-2 FITS file (calints product).

    Returns
    -------
    r_keys : dict
        Mapping of ``R_*`` keywords to string values from the header.
    s_keys : dict
        Mapping of ``S_*`` keywords to string values from the header.
    crds_ver : str or None
        CRDS software version (``CRDS_VER``) if present.
    crds_ctx : str or None
        CRDS context (``CRDS_CTX`` or ``CRDS_PMAP``) if present.
    filt : str
        Filter name (``FILTER``) or empty string if not present.
    subarray : str
        Subarray name (``SUBARRAY``) or empty string if not present.
    date_obs : numpy.datetime64
        Combined ``DATE-OBS`` and ``TIME-OBS`` as ``datetime64[ns]``.

    Notes
    -----
    Only the primary header (HDU 0) is inspected. Values are returned as
    strings when available; missing cards yield empty strings or ``None``.
    """
    with fits.open(stage2_fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
        keys = list(hdr.keys())
        r_keys = {k: hdr[k] for k in keys
                  if isinstance(k, str) and k[:2] == 'R_'}
        s_keys = {k: hdr[k] for k in keys
                  if isinstance(k, str) and k[:2] == 'S_'}
        crds_ver = hdr.get('CRDS_VER')
        crds_ctx = hdr.get('CRDS_CTX', hdr.get('CRDS_PMAP'))
        filt = hdr.get('FILTER', '') or ''
        subarray = hdr.get('SUBARRAY', '') or ''
        date = str(hdr.get('DATE-OBS', '')).strip()
        time = str(hdr.get('TIME-OBS', '')).strip() or '00:00:00'
    if date:
        try:
            date_obs = np.datetime64(f'{date}T{time}')
        except Exception:
            date_obs = np.datetime64('NaT')
    else:
        date_obs = np.datetime64('NaT')
    return r_keys, s_keys, crds_ver, crds_ctx, filt, subarray, date_obs


def build_single_eclipse(
    stage2_fits,
    stage3_specdata,
    stage4cal,
    stage5_samples,
    visit,
    STAR,
    PLANET,
    SRC_DOI,
    HLSPVER,
    stage5_epf=None,
    stage5_ecf=None,
    eclipse_epoch=None,
):
    """
    Build a one-row (visit) Dataset with visit-varying fields as columns.

    This function loads all required inputs from file paths. It computes the
    eclipse depth, timing, and absolute flux with uncertainties, collects
    per-visit provenance from the Stage-2 FITS header (``R_*``, ``S_*``, and
    ``CRDS_*``), and stores HLSP invariants in global attributes.

    Parameters
    ----------
    stage2_fits : str
        Path to the Stage 2 FITS file.
    stage3_specdata : str
        Path to Stage 3 product providing per-segment file list and versions.
    stage4cal : str
        Path to Stage 4 flux calibration product holding stellar flux and
        error.
    stage5_samples : str
        Path to Stage 5 posterior samples. The samples may contain direct
        ``fp`` and ``t_secondary`` variables, or timing parameters that can
        be combined with fixed values from ``stage5_epf``.
    visit : int
        Eclipse index you assign (1, 2, ...) used for the ``visit`` coord
        and in output filenames. This is distinct from JWST VISIT_ID(s).
    STAR : str
        Host star identifier (per visit).
    PLANET : str
        Planet identifier (per visit).
    SRC_DOI : str
        DOI for the source data specific to this visit.
    HLSPVER : str
        HLSP version string to store in the dataset attributes.
    stage5_epf : str or None, optional
        Eureka Stage-5 parameter file. Fixed values from this file are used
        when posterior samples do not directly contain a needed parameter.
    stage5_ecf : str or None, optional
        Eureka Stage-5 control file. Its ``compute_ltt`` setting controls
        whether derived secondary-eclipse times include the light-travel-time
        correction. Defaults to ``True`` when absent.
    eclipse_epoch : int or None, optional
        Explicit eclipse epoch for derived secondary-eclipse times. If
        ``None``, the epoch is inferred from the Stage-2 exposure midpoint.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with a single row along ``visit`` that contains results,
        visit-varying metadata as columns, and global HLSP invariants.

    Notes
    -----
    The output is designed for later concatenation along the ``visit``
    dimension using :func:`xarray.concat`. The ``VISIT_ID`` column is
    derived from Stage-3 segment names and may contain multiple IDs
    (comma-separated) if the eclipse spans several JWST observations. When
    ``t_secondary`` is absent or fixed indirectly, the function derives the
    mid-eclipse time from ``t0``, ``per``, eccentricity parameters, and the
    EPF/ECF context.
    """
    # Load inputs (the LC file path is accepted but not used nor stored)
    spec_ds = xr.load_dataset(stage3_specdata)
    fluxcal = xr.load_dataset(stage4cal)
    samples = _load_samples_dataset(stage5_samples)
    dm = JwstDataModel(stage2_fits)

    # FITS header provenance and context
    (
        r_from_fits,
        s_from_fits,
        crds_ver_hdr,
        crds_ctx_hdr,
        filt_hdr,
        subarray_hdr,
        date_obs,
    ) = read_stage2_r_s_meta(stage2_fits)

    # Prefer header values when available
    instrume = dm.meta.instrument.name
    filter_name = filt_hdr or dm.meta.instrument.filter
    subarray = subarray_hdr or getattr(dm.meta.subarray, 'name', '')
    telescop = dm.meta.telescope
    observat = dm.meta.telescope

    # Pointing / program context
    proposid = dm.meta.observation.program_number
    ra_targ = dm.meta.target.ra
    dec_targ = dm.meta.target.dec

    # Segment list to FILE_ID (comma-separated string)
    file_ids = []
    if hasattr(spec_ds, 'segment_list'):
        for seg in spec_ds.segment_list:
            name = Path(seg).name
            file_ids.append(name.split('_calints')[0])
    file_id_str = ','.join(file_ids)

    # Derive VISIT_IDs from segment base names
    visit_id_str = _derive_visit_id_string(file_ids)

    # Exposure timing
    mjd_beg = float(dm.meta.exposure.start_time_tdb)
    mjd_mid = float(dm.meta.exposure.mid_time_tdb)
    mjd_end = float(dm.meta.exposure.end_time_tdb)
    xposure = float(dm.meta.exposure.effective_exposure_time)

    # Software versions
    cal_ver = str(dm.meta.calibration_software_version)
    pipe_line = str(getattr(spec_ds, 'data_format', ''))
    pipe_ver = str(getattr(spec_ds, 'version', ''))
    crds_ver = str(
        crds_ver_hdr or getattr(dm.meta.ref_file.crds, 'sw_version', '')
    )
    crds_ctx = str(
        crds_ctx_hdr or getattr(dm.meta.ref_file.crds, 'context_used', '')
    )

    v = int(visit)
    epf_params = parse_epf(stage5_epf)
    ecf_config = parse_ecf(stage5_ecf)
    compute_ltt = get_compute_ltt(ecf_config, default=True)
    n_samples = _infer_n_samples(samples)
    ch = 0

    # Science results: eclipse depth (ppm)
    fp_samples, _ = _resolve_visit_param(
        samples,
        epf_params,
        'fp',
        v,
        n_samples,
        channel=ch,
    )
    depth_vals = np.asarray(fp_samples, dtype=float) * 1.0e6
    d_med, d_std, d_up, d_lo = _array_stats(
        depth_vals,
        eclipseDepthDecimals,
    )

    # Science results: eclipse time
    t_vals = _resolve_secondary_time_samples(
        samples,
        epf_params,
        v,
        n_samples,
        mjd_mid=mjd_mid,
        channel=ch,
        eclipse_epoch=eclipse_epoch,
        compute_ltt=compute_ltt,
    )
    t_med, t_std, t_up, t_lo = _array_stats(
        t_vals,
        eclipseTimeDecimals,
    )

    # Science results: absolute flux (with systematic inflation)
    stellar_flux = float(fluxcal.ecl_flux.data[0][0])
    ferr = float(fluxcal.ecl_ferr.data[0][0])
    sys_err = (0.0048 * stellar_flux) ** 2 + (0.0045 * stellar_flux) ** 2
    stellar_flux_err = np.sqrt(ferr**2 + sys_err)
    f_med = np.round(stellar_flux, absoluteFluxDecimals)
    f_err = np.round(stellar_flux_err, absoluteFluxDecimals)

    # Build Dataset (one row)
    ds = xr.Dataset()

    # Results with units (via helper to avoid duplication)
    ds['eclipseDepth'] = _da1(d_med, v, units=eclipseDepth_units)
    ds['eclipseDepthError'] = _da1(d_std, v, units=eclipseDepth_units)
    ds['eclipseDepthUpperError'] = _da1(d_up, v, units=eclipseDepth_units)
    ds['eclipseDepthLowerError'] = _da1(d_lo, v, units=eclipseDepth_units)

    ds['eclipseTime'] = _da1(t_med, v, units=time_units)
    ds['eclipseTimeError'] = _da1(t_std, v, units=time_units)
    ds['eclipseTimeUpperError'] = _da1(t_up, v, units=time_units)
    ds['eclipseTimeLowerError'] = _da1(t_lo, v, units=time_units)

    ds['absFlux'] = _da1(f_med, v, units=TUNIT)
    ds['absFluxError'] = _da1(f_err, v, units=TUNIT)
    ds['absFluxUpperError'] = _da1(f_err, v, units=TUNIT)
    ds['absFluxLowerError'] = _da1(f_err, v, units=TUNIT)

    # Only "visit" is a coordinate
    ds = ds.assign_coords(visit=[v])

    # Visit context as DATA VARIABLES (not coordinates)
    ds['date_obs'] = _da1(date_obs, v)
    ds['filter'] = _da1(str(filter_name), v)
    ds['subarray'] = _da1(str(subarray), v)

    # Visit-varying metadata as columns
    ds['SRC_DOI'] = _da1(str(SRC_DOI), v)
    ds['VISIT_ID'] = _da1(visit_id_str, v)       # was 'VISIT'
    ds['FILE_ID'] = _da1(file_id_str, v)
    ds['MJD_BEG'] = _da1(mjd_beg, v)
    ds['MJD_MID'] = _da1(mjd_mid, v)
    ds['MJD_END'] = _da1(mjd_end, v)
    ds['XPOSURE'] = _da1(xposure, v)
    ds['CAL_VER'] = _da1(cal_ver, v)
    ds['CRDS_VER'] = _da1(crds_ver, v)
    ds['CRDS_CTX'] = _da1(crds_ctx, v)
    ds['PIPELINE'] = _da1(pipe_line, v)
    ds['PIPE_VER'] = _da1(pipe_ver, v)

    # Per-visit provenance from FITS header (R_* and S_* cards)
    for key, val in r_from_fits.items():
        ds[key] = _da1(str(val), v)
    for key, val in s_from_fits.items():
        ds[key] = _da1(str(val), v)

    # Global, visit-invariant attributes (constants + visit context)
    ds.attrs.update(
        {
            'HLSPVER': HLSPVER,
            'HLSPID': HLSPID,
            'HLSPNAME': HLSPNAME,
            'HLSP_PI': HLSP_PI,
            'HLSPLEAD': HLSPLEAD,
            'DOI': DOI,
            'STAR': STAR,
            'PLANET': PLANET,
            'HLSPTARG': PLANET,
            'OBSERVAT': observat,
            'TELESCOP': telescop,
            'INSTRUME': instrume,
            'RADESYS': RADESYS,
            'TIMESYS': TIMESYS,
            'TUNIT': TUNIT,
            'LICENSE': LICENSE,
            'LICENURL': LICENURL,
            'RA_TARG': ra_targ,
            'DEC_TARG': dec_targ,
            'PROPOSID': proposid,
            'LTT_CORR': int(bool(compute_ltt)),
        }
    )

    return ds


def save_single_eclipse_hdf5(ds, out_dir='.'):
    """
    Save the dataset to an HDF5-based netCDF file with a standard filename.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder. Must contain a single visit row.
    out_dir : str, optional
        Directory where the file will be written. Defaults to current dir.

    Returns
    -------
    out_path : str
        Path to the written file on disk.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    instrume = ds.attrs['INSTRUME']
    planet = ds.attrs['PLANET']
    visit = int(np.asarray(ds['visit'].values).item())
    filt = str(ds['filter'].values[0])
    hlspver = ds.attrs['HLSPVER']

    planet_fn = ''.join(planet.lower().split())
    out_name = (
        f'hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-'
        f'ecl{visit:03d}_{filt.lower()}_v{hlspver.lower()}_'
        f'eclipse-cat.h5'
    )
    out_path = str(Path(out_dir) / out_name)

    enc = {
        name: {'zlib': True, 'complevel': 4}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.floating)
    }
    ds.to_netcdf(
        out_path,
        engine='netcdf4',
        encoding=enc,
        mode='w',
    )
    return out_path

###############################################################################
# Code to combine multiple single-eclipse datasets into one dataset, for use
# at RWDDT checkpoints.


def _ensure_var(ds, name, vtype, visit_idx):
    """
    Ensure a variable exists with the requested type, creating if needed.

    Parameters
    ----------
    ds : xarray.Dataset
        Single-visit dataset to modify in place.
    name : str
        Variable name to ensure.
    vtype : {'float', 'str', 'datetime'}
        Desired variable type.
    visit_idx : int
        Visit index for the single row.

    Returns
    -------
    xarray.Dataset
        The input dataset (modified) is also returned for chaining.
    """
    if name in ds:
        return ds
    if vtype == 'float':
        ds[name] = _da1(np.nan, visit_idx)
    elif vtype == 'datetime':
        ds[name] = _da1(np.datetime64('NaT'), visit_idx)
    else:
        ds[name] = _da1('', visit_idx)
    return ds


def concat_eclipse_visits(datasets):
    """
    Concatenate multiple single-visit eclipse datasets along ``visit``.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Each item must be a one-row dataset with a ``visit`` coordinate.

    Returns
    -------
    combined : xarray.Dataset
        Dataset with one row per input visit. Variables are the union of all
        inputs; missing values are filled with suitable blanks.

    Notes
    -----
    - Variable fill types are inferred from the dtypes present in the input
      datasets.
    - Numeric visit fields are filled with NaN.
    - Datetime fields use NaT.
    - All provenance text (``R_*``, ``S_*``, and ``CRDS_*``) are strings.
    - Text/provenance fields are filled with empty strings.
    """
    if not datasets:
        raise ValueError('No datasets provided for concatenation.')

    for ds in datasets:
        if 'visit' not in ds.coords:
            raise ValueError('Each dataset must have a "visit" coordinate.')
        if ds.sizes.get('visit', 0) != 1:
            raise ValueError('Each dataset must be a single-visit dataset.')

    # Union of variable names across all visits.
    all_vars = set()
    for ds in datasets:
        all_vars.update(ds.data_vars)

    def _kind(name):
        dtypes = [ds[name].dtype for ds in datasets if name in ds]
        if name.startswith('R_') or name.startswith('S_'):
            return 'str'
        elif any(np.issubdtype(dtype, np.datetime64) for dtype in dtypes):
            return 'datetime'
        elif any(np.issubdtype(dtype, np.number) for dtype in dtypes):
            return 'float'
        return 'str'

    # Normalize variable sets across inputs before concatenating.
    norm = []
    for ds in datasets:
        # Get Python int visit index from coordinate.
        v = int(np.asarray(ds['visit'].values).item())
        for name in sorted(all_vars):
            ds = _ensure_var(ds, name, _kind(name), v)
        norm.append(ds)

    combined = xr.concat(norm, dim='visit', join='outer')

    # Sort rows by visit index.
    order = np.argsort(combined['visit'].values)
    combined = combined.isel(visit=order)
    combined.attrs.update(datasets[0].attrs)
    return combined


def save_multi_eclipse_hdf5(
    datasets,
    checkpoint,
    out_dir='.',
    hlspver=None,
    filtername=None,
):
    """
    Save multiple visits as a single HDF5 file named with ``checkpoint##``.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of single-visit datasets created by the builder. Each must have
        exactly one ``visit`` row.
    checkpoint : int
        Integer checkpoint number used in the filename. It is rendered as
        a two-digit, zero-padded integer (e.g., 1 -> ``checkpoint01``).
    out_dir : str, optional
        Directory where the file is written. Defaults to current directory.
    hlspver : str or None, optional
        HLSP version to embed in the filename. If ``None``, the function
        uses the ``HLSPVER`` attribute from the first dataset.
    filtername : str or None, optional
        Filter name to embed in the filename. If ``None``, the function
        uses the shared ``filter`` value from the input datasets.

    Returns
    -------
    combined : xarray.Dataset
        The concatenated multi-visit dataset that was written to disk.
    out_path : str
        Path to the written file.

    Raises
    ------
    ValueError
        If input datasets are empty, inconsistent in key invariants, or
        if HLSP version cannot be determined.

    Notes
    -----
    The output filename pattern is:

    ``hlsp_rocky-worlds_jwst_{instrume}_{planet}-checkpoint##_{filter}_``
    ``v{ver}_eclipse-cat.h5``

    Per-visit light-curve paths are not part of the name.
    """
    if not datasets:
        raise ValueError("No datasets provided.")

    # Verify visit-invariant attrs are consistent across inputs
    keys = ['INSTRUME', 'PLANET']
    vals = {k: set() for k in keys}
    for ds in datasets:
        for k in keys:
            vals[k].add(str(ds.attrs.get(k, '')))
    for k in keys:
        if len(vals[k]) != 1:
            raise ValueError(f"Inconsistent attribute across inputs: {k}")

    instrume = list(vals['INSTRUME'])[0]
    planet = list(vals['PLANET'])[0]

    # Determine HLSP version for filename
    ver = hlspver or str(datasets[0].attrs.get('HLSPVER', '')).strip()
    if not ver:
        raise ValueError("HLSP version not provided and not found in attrs.")

    # Concat visits
    combined = concat_eclipse_visits(datasets)

    if filtername is None:
        filters = {
            str(np.asarray(ds['filter'].values).item())
            for ds in datasets
        }
        if len(filters) != 1:
            raise ValueError("Inconsistent filter across inputs.")
        filtername = filters.pop()

    # Build filename with checkpoint## (two digits)
    planet_fn = ''.join(planet.lower().split())
    ckpt = f"checkpoint{int(checkpoint):02d}"
    out_name = (
        f"hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-"
        f"{ckpt}_{filtername.lower()}_v{ver.lower()}_eclipse-cat.h5"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / out_name)

    # Compression for floats
    enc = {
        name: {'zlib': True, 'complevel': 4}
        for name in combined.data_vars
        if np.issubdtype(combined[name].dtype, np.floating)
    }

    combined.to_netcdf(
        out_path,
        engine='netcdf4',
        encoding=enc,
        mode='w',
    )
    return combined, out_path

###############################################################################
# Code to convert Xarray into FITS for MAST automatic metadata ingest


def _to_ascii(text):
    """
    Normalize a string to plain ASCII for FITS header compatibility.

    Parameters
    ----------
    text : Any
        Value to convert to ASCII. Non-strings are returned unchanged.

    Returns
    -------
    str or Any
        ASCII-only string if input was a string; otherwise the original
        value.
    """
    if not isinstance(text, str):
        return text
    norm = unicodedata.normalize('NFKD', text)
    return norm.encode('ascii', 'ignore').decode('ascii')


def _set_card(hdr, key, value, comment=None):
    """
    Set a FITS header card with ASCII-safe text conversion.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        Header to be modified.
    key : str
        FITS keyword to write.
    value : Any
        Value to write. Strings are ASCII-normalized. List-like values are
        joined into one string because FITS header cards require scalars.
    comment : str or None, optional
        Optional comment for the card.
    """
    if value is None:
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        separator = '; ' if key == 'NOTES' else ' '
        value = separator.join(str(v) for v in np.asarray(value).ravel())
    if isinstance(value, str):
        value = _to_ascii(value)
    hdr[key] = (value, None if comment is None else _to_ascii(comment))


def _collect_visit_meta(ds, i):
    """
    Collect per-visit metadata from dataset for visit index ``i``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    i : int
        Row index along ``visit``.

    Returns
    -------
    meta : dict
        Mapping of per-visit keys to scalars (strings/floats).
    """
    def s(name):
        return None if name not in ds else ds[name].values[i].item()

    out = {
        'SRC_DOI': s('SRC_DOI'),
        'VISIT_ID': s('VISIT_ID'),
        'FILE_ID': s('FILE_ID'),
        'MJD_BEG': s('MJD_BEG'),
        'MJD_MID': s('MJD_MID'),
        'MJD_END': s('MJD_END'),
        'XPOSURE': s('XPOSURE'),
        'CAL_VER': s('CAL_VER'),
        'CRDS_VER': s('CRDS_VER'),
        'CRDS_CTX': s('CRDS_CTX'),
        'PIPELINE': s('PIPELINE'),
        'PIPE_VER': s('PIPE_VER'),
        'FILTER': s('filter'),
        'SUBARRAY': s('subarray'),
    }
    if 'date_obs' in ds:
        # date_obs may be datetime64; render it as an ISO string for FITS.
        val = ds['date_obs'].values[i]
        out['DATE-OBS'] = (np.datetime_as_string(val)
                           if not np.isnat(val) else '')
    return out


def _r_s_pairs_for_visit(ds, i):
    """
    Return (R_*, S_*) pairs for visit ``i`` from dataset variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    i : int
        Row index along ``visit``.

    Returns
    -------
    r_pairs : list of tuple
        List of (key, str_value) for R_* cards.
    s_pairs : list of tuple
        List of (key, str_value) for S_* cards.
    """
    r_pairs, s_pairs = [], []
    for name in ds.data_vars:
        if name.startswith('R_') or name.startswith('S_'):
            val = ds[name].values[i].item()
            pair = (name, '' if val is None else str(val))
            if name.startswith('R_'):
                r_pairs.append(pair)
            else:
                s_pairs.append(pair)
    r_pairs.sort()
    s_pairs.sort()
    return r_pairs, s_pairs


def _build_measurements_hdu(ds):
    """
    Build the MEASUREMENTS binary table from dataset variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Single- or multi-visit dataset produced by the builder.

    Returns
    -------
    hdu : astropy.io.fits.BinTableHDU
        Table with one row per visit containing numeric results and a few
        context strings.

    Notes
    -----
    This rigid version assumes all required variables exist in ``ds``.
    """
    n = ds.sizes['visit']

    # Fixed units from variable attrs (no presence checks)
    units = {
        'ECLIPSEDEPTH': ds['eclipseDepth'].attrs.get('units', ''),
        'ECLIPSEDEPTHERROR': ds['eclipseDepthError'].attrs.get('units', ''),
        'ECLIPSEDEPTHUPPERERROR':
        ds['eclipseDepthUpperError'].attrs.get('units', ''),
        'ECLIPSEDEPTHLOWERERROR':
        ds['eclipseDepthLowerError'].attrs.get('units', ''),
        'ECLIPSETIME': ds['eclipseTime'].attrs.get('units', ''),
        'ECLIPSETIMEERROR': ds['eclipseTimeError'].attrs.get('units', ''),
        'ECLIPSETIMEUPPERERROR':
        ds['eclipseTimeUpperError'].attrs.get('units', ''),
        'ECLIPSETIMELOWERERROR':
        ds['eclipseTimeLowerError'].attrs.get('units', ''),
        'ABSFLUX': ds['absFlux'].attrs.get('units', ''),
        'ABSFLUXERROR': ds['absFluxError'].attrs.get('units', ''),
        'ABSFLUXUPPERERROR': ds['absFluxUpperError'].attrs.get('units', ''),
        'ABSFLUXLOWERERROR': ds['absFluxLowerError'].attrs.get('units', ''),
        'MJD_BEG': 'BMJD_TDB',
        'MJD_MID': 'BMJD_TDB',
        'MJD_END': 'BMJD_TDB',
        'XPOSURE': 's',
    }

    # Helper for float columns (format 'D' = float64)
    def f(name):
        up = name.upper()
        arr = ds[name].values.astype(np.float64)
        unit = _to_ascii(units.get(up, '')) or None
        return fits.Column(name=up, array=arr, format='D', unit=unit)

    cols = [
        f('eclipseDepth'),
        f('eclipseDepthError'),
        f('eclipseDepthUpperError'),
        f('eclipseDepthLowerError'),
        f('eclipseTime'),
        f('eclipseTimeError'),
        f('eclipseTimeUpperError'),
        f('eclipseTimeLowerError'),
        f('absFlux'),
        f('absFluxError'),
        f('absFluxUpperError'),
        f('absFluxLowerError'),
        f('MJD_BEG'),
        f('MJD_MID'),
        f('MJD_END'),
        f('XPOSURE'),
    ]

    # Helper for string columns (variable width up to 64)
    def scol(name, outname):
        vals = []
        if name == 'DATEOBS':
            for i in range(n):
                t = ds['date_obs'].values[i]
                s = np.datetime_as_string(t) if not np.isnat(t) else ''
                vals.append(_to_ascii(s))
        else:
            for i in range(n):
                v = ds[name].values[i].item()
                vals.append(_to_ascii('' if v is None else str(v)))
        width = max(1, min(64, max(len(v) for v in vals)))
        return fits.Column(name=outname, array=np.asarray(vals),
                           format=f'A{width}')

    cols += [
        scol('VISIT_ID', 'VISIT_ID'),
        scol('filter', 'FILTER'),
        scol('subarray', 'SUBARRAY'),
        scol('DATEOBS', 'DATEOBS'),
    ]

    return fits.BinTableHDU.from_columns(cols, name='MEASUREMENTS')


def _primary_hdu_from_attrs(ds, visit_index=None):
    """
    Construct a PRIMARY HDU from dataset attributes (+ optional visit meta).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    visit_index : int or None, optional
        If given, include per-visit metadata for that row in PRIMARY.
        If ``None``, include only visit-invariant HLSP attributes.

    Returns
    -------
    hdu : astropy.io.fits.PrimaryHDU
        Primary HDU with populated header and no data.
    """
    hdr = fits.Header()
    # HLSP/target invariants
    for key in [
        'HLSPVER', 'HLSPID', 'HLSPNAME', 'HLSP_PI', 'HLSPLEAD', 'DOI',
        'STAR', 'PLANET', 'HLSPTARG', 'OBSERVAT', 'TELESCOP', 'INSTRUME',
        'RADESYS', 'TIMESYS', 'TUNIT', 'LICENSE', 'LICENURL', 'RA_TARG',
        'DEC_TARG', 'PROPOSID', 'LTT_CORR', 'NOTES',
    ]:
        _set_card(hdr, key, ds.attrs.get(key))
    # Single-visit: include per-visit metadata in PRIMARY
    if visit_index is not None:
        meta = _collect_visit_meta(ds, visit_index)
        for k, v in meta.items():
            _set_card(hdr, k, v)
        # Also include all R_*/S_* cards
        r_pairs, s_pairs = _r_s_pairs_for_visit(ds, visit_index)
        for k, v in r_pairs + s_pairs:
            _set_card(hdr, k, v)
    return fits.PrimaryHDU(header=hdr)


def _obs_hdu_for_visit(ds, i):
    """
    Build a header-only table HDU named OBS### for visit ``i``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by the builder.
    i : int
        Visit row index.

    Returns
    -------
    hdu : astropy.io.fits.BinTableHDU
        Empty binary table with header cards carrying visit metadata and
        provenance. Name is ``OBS###`` (1-based, zero-padded).
    """
    # Empty table; header-only payload
    hdu = fits.BinTableHDU.from_columns([], nrows=0)
    hdu.header['EXTNAME'] = f'OBS{(i+1):03d}'
    meta = _collect_visit_meta(ds, i)
    for k, v in meta.items():
        _set_card(hdu.header, k, v)
    r_pairs, s_pairs = _r_s_pairs_for_visit(ds, i)
    for k, v in r_pairs + s_pairs:
        _set_card(hdu.header, k, v)
    return hdu


def hdf5_single_to_fits(h5_path):
    """
    Convert a single-visit HDF5 eclipse-cat file to a FITS file.

    Parameters
    ----------
    h5_path : str
        Path to a single-visit HDF5 produced by the builder.

    Returns
    -------
    out_path : str
        The path written.

    Raises
    ------
    ValueError
        If the input dataset is not a single-visit dataset.
    """
    ds = xr.load_dataset(h5_path)
    n = ds.sizes.get('visit', 0)
    if n != 1:
        raise ValueError('Input is not a single-visit dataset.')
    phdu = _primary_hdu_from_attrs(ds, visit_index=0)
    meas = _build_measurements_hdu(ds)
    hdul = fits.HDUList([phdu, meas])
    out_path = str(Path(h5_path).with_suffix('.fits'))
    hdul.writeto(out_path, overwrite=True)
    return out_path


def hdf5_checkpoint_to_fits(h5_path):
    """
    Convert a multi-visit checkpoint HDF5 file to a FITS file.

    Parameters
    ----------
    h5_path : str
        Path to a multi-visit HDF5 produced by concatenation.

    Returns
    -------
    out_path : str
        The path written.

    Raises
    ------
    ValueError
        If the input dataset has fewer than two visits.
    """
    ds = xr.load_dataset(h5_path)
    n = ds.sizes.get('visit', 0)
    if n < 2:
        raise ValueError('Input is not multi-visit (>=2 required).')
    phdu = _primary_hdu_from_attrs(ds, visit_index=None)
    hdus = [phdu]
    # One header-only OBS### per visit
    for i in range(n):
        hdus.append(_obs_hdu_for_visit(ds, i))
    # Measurements table with all visits
    hdus.append(_build_measurements_hdu(ds))
    out_path = str(Path(h5_path).with_suffix('.fits'))
    fits.HDUList(hdus).writeto(out_path, overwrite=True)
    return out_path


###############################################################################
# Code to make and save the light curve xarray/HDF5 datasets

def _series1(values, time, visit_idx, units=None, decimals=None):
    """
    Create a 1xN (visit, time) DataArray for a light-curve series.

    Parameters
    ----------
    values : array-like
        1-D time series values.
    time : array-like
        1-D time coordinate (same length as ``values``).
    visit_idx : int
        Visit index to use for the ``visit`` coordinate.
    units : str or None, optional
        Units stored in ``attrs['units']``.
    decimals : int or None, optional
        If given, round the data to this many decimals.

    Returns
    -------
    xarray.DataArray
        DataArray with dims ``('time',)`` expanded to ``('visit','time')``.
    """
    arr = np.asarray(values)
    if decimals is not None:
        arr = np.round(arr, decimals)
    da = xr.DataArray(arr, coords={'time': time}, dims=['time'])
    da = da.expand_dims(visit=[visit_idx])
    if units is not None:
        da.attrs['units'] = units
    return da


def _time_group_slices(time, gap_factor=5.0, min_gap_days=None):
    """
    Split a time axis into contiguous observation-like groups.

    Parameters
    ----------
    time : array-like
        Monotonic 1-D time axis.
    gap_factor : float, optional
        Multiplicative factor above the median cadence used to identify a
        discontinuity between observations.
    min_gap_days : float or None, optional
        Optional absolute minimum gap in days. If ``None``, no absolute
        threshold is applied.

    Returns
    -------
    groups : list of numpy.ndarray
        Integer index arrays, one for each contiguous time group.
    """
    arr = np.asarray(time, dtype=float)
    if arr.size == 0:
        return []
    if arr.size == 1:
        return [np.array([0], dtype=int)]

    dt = np.diff(arr)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if finite_dt.size == 0:
        return [np.arange(arr.size, dtype=int)]

    cadence = np.nanmedian(finite_dt)
    gap_threshold = cadence * float(gap_factor)
    if min_gap_days is not None:
        gap_threshold = max(gap_threshold, float(min_gap_days))

    breaks = np.where(dt > gap_threshold)[0] + 1
    edges = np.concatenate(([0], breaks, [arr.size]))

    return [
        np.arange(edges[i], edges[i + 1], dtype=int)
        for i in range(len(edges) - 1)
    ]


def _normalize_raw_flux_by_time_groups(
    raw_flux,
    raw_err,
    full_time,
    fit_time,
    fit_flux,
    gap_factor=5.0,
):
    """
    Normalize raw Stage-4 flux separately for each observation-like group.

    Parameters
    ----------
    raw_flux : array-like
        Full Stage-4 raw flux array.
    raw_err : array-like
        Full Stage-4 raw flux uncertainty array.
    full_time : array-like
        Full Stage-4 time axis, in the same units as ``fit_time``.
    fit_time : array-like
        Stage-5 fit-table time axis.
    fit_flux : array-like
        Stage-5 normalized light-curve flux values.
    gap_factor : float, optional
        Factor above the median cadence used to split observation groups.

    Returns
    -------
    raw_flux_norm : numpy.ndarray
        Raw flux divided by one scale factor per time group.
    raw_err_norm : numpy.ndarray
        Raw flux errors divided by the same per-group scale factors.

    Notes
    -----
    A single eclipse visit can include multiple close-in-time JWST
    observations. The Stage-5 fit table normalizes each observation
    separately, so the Stage-4 raw flux and error must be divided by a
    separate scale factor in each observation group rather than by one
    scalar over the entire visit. If Stage 5 only fitted a subset of those
    observations, groups with no fitted integrations use the nearest
    fitted group's scale as a fallback so they are still delivered in
    normalized-flux units.
    """
    raw_flux = np.asarray(raw_flux, dtype=float)
    raw_err = np.asarray(raw_err, dtype=float)
    full_time = np.asarray(full_time, dtype=float)
    fit_time = np.asarray(fit_time, dtype=float)
    fit_flux = np.asarray(fit_flux, dtype=float)

    raw_flux_norm = np.full_like(raw_flux, np.nan, dtype=float)
    raw_err_norm = np.full_like(raw_err, np.nan, dtype=float)

    groups = _time_group_slices(full_time, gap_factor=gap_factor)
    if not groups:
        return raw_flux_norm, raw_err_norm

    # Rounded times can have exact matches after the notebook's time
    # rounding. Use an index map for deterministic reindexing.
    fit_df = pd.DataFrame({'flux': fit_flux}, index=fit_time)
    fit_df = fit_df[~fit_df.index.duplicated(keep='first')]

    scales = np.full(len(groups), np.nan, dtype=float)
    centers = np.full(len(groups), np.nan, dtype=float)

    for i, group in enumerate(groups):
        group_time = full_time[group]
        centers[i] = np.nanmedian(group_time)
        y = fit_df.reindex(group_time)['flux'].values
        x = raw_flux[group]

        finite = np.isfinite(x) & np.isfinite(y)
        denom = np.dot(x[finite], y[finite])
        numer = np.dot(x[finite], x[finite])

        if finite.sum() == 0 or not np.isfinite(denom) or denom == 0.0:
            scale = np.nan
        else:
            scale = numer / denom

        if np.isfinite(scale) and scale != 0.0:
            scales[i] = scale

    valid_scales = np.isfinite(scales) & (scales != 0.0)

    for i, group in enumerate(groups):
        scale = scales[i]
        if not np.isfinite(scale) or scale == 0.0:
            if np.any(valid_scales) and np.isfinite(centers[i]):
                dist = np.abs(centers[valid_scales] - centers[i])
                scale = scales[valid_scales][np.argmin(dist)]

        if not np.isfinite(scale) or scale == 0.0:
            raw_flux_norm[group] = raw_flux[group]
            raw_err_norm[group] = raw_err[group]
        else:
            raw_flux_norm[group] = raw_flux[group] / scale
            raw_err_norm[group] = raw_err[group] / scale

    return raw_flux_norm, raw_err_norm


def _match_stage4_errors_to_fit_lcerr(
    raw_err_stage4_norm,
    full_time,
    fit_time,
    fit_err,
    gap_factor=5.0,
):
    """
    Match scaled Stage-4 errors to Stage-5 ``lcerr`` per time group.

    Parameters
    ----------
    raw_err_stage4_norm : array-like
        Stage-4 error array after the raw-flux normalization divisor has
        been applied.
    full_time : array-like
        Full Stage-4 time axis, in the same units as ``fit_time``.
    fit_time : array-like
        Stage-5 fit-table time axis.
    fit_err : array-like
        Stage-5 fit-table normalized light-curve uncertainties.
    gap_factor : float, optional
        Factor above the median cadence used to split observation groups.

    Returns
    -------
    raw_err_matched : numpy.ndarray
        Stage-4-normalized errors multiplied by the per-group median ratio
        ``lcerr / raw_err_stage4_norm`` measured on retained integrations.
        Groups with no retained integrations use the nearest fitted group's
        ratio as a fallback.

    Notes
    -----
    Eureka! can apply an additional multiplicative uncertainty scaling during
    the Stage-5 fit. The Stage-5 table's ``lcerr`` already includes that
    scaling, but only for retained integrations. This helper infers the
    corresponding multiplicative factor from retained integrations in each
    observation-like time group and applies it to dropped integrations too.
    If Stage 5 fitted only a subset of observation groups, unfitted groups
    use the nearest fitted group's uncertainty scale.
    """
    raw_err_stage4_norm = np.asarray(raw_err_stage4_norm, dtype=float)
    full_time = np.asarray(full_time, dtype=float)
    fit_time = np.asarray(fit_time, dtype=float)
    fit_err = np.asarray(fit_err, dtype=float)

    raw_err_matched = raw_err_stage4_norm.copy()
    fit_df = pd.DataFrame({'err': fit_err}, index=fit_time)
    fit_df = fit_df[~fit_df.index.duplicated(keep='first')]

    groups = _time_group_slices(full_time, gap_factor=gap_factor)
    err_scales = np.full(len(groups), np.nan, dtype=float)
    centers = np.full(len(groups), np.nan, dtype=float)

    for i, group in enumerate(groups):
        group_time = full_time[group]
        centers[i] = np.nanmedian(group_time)
        fit_err_group = fit_df.reindex(group_time)['err'].values
        stage4_err_group = raw_err_stage4_norm[group]
        finite = (
            np.isfinite(fit_err_group)
            & np.isfinite(stage4_err_group)
            & (stage4_err_group > 0.0)
        )
        if not np.any(finite):
            continue
        ratios = fit_err_group[finite] / stage4_err_group[finite]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
        if ratios.size == 0:
            continue
        scale = float(np.nanmedian(ratios))
        if np.isfinite(scale) and scale > 0.0:
            err_scales[i] = scale

    valid_scales = np.isfinite(err_scales) & (err_scales > 0.0)

    for i, group in enumerate(groups):
        scale = err_scales[i]
        if not np.isfinite(scale) or scale <= 0.0:
            if np.any(valid_scales) and np.isfinite(centers[i]):
                dist = np.abs(centers[valid_scales] - centers[i])
                scale = err_scales[valid_scales][np.argmin(dist)]
        if np.isfinite(scale) and scale > 0.0:
            raw_err_matched[group] = raw_err_matched[group] * scale

    return raw_err_matched


def build_lightcurve_dataset(
    stage2_fits,
    stage3_specdata,
    stage4_lcdata,
    stage5_fit_table,
    visit,
    STAR,
    PLANET,
    SRC_DOI,
    HLSPVER,
):
    """
    Build a single-visit light-curve Dataset (multi-visit ready).

    The time axis is taken from the full Stage-4 LCData product. Series
    from the Stage-5 fit table are aligned onto that axis, with NaNs
    where the fit dropped integrations. Stage-4 raw flux is scaled by
    the same multiplicative divisor used in Stage-5, and its errors are
    divided by that same factor.

    Parameters
    ----------
    stage2_fits : str
        Path to the Stage 2 FITS file.
    stage3_specdata : str
        Path to Stage 3 product with ``segment_list``.
    stage4_lcdata : str
        Path to Stage 4 light-curve HDF5 (full time and centroids).
    stage5_fit_table : str
        Path to Stage 5 table (text) with columns:
        ``time``, ``lcdata``, ``lcerr``, ``astrophysical model``,
        ``model``, ``GP``, ``residuals``.
    visit : int
        Eclipse index you assign (1, 2, ...) used for the ``visit`` coord
        and in output filenames. Not the JWST VISIT_ID.
    STAR : str
        Host star identifier (per visit).
    PLANET : str
        Planet identifier (per visit).
    SRC_DOI : str
        DOI for the source data specific to this visit.
    HLSPVER : str
        HLSP version string to store in the dataset attributes.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with dims ``(visit, time)`` for series and per-visit
        metadata as 1× columns. Ready for later concatenation on
        ``visit``.
    """
    # Load inputs
    spec_ds = xr.load_dataset(stage3_specdata)
    lc = xr.load_dataset(stage4_lcdata)
    dm = JwstDataModel(stage2_fits)
    fit = pd.read_csv(stage5_fit_table, sep=r'\s+', comment='#')

    # Provenance/context from Stage-2 header
    (
        r_from_fits,
        s_from_fits,
        crds_ver_hdr,
        crds_ctx_hdr,
        filt_hdr,
        subarray_hdr,
        date_obs,
    ) = read_stage2_r_s_meta(stage2_fits)

    instrume = dm.meta.instrument.name
    filter_name = filt_hdr or dm.meta.instrument.filter
    subarray = subarray_hdr or getattr(dm.meta.subarray, 'name', '')
    telescop = dm.meta.telescope
    observat = dm.meta.telescope

    proposid = dm.meta.observation.program_number
    ra_targ = dm.meta.target.ra
    dec_targ = dm.meta.target.dec

    # FILE_ID list and VISIT_IDs derived from Stage-3 segment_list
    base_ids = []
    if hasattr(spec_ds, 'segment_list'):
        for seg in spec_ds.segment_list:
            name = Path(seg).name
            base_ids.append(name.split('_calints')[0])
    file_id_str = ','.join(base_ids)
    visit_id_str = _derive_visit_id_string(base_ids)

    # Exposure timing
    mjd_beg = float(dm.meta.exposure.start_time_tdb)
    mjd_mid = float(dm.meta.exposure.mid_time_tdb)
    mjd_end = float(dm.meta.exposure.end_time_tdb)
    xposure = float(dm.meta.exposure.effective_exposure_time)

    # Software versions
    cal_ver = str(dm.meta.calibration_software_version)
    pipe_line = str(getattr(spec_ds, 'data_format', ''))
    pipe_ver = str(getattr(spec_ds, 'version', ''))
    crds_ver = str(
        crds_ver_hdr or getattr(dm.meta.ref_file.crds, 'sw_version', '')
    )
    crds_ctx = str(
        crds_ctx_hdr or getattr(dm.meta.ref_file.crds, 'context_used', '')
    )

    # Full time axis from Stage-4 LCData (rounded and offset)
    if 'time' not in lc.coords and 'time' not in lc:
        raise KeyError("Stage-4 LCData must contain a 'time' coordinate.")
    lc_time = lc.coords.get('time', lc['time']).values
    full_time = np.round(lc_time + timeOffset, timeDecimals)

    # Stage-5 fit time (subset) aligned to full_time
    fit_time = np.round(fit['time'].values + timeOffset, timeDecimals)

    def _to_full(values):
        s = pd.Series(np.asarray(values), index=fit_time)
        s = s[~s.index.duplicated(keep='first')]
        return s.reindex(full_time).values

    # ---- Series on full_time ----
    v = int(visit)

    # RAW FLUX from Stage-4 'data' (flatten), scaled by the same
    # observation-local divisors used in Stage 5.
    lc_raw_full = np.ravel(lc['data'].values).astype(float)
    lc_err_full = np.ravel(lc['err'].values).astype(float)

    raw_full_norm, raw_err_stage4_norm = _normalize_raw_flux_by_time_groups(
        lc_raw_full,
        lc_err_full,
        full_time,
        fit_time,
        fit['lcdata'].values,
    )

    # Prefer the Stage-5 fit-table uncertainties where they exist, because
    # those are paired with the delivered normalized ``lcdata`` values. The
    # fit table only includes integrations retained by Stage 5, though. For
    # dropped integrations, use Stage-4 errors after both the raw-flux
    # normalization and the Stage-5 multiplicative uncertainty rescaling.
    if 'lcerr' in fit.keys():
        raw_err_fit = _to_full(fit['lcerr'].values)
        raw_err_stage4_matched = _match_stage4_errors_to_fit_lcerr(
            raw_err_stage4_norm,
            full_time,
            fit_time,
            fit['lcerr'].values,
        )
        raw_err_norm = np.where(
            np.isfinite(raw_err_fit),
            raw_err_fit,
            raw_err_stage4_matched,
        )
    else:
        raw_err_norm = raw_err_stage4_norm

    raw_flux = _series1(
        np.round(raw_full_norm, lightcurveDecimals),
        full_time, v, units=lightcurve_units, decimals=None,
    )
    raw_flux.name = 'rawFlux'

    raw_flux_err = _series1(
        np.round(raw_err_norm, lightcurveDecimals),
        full_time, v, units=lightcurve_units, decimals=None,
    )
    raw_flux_err.name = 'rawFluxErr'

    # Stage-5 models reindexed to full_time (NaN where dropped)
    astro = _series1(
        _to_full(fit['astrophysical model'].values),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    astro.name = 'astroModel'

    if 'GP' in fit.keys():
        GP = fit['GP'].values
    else:
        GP = np.zeros_like(fit['model'].values)
    noise_vals = fit['model'].values/fit['astrophysical model'].values + GP
    noise = _series1(
        _to_full(noise_vals),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    noise.name = 'noiseModel'

    full_vals = fit['model'].values + GP
    full = _series1(
        _to_full(full_vals),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    full.name = 'fullModel'

    cleaned_vals = (
        fit['astrophysical model'].values + (fit['residuals'].values - GP)
    )
    cleaned = _series1(
        _to_full(cleaned_vals),
        full_time, v, units=lightcurve_units, decimals=lightcurveDecimals,
    )
    cleaned.name = 'cleanedFlux'

    # Centroid diagnostics from Stage-4 (on the full time axis)
    x = lc['centroid_x'].values + np.nanmin(spec_ds.x.values)
    y = lc['centroid_y'].values + np.nanmin(spec_ds.y.values)
    cenx = _series1(x, full_time, v, units='pix')
    ceny = _series1(y, full_time, v, units='pix')
    censx = _series1(lc['centroid_sx'].values, full_time, v, units='pix')
    censy = _series1(lc['centroid_sy'].values, full_time, v, units='pix')

    cenx.name, ceny.name = 'centroid_x', 'centroid_y'
    censx.name, censy.name = 'centroid_sx', 'centroid_sy'

    # Assemble dataset
    ds = xr.Dataset(
        {
            'rawFlux': raw_flux,
            'rawFluxErr': raw_flux_err,
            'astroModel': astro,
            'noiseModel': noise,
            'fullModel': full,
            'cleanedFlux': cleaned,
            'centroid_x': cenx,
            'centroid_y': ceny,
            'centroid_sx': censx,
            'centroid_sy': censy,
        }
    )

    # Set coordinates explicitly
    ds = ds.assign_coords(visit=[v])
    ds = ds.assign_coords(time=full_time)
    ds.time.attrs['units'] = time_units

    # Per-visit metadata as columns (1× strings/floats)
    ds['SRC_DOI'] = _da1(str(SRC_DOI), v)
    ds['VISIT_ID'] = _da1(visit_id_str, v)
    ds['FILE_ID'] = _da1(file_id_str, v)
    ds['MJD_BEG'] = _da1(mjd_beg, v)
    ds['MJD_MID'] = _da1(mjd_mid, v)
    ds['MJD_END'] = _da1(mjd_end, v)
    ds['XPOSURE'] = _da1(xposure, v)
    ds['CAL_VER'] = _da1(cal_ver, v)
    ds['CRDS_VER'] = _da1(crds_ver, v)
    ds['CRDS_CTX'] = _da1(crds_ctx, v)
    ds['PIPELINE'] = _da1(pipe_line, v)
    ds['PIPE_VER'] = _da1(pipe_ver, v)
    ds['filter'] = _da1(str(filter_name), v)
    ds['subarray'] = _da1(str(subarray), v)
    ds['date_obs'] = _da1(date_obs, v)

    # Optional: include R_*/S_* cards for provenance completeness
    for key, val in r_from_fits.items():
        ds[key] = _da1(str(val), v)
    for key, val in s_from_fits.items():
        ds[key] = _da1(str(val), v)

    # Global invariants
    ds.attrs.update(
        {
            'HLSPVER': HLSPVER,
            'HLSPID': HLSPID,
            'HLSPNAME': HLSPNAME,
            'HLSP_PI': HLSP_PI,
            'HLSPLEAD': HLSPLEAD,
            'DOI': DOI,
            'STAR': STAR,
            'PLANET': PLANET,
            'HLSPTARG': PLANET,
            'OBSERVAT': observat,
            'TELESCOP': telescop,
            'INSTRUME': instrume,
            'RADESYS': RADESYS,
            'TIMESYS': TIMESYS,
            'LICENSE': LICENSE,
            'LICENURL': LICENURL,
            'RA_TARG': ra_targ,
            'DEC_TARG': dec_targ,
            'PROPOSID': proposid,
        }
    )

    return ds


def save_lightcurve_hdf5(ds, out_dir='.'):
    """
    Save the light-curve dataset to an HDF5-based netCDF file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by :func:`build_lightcurve_dataset`.
    out_dir : str, optional
        Directory where the file will be written. Defaults to current dir.

    Returns
    -------
    out_path : str
        Path to the written file on disk.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    instrume = ds.attrs['INSTRUME']
    planet = ds.attrs['PLANET']
    visit = int(np.asarray(ds['visit'].values).item())
    filt = str(ds['filter'].values[0])
    hlspver = ds.attrs['HLSPVER']

    planet_fn = ''.join(planet.lower().split())
    out_name = (
        f'hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-'
        f'ecl{visit:03d}_{filt.lower()}_v{hlspver.lower()}_lc.h5'
    )
    out_path = str(Path(out_dir) / out_name)

    # Compress floating arrays
    enc = {
        name: {'zlib': True, 'complevel': 4}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.floating)
    }
    ds.to_netcdf(
        out_path,
        engine='netcdf4',
        encoding=enc,
        mode='w',
    )
    return out_path


def _float_encoding(ds):
    """
    Build compression encoding for floating-point data variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose floating-point data variables should be compressed.

    Returns
    -------
    dict
        NetCDF encoding dictionary for floating-point data variables.
    """
    return {
        name: {'zlib': True, 'complevel': 4}
        for name in ds.data_vars
        if np.issubdtype(ds[name].dtype, np.floating)
    }


def build_lightcurve_datatree(datasets):
    """
    Build a DataTree for a checkpoint light-curve product.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Single-visit light-curve datasets produced by
        :func:`build_lightcurve_dataset`. Each item must have one ``visit``
        row and its own native ``time`` coordinate.

    Returns
    -------
    tree : xarray.DataTree
        Root node stores shared HLSP attributes. Each child node is named
        ``/visit_###`` and stores one visit light curve without padding onto
        the time axis of any other visit. Child-node dataset attributes are
        cleared to avoid duplicating root-level metadata.
    """
    if not hasattr(xr, 'DataTree'):
        raise ImportError(
            'Checkpoint light-curve products require xarray.DataTree. '
            'Please use a recent xarray release with DataTree support.'
        )
    if not datasets:
        raise ValueError('No datasets provided.')

    first = datasets[0]
    tree_nodes = {'/': xr.Dataset(attrs=dict(first.attrs))}

    for ds in datasets:
        if 'visit' not in ds.coords or 'time' not in ds.dims:
            raise ValueError("Each dataset must have 'visit' coord "
                             "and 'time'.")
        if ds.sizes.get('visit', 0) != 1:
            raise ValueError('Each dataset must be single-visit (visit=1).')

        visit_idx = int(np.asarray(ds['visit'].values).item())
        child = ds.copy(deep=False)
        child.attrs = {}
        tree_nodes[f'/visit_{visit_idx:03d}'] = child

    return xr.DataTree.from_dict(tree_nodes)


def _datatree_encoding(tree):
    """
    Build a nested DataTree encoding dictionary.

    Parameters
    ----------
    tree : xarray.DataTree
        DataTree to be written with ``DataTree.to_netcdf``.

    Returns
    -------
    dict
        Nested encoding dictionary keyed by DataTree group path.
    """
    encoding = {}
    for node in tree.subtree:
        ds = node.to_dataset(inherit=False)
        enc = _float_encoding(ds)
        if enc:
            encoding[node.path] = enc
    return encoding


def save_lightcurve_multi_hdf5(
    datasets,
    checkpoint,
    out_dir='.',
    hlspver=None,
    filtername=None,
):
    """
    Save checkpoint light curves as one grouped netCDF4/HDF5 file.

    Parameters
    ----------
    datasets : list of xarray.Dataset or xarray.DataTree
        Single-visit light-curve datasets, or an already-built DataTree from
        :func:`build_lightcurve_datatree`.
    checkpoint : int
        Integer checkpoint number used in the filename, written as
        ``checkpoint##`` (two digits).
    out_dir : str, optional
        Directory where the file is written. Defaults to current directory.
    hlspver : str or None, optional
        HLSP version for the filename. If None, uses the root attrs.
    filtername : str or None, optional
        Filter name to embed in the filename. Required when ``datasets`` is
        already a DataTree; otherwise inferred from the input datasets.

    Returns
    -------
    out_path : str
        Path to the written file.

    Notes
    -----
    The on-disk product is a valid netCDF4/HDF5 file with one group per
    visit. This avoids forcing disjoint eclipses onto a sparse rectangular
    ``(visit, time)`` array while keeping all visits in one file.
    """
    if hasattr(xr, 'DataTree') and isinstance(datasets, xr.DataTree):
        tree = datasets
    else:
        if filtername is None:
            filters = {
                str(np.asarray(ds['filter'].values).item())
                for ds in datasets
            }
            if len(filters) != 1:
                raise ValueError('Inconsistent filter across inputs.')
            filtername = filters.pop()
        tree = build_lightcurve_datatree(datasets)

    attrs = tree.attrs
    instrume = attrs['INSTRUME']
    planet = attrs['PLANET']
    ver = hlspver or attrs.get('HLSPVER', '')
    if not ver:
        raise ValueError('HLSP version missing; pass hlspver or set attrs.')
    if not filtername:
        raise ValueError(
            'Filter name missing; pass filtername when saving a DataTree.'
        )

    planet_fn = ''.join(planet.lower().split())
    ckpt = f'checkpoint{int(checkpoint):02d}'
    out_name = (
        f'hlsp_rocky-worlds_jwst_{instrume.lower()}_{planet_fn}-'
        f'{ckpt}_{filtername.lower()}_v{ver.lower()}_lc.h5'
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / out_name)

    tree.to_netcdf(
        out_path,
        engine='netcdf4',
        encoding=_datatree_encoding(tree),
        mode='w',
    )
    return out_path
###############################################################################
# High-level builders for shared-fit single-eclipse and checkpoint products.


def _normalize_control_key(name):
    """
    Normalize a Eureka control-file key for dictionary lookup.

    Parameters
    ----------
    name : str
        Raw key name from an EPF/ECF file.

    Returns
    -------
    str
        Lower-case key with hyphens converted to underscores.
    """
    return str(name).strip().replace('-', '_').lower()


def _parse_scalar(value):
    """
    Parse a simple scalar from a Eureka control file.

    Parameters
    ----------
    value : str
        Text value to parse.

    Returns
    -------
    object
        ``bool``, ``None``, ``float``, or stripped string.
    """
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    low = text.lower()
    if low in {'true', 't', 'yes'}:
        return True
    if low in {'false', 'f', 'no'}:
        return False
    if low in {'none', 'null'}:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def parse_epf(epf_path):
    """
    Parse a Eureka Stage-5 parameter file.

    Parameters
    ----------
    epf_path : str or None
        Path to the ``.epf`` file. If ``None`` or empty, an empty
        dictionary is returned.

    Returns
    -------
    params : dict
        Mapping from normalized parameter name to a record with keys
        ``name``, ``value``, ``free``, ``prior_par1``, ``prior_par2``, and
        ``prior_type`` where available.
    """
    import shlex

    if epf_path in {None, ''}:
        return {}

    params = {}
    with open(epf_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            try:
                parts = shlex.split(line, comments=False, posix=True)
            except ValueError:
                continue
            if len(parts) < 2:
                continue

            name = parts[0]
            record = {
                'name': name,
                'value': _parse_scalar(parts[1]),
                'free': parts[2].strip('"\'') if len(parts) > 2 else '',
                'prior_par1': _parse_scalar(parts[3])
                if len(parts) > 3 else None,
                'prior_par2': _parse_scalar(parts[4])
                if len(parts) > 4 else None,
                'prior_type': parts[5].strip('"\'') if len(parts) > 5 else '',
            }
            params[_normalize_control_key(name)] = record
    return params


def parse_ecf(ecf_path):
    """
    Parse simple key/value pairs from a Eureka control file.

    Parameters
    ----------
    ecf_path : str or None
        Path to the ``.ecf`` file. If ``None`` or empty, an empty
        dictionary is returned.

    Returns
    -------
    config : dict
        Mapping from normalized key to parsed scalar value.

    Notes
    -----
    This parser intentionally handles the simple forms used by Eureka ECF
    files, including ``key value`` and ``key = value`` lines. Comments and
    blank lines are ignored.
    """
    import shlex

    if ecf_path in {None, ''}:
        return {}

    config = {}
    with open(ecf_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
            else:
                try:
                    parts = shlex.split(line, comments=False, posix=True)
                except ValueError:
                    continue
                if len(parts) < 2:
                    continue
                key = parts[0]
                value = parts[1]

            config[_normalize_control_key(key)] = _parse_scalar(value)
    return config


def get_compute_ltt(ecf_config=None, default=True):
    """
    Return the effective ``compute_ltt`` value for an eclipse fit.

    Parameters
    ----------
    ecf_config : dict or None, optional
        Parsed ECF dictionary from :func:`parse_ecf`.
    default : bool, optional
        Value to use when ``compute_ltt`` is absent. Defaults to ``True``
        for secondary-eclipse products.

    Returns
    -------
    bool
        Whether to include the light-travel-time correction.
    """
    if not ecf_config:
        return bool(default)
    val = ecf_config.get('compute_ltt', default)
    return bool(val)


def _load_samples_dataset(samples_path):
    """
    Load a Stage-5 samples HDF5 file as an xarray Dataset.

    Parameters
    ----------
    samples_path : str or xarray.Dataset
        Path to the samples file or an already-loaded Dataset.

    Returns
    -------
    samples : xarray.Dataset
        Dataset containing posterior samples.
    """
    if isinstance(samples_path, xr.Dataset):
        return samples_path

    try:
        return xr.load_dataset(samples_path)
    except Exception:
        # Some Eureka sample files are plain HDF5 datasets at the root rather
        # than fully self-describing netCDF files. Fall back to h5py.
        import h5py

        arrays = {}
        with h5py.File(samples_path, 'r') as handle:
            for name, obj in handle.items():
                if hasattr(obj, 'shape'):
                    arrays[name] = (['draw'], np.asarray(obj[...]))
        return xr.Dataset(arrays)


def _infer_n_samples(samples):
    """
    Infer the number of posterior draws in a samples Dataset.

    Parameters
    ----------
    samples : xarray.Dataset
        Stage-5 samples Dataset.

    Returns
    -------
    int
        Number of samples/draws.
    """
    preferred = ['fp', 'ecosw', 'esinw', 't_secondary', 't0', 'per']
    for name in preferred:
        if name in samples:
            return int(np.asarray(samples[name].values).size)
    for name in samples.data_vars:
        if name == 'sample':
            continue
        return int(np.asarray(samples[name].values).size)
    raise ValueError('Could not infer number of posterior samples.')


def _visit_channel(visit, channel=None):
    """
    Return the zero-based channel index for a one-based visit number.

    Parameters
    ----------
    visit : int
        One-based visit number.
    channel : int or None, optional
        Explicit zero-based channel override.

    Returns
    -------
    int
        Zero-based channel index.
    """
    if channel is not None:
        return int(channel)
    return int(visit) - 1


def _visit_candidates(name, visit, channel=None):
    """
    Return Eureka parameter candidates for a visit/channel.

    Parameters
    ----------
    name : str
        Base parameter name.
    visit : int
        One-based visit number.
    channel : int or None, optional
        Explicit zero-based channel override.

    Returns
    -------
    candidates : list of str
        Candidate names, respecting Eureka's implicit ``_ch0`` convention.
    """
    ch = _visit_channel(visit, channel=channel)
    if ch == 0:
        return [name]
    return [f'{name}_ch{ch}', name]


def _epf_record(epf_params, candidate):
    """
    Look up a candidate parameter name in parsed EPF records.

    Parameters
    ----------
    epf_params : dict
        Parsed EPF parameter dictionary.
    candidate : str
        Candidate parameter name.

    Returns
    -------
    dict or None
        EPF record if present, otherwise ``None``.
    """
    if not epf_params:
        return None
    return epf_params.get(_normalize_control_key(candidate))


def _broadcast(value, n_samples):
    """
    Broadcast a scalar value to posterior-sample length.

    Parameters
    ----------
    value : float
        Scalar value.
    n_samples : int
        Number of posterior samples.

    Returns
    -------
    numpy.ndarray
        Length-``n_samples`` float array.
    """
    return np.full(int(n_samples), float(value), dtype=float)


def _resolve_visit_param(
    samples,
    epf_params,
    name,
    visit,
    n_samples,
    channel=None,
    required=True,
    default=None,
):
    """
    Resolve a shared or visit-specific parameter to samples.

    Parameters
    ----------
    samples : xarray.Dataset
        Stage-5 posterior samples.
    epf_params : dict
        Parsed EPF parameters.
    name : str
        Base parameter name to resolve.
    visit : int
        One-based visit number.
    n_samples : int
        Number of posterior draws.
    channel : int or None, optional
        Explicit zero-based channel override.
    required : bool, optional
        If ``True``, raise an error when missing.
    default : float or None, optional
        Default scalar value to broadcast when missing.

    Returns
    -------
    values : numpy.ndarray or None
        Posterior samples or fixed value broadcast to posterior length.
    source : str
        Description of where the parameter was found.
    """
    candidates = _visit_candidates(name, visit, channel=channel)

    for candidate in candidates:
        if samples is not None and candidate in samples:
            arr = np.asarray(samples[candidate].values, dtype=float).ravel()
            if arr.size == 1:
                arr = _broadcast(arr.item(), n_samples)
            return arr, f'samples:{candidate}'

    for candidate in candidates:
        record = _epf_record(epf_params, candidate)
        if record is None:
            continue
        value = record.get('value')
        if value is None:
            continue
        try:
            return _broadcast(value, n_samples), f'epf:{candidate}'
        except (TypeError, ValueError):
            continue

    if default is not None:
        return _broadcast(default, n_samples), 'default'

    if required:
        tried = ', '.join(candidates)
        raise KeyError(f'Could not resolve parameter {name!r}; tried {tried}')

    return None, 'missing'


def _to_bjd_tdb(values):
    """
    Convert MJD-like times to BJD-like times when needed.

    Parameters
    ----------
    values : array-like
        Time samples.

    Returns
    -------
    numpy.ndarray
        Times in BJD_TDB convention used by the output products.
    """
    arr = np.asarray(values, dtype=float)
    med = np.nanmedian(np.abs(arr))
    if med < 1000000.0:
        return arr + timeOffset
    return arr


def _conjunction_derivative(f, ecc, omega, sin_i2):
    """
    Derivative of projected separation squared with true anomaly.

    Parameters
    ----------
    f : numpy.ndarray
        True anomaly in radians.
    ecc : numpy.ndarray
        Eccentricity.
    omega : numpy.ndarray
        Argument of periastron in radians.
    sin_i2 : numpy.ndarray
        ``sin(inclination)**2``.

    Returns
    -------
    numpy.ndarray
        Derivative of ``log(projected_separation**2)``.
    """
    u = omega + f
    denom = 1.0 + ecc * np.cos(f)
    q = 1.0 - sin_i2 * np.sin(u) ** 2
    denom = np.maximum(denom, 1.0e-14)
    q = np.maximum(q, 1.0e-14)
    return 2.0 * ecc * np.sin(f) / denom - sin_i2 * np.sin(2.0 * u) / q


def _find_conjunction_true_anomaly(ecc, omega, sin_i2, primary=True):
    """
    Find the true anomaly of the projected-separation minimum.

    Parameters
    ----------
    ecc : numpy.ndarray
        Eccentricity.
    omega : numpy.ndarray
        Argument of periastron in radians.
    sin_i2 : numpy.ndarray
        ``sin(inclination)**2``.
    primary : bool, optional
        If ``True``, solve near primary transit; otherwise near eclipse.

    Returns
    -------
    numpy.ndarray
        True anomaly in radians.
    """
    if primary:
        f = 0.5 * np.pi - omega
    else:
        f = 1.5 * np.pi - omega
    f = np.mod(f, 2.0 * np.pi)

    eps = 1.0e-5
    for _ in range(20):
        val = _conjunction_derivative(f, ecc, omega, sin_i2)
        hi = _conjunction_derivative(f + eps, ecc, omega, sin_i2)
        lo = _conjunction_derivative(f - eps, ecc, omega, sin_i2)
        der = (hi - lo) / (2.0 * eps)
        step = np.divide(
            val,
            der,
            out=np.zeros_like(val, dtype=float),
            where=np.abs(der) > 1.0e-14,
        )
        step = np.clip(step, -0.25, 0.25)
        f = np.mod(f - step, 2.0 * np.pi)
    return f


def _mean_anomaly_from_true(f, ecc):
    """
    Convert true anomaly to mean anomaly.

    Parameters
    ----------
    f : numpy.ndarray
        True anomaly in radians.
    ecc : numpy.ndarray
        Eccentricity.

    Returns
    -------
    numpy.ndarray
        Mean anomaly in radians on ``[0, 2*pi)``.
    """
    s = np.sqrt(np.maximum(1.0 - ecc, 0.0)) * np.sin(0.5 * f)
    c = np.sqrt(1.0 + ecc) * np.cos(0.5 * f)
    e_anom = 2.0 * np.arctan2(s, c)
    mean_anom = e_anom - ecc * np.sin(e_anom)
    return np.mod(mean_anom, 2.0 * np.pi)


def _light_travel_days(z_diff_rs, rs_rsun):
    """
    Convert a line-of-sight distance difference to days.

    Parameters
    ----------
    z_diff_rs : array-like
        Distance difference in stellar-radius units.
    rs_rsun : array-like
        Stellar radius in Solar-radius units.

    Returns
    -------
    numpy.ndarray
        Light-travel time in days.
    """
    from astropy import constants as const

    seconds = np.asarray(z_diff_rs) * np.asarray(rs_rsun)
    seconds = seconds * const.R_sun.value / const.c.value
    return seconds / 86400.0


def _secondary_delta_days(
    per,
    ecosw,
    esinw,
    b,
    a,
    rs,
    compute_ltt=True,
):
    """
    Compute eclipse timing offset from the preceding transit.

    Parameters
    ----------
    per : array-like
        Orbital period in days.
    ecosw : array-like
        ``ecc * cos(omega)``.
    esinw : array-like
        ``ecc * sin(omega)``.
    b : array-like
        Transit impact parameter.
    a : array-like
        Scaled semi-major axis, ``a/Rs``.
    rs : array-like
        Stellar radius in Solar radii.
    compute_ltt : bool, optional
        Whether to include the light-travel-time correction.

    Returns
    -------
    numpy.ndarray
        Secondary-eclipse time minus primary-transit time, in days.
    """
    per = np.asarray(per, dtype=float)
    ecosw = np.asarray(ecosw, dtype=float)
    esinw = np.asarray(esinw, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    rs = np.asarray(rs, dtype=float)

    ecc = np.sqrt(ecosw**2 + esinw**2)
    ecc = np.clip(ecc, 0.0, 1.0 - 1.0e-10)
    omega = np.arctan2(esinw, ecosw)

    # Infer inclination from the transit impact parameter relation.
    cos_i = b / a * (1.0 + esinw) / np.maximum(1.0 - ecc**2, 1.0e-14)
    cos_i = np.clip(cos_i, 0.0, 1.0)
    sin_i = np.sqrt(np.maximum(1.0 - cos_i**2, 0.0))
    sin_i2 = sin_i**2

    f_tr = _find_conjunction_true_anomaly(ecc, omega, sin_i2, primary=True)
    f_sec = _find_conjunction_true_anomaly(ecc, omega, sin_i2, primary=False)

    m_tr = _mean_anomaly_from_true(f_tr, ecc)
    m_sec = _mean_anomaly_from_true(f_sec, ecc)
    delta_m = np.mod(m_sec - m_tr, 2.0 * np.pi)
    delta_days = per * delta_m / (2.0 * np.pi)

    if compute_ltt:
        r_tr = a * (1.0 - ecc**2) / (1.0 + ecc * np.cos(f_tr))
        r_sec = a * (1.0 - ecc**2) / (1.0 + ecc * np.cos(f_sec))
        z_tr = -r_tr * sin_i * np.sin(omega + f_tr)
        z_sec = -r_sec * sin_i * np.sin(omega + f_sec)
        delta_days = delta_days + _light_travel_days(z_sec - z_tr, rs)

    return delta_days


def _stage2_mjd_mid(stage2_fits):
    """
    Read the Stage-2 exposure midpoint from a JWST data model.

    Parameters
    ----------
    stage2_fits : str
        Path to a Stage-2 FITS file.

    Returns
    -------
    float
        Exposure midpoint in BMJD_TDB/MJD_TDB days.
    """
    dm = JwstDataModel(stage2_fits)
    return float(dm.meta.exposure.mid_time_tdb)


def _infer_eclipse_epoch(mjd_mid, t0, per, delta_days):
    """
    Infer the integer eclipse epoch nearest an exposure midpoint.

    Parameters
    ----------
    mjd_mid : float
        Stage-2 exposure midpoint in MJD-like days.
    t0 : array-like
        Primary-transit time samples in MJD-like days.
    per : array-like
        Period samples in days.
    delta_days : array-like
        Eclipse-minus-transit timing offsets in days.

    Returns
    -------
    int
        Integer epoch such that ``t0 + epoch*per + delta`` is near
        ``mjd_mid``.
    """
    t0_ref = float(np.nanmedian(t0))
    per_ref = float(np.nanmedian(per))
    delta_ref = float(np.nanmedian(delta_days))
    return int(np.round((float(mjd_mid) - t0_ref - delta_ref) / per_ref))


def _resolve_secondary_time_samples(
    samples,
    epf_params,
    visit,
    n_samples,
    mjd_mid=None,
    channel=None,
    eclipse_epoch=None,
    compute_ltt=True,
):
    """
    Resolve or derive secondary-eclipse timing samples.

    Parameters
    ----------
    samples : xarray.Dataset
        Shared or single-visit Stage-5 posterior samples.
    epf_params : dict
        Parsed EPF parameters.
    visit : int
        One-based visit number.
    n_samples : int
        Number of posterior samples.
    mjd_mid : float or None, optional
        Exposure midpoint used to infer the eclipse epoch.
    channel : int or None, optional
        Explicit zero-based channel override.
    eclipse_epoch : int or None, optional
        Explicit epoch override. If ``None``, inferred from ``mjd_mid``.
    compute_ltt : bool, optional
        Whether to include light-travel time in derived timings.

    Returns
    -------
    t_secondary_bjd : numpy.ndarray
        Secondary-eclipse timing samples in BJD_TDB.
    """
    direct_names = ['t_secondary', 'tsec', 't_eclipse', 'teclipse']
    for name in direct_names:
        values, _ = _resolve_visit_param(
            samples,
            epf_params,
            name,
            visit,
            n_samples,
            channel=channel,
            required=False,
        )
        if values is not None:
            return _to_bjd_tdb(values)

    t0, _ = _resolve_visit_param(
        samples, epf_params, 't0', visit, n_samples, channel=channel
    )
    per, _ = _resolve_visit_param(
        samples, epf_params, 'per', visit, n_samples, channel=channel
    )
    ecosw, _ = _resolve_visit_param(
        samples,
        epf_params,
        'ecosw',
        visit,
        n_samples,
        channel=channel,
        required=False,
        default=0.0,
    )
    esinw, _ = _resolve_visit_param(
        samples,
        epf_params,
        'esinw',
        visit,
        n_samples,
        channel=channel,
        required=False,
        default=0.0,
    )
    b, _ = _resolve_visit_param(
        samples,
        epf_params,
        'b',
        visit,
        n_samples,
        channel=channel,
        required=False,
        default=0.0,
    )
    a, _ = _resolve_visit_param(
        samples,
        epf_params,
        'a',
        visit,
        n_samples,
        channel=channel,
        required=bool(compute_ltt),
        default=None if compute_ltt else 1.0,
    )
    rs, _ = _resolve_visit_param(
        samples,
        epf_params,
        'Rs',
        visit,
        n_samples,
        channel=channel,
        required=bool(compute_ltt),
        default=None if compute_ltt else 1.0,
    )

    delta_days = _secondary_delta_days(
        per,
        ecosw,
        esinw,
        b,
        a,
        rs,
        compute_ltt=compute_ltt,
    )

    epoch = eclipse_epoch
    if epoch is None:
        if mjd_mid is None:
            epoch = 0
        else:
            epoch = _infer_eclipse_epoch(mjd_mid, t0, per, delta_days)
    epoch = int(epoch)

    t_secondary_mjd = t0 + epoch * per + delta_days
    return _to_bjd_tdb(t_secondary_mjd)


def _array_stats(values, decimals):
    """
    Compute median, standard deviation, and percentile errors.

    Parameters
    ----------
    values : array-like
        Posterior samples.
    decimals : int
        Decimal places for rounding.

    Returns
    -------
    tuple
        ``median, std, upper_error, lower_error``.
    """
    arr = np.asarray(values, dtype=float)
    med = np.round(np.nanmedian(arr), decimals)
    std = np.round(np.nanstd(arr), decimals)
    pct = np.nanpercentile(
        arr,
        [lowerErrorPercentile, medianPercentile, upperErrorPercentile],
    )
    up = np.round(pct[2] - pct[1], decimals)
    lo = np.round(pct[1] - pct[0], decimals)
    return med, std, up, lo


def build_single_eclipse_from_arrays(
    stage2_fits,
    stage3_specdata,
    stage4cal,
    fp_samples,
    t_secondary_samples_bjd,
    visit,
    STAR,
    PLANET,
    SRC_DOI,
    HLSPVER,
    ltt_corr=True,
):
    """
    Build a single-visit catalog Dataset from resolved posterior arrays.

    Parameters
    ----------
    stage2_fits : str
        Path to the Stage-2 FITS file.
    stage3_specdata : str
        Path to Stage-3 spectral data product.
    stage4cal : str
        Path to Stage-4 flux calibration product.
    fp_samples : array-like
        Eclipse-depth samples as planet/star flux ratio.
    t_secondary_samples_bjd : array-like
        Secondary-eclipse time samples in BJD_TDB.
    visit : int
        One-based visit index for the output product.
    STAR : str
        Host star identifier.
    PLANET : str
        Planet identifier.
    SRC_DOI : str
        Source-data DOI for this visit.
    HLSPVER : str
        HLSP version.
    ltt_corr : bool, optional
        Whether reported eclipse times include a light-travel-time
        correction.

    Returns
    -------
    xarray.Dataset
        Single-row eclipse catalog dataset.
    """
    spec_ds = xr.load_dataset(stage3_specdata)
    fluxcal = xr.load_dataset(stage4cal)
    dm = JwstDataModel(stage2_fits)

    (
        r_from_fits,
        s_from_fits,
        crds_ver_hdr,
        crds_ctx_hdr,
        filt_hdr,
        subarray_hdr,
        date_obs,
    ) = read_stage2_r_s_meta(stage2_fits)

    instrume = dm.meta.instrument.name
    filter_name = filt_hdr or dm.meta.instrument.filter
    subarray = subarray_hdr or getattr(dm.meta.subarray, 'name', '')
    telescop = dm.meta.telescope
    observat = dm.meta.telescope

    proposid = dm.meta.observation.program_number
    ra_targ = dm.meta.target.ra
    dec_targ = dm.meta.target.dec

    file_ids = []
    if hasattr(spec_ds, 'segment_list'):
        for seg in spec_ds.segment_list:
            name = Path(seg).name
            file_ids.append(name.split('_calints')[0])
    file_id_str = ','.join(file_ids)
    visit_id_str = _derive_visit_id_string(file_ids)

    mjd_beg = float(dm.meta.exposure.start_time_tdb)
    mjd_mid = float(dm.meta.exposure.mid_time_tdb)
    mjd_end = float(dm.meta.exposure.end_time_tdb)
    xposure = float(dm.meta.exposure.effective_exposure_time)

    cal_ver = str(dm.meta.calibration_software_version)
    pipe_line = str(getattr(spec_ds, 'data_format', ''))
    pipe_ver = str(getattr(spec_ds, 'version', ''))
    crds_ver = str(
        crds_ver_hdr or getattr(dm.meta.ref_file.crds, 'sw_version', '')
    )
    crds_ctx = str(
        crds_ctx_hdr or getattr(dm.meta.ref_file.crds, 'context_used', '')
    )

    depth_vals = np.asarray(fp_samples, dtype=float) * 1.0e6
    d_med, d_std, d_up, d_lo = _array_stats(
        depth_vals,
        eclipseDepthDecimals,
    )
    t_med, t_std, t_up, t_lo = _array_stats(
        t_secondary_samples_bjd,
        eclipseTimeDecimals,
    )

    stellar_flux = float(fluxcal.ecl_flux.data[0][0])
    ferr = float(fluxcal.ecl_ferr.data[0][0])
    sys_err = (0.0048 * stellar_flux) ** 2 + (0.0045 * stellar_flux) ** 2
    stellar_flux_err = np.sqrt(ferr**2 + sys_err)
    f_med = np.round(stellar_flux, absoluteFluxDecimals)
    f_err = np.round(stellar_flux_err, absoluteFluxDecimals)

    v = int(visit)
    ds = xr.Dataset()

    ds['eclipseDepth'] = _da1(d_med, v, units=eclipseDepth_units)
    ds['eclipseDepthError'] = _da1(d_std, v, units=eclipseDepth_units)
    ds['eclipseDepthUpperError'] = _da1(d_up, v, units=eclipseDepth_units)
    ds['eclipseDepthLowerError'] = _da1(d_lo, v, units=eclipseDepth_units)

    ds['eclipseTime'] = _da1(t_med, v, units=time_units)
    ds['eclipseTimeError'] = _da1(t_std, v, units=time_units)
    ds['eclipseTimeUpperError'] = _da1(t_up, v, units=time_units)
    ds['eclipseTimeLowerError'] = _da1(t_lo, v, units=time_units)

    ds['absFlux'] = _da1(f_med, v, units=TUNIT)
    ds['absFluxError'] = _da1(f_err, v, units=TUNIT)
    ds['absFluxUpperError'] = _da1(f_err, v, units=TUNIT)
    ds['absFluxLowerError'] = _da1(f_err, v, units=TUNIT)

    ds = ds.assign_coords(visit=[v])

    ds['date_obs'] = _da1(date_obs, v)
    ds['filter'] = _da1(str(filter_name), v)
    ds['subarray'] = _da1(str(subarray), v)

    ds['SRC_DOI'] = _da1(str(SRC_DOI), v)
    ds['VISIT_ID'] = _da1(visit_id_str, v)
    ds['FILE_ID'] = _da1(file_id_str, v)
    ds['MJD_BEG'] = _da1(mjd_beg, v)
    ds['MJD_MID'] = _da1(mjd_mid, v)
    ds['MJD_END'] = _da1(mjd_end, v)
    ds['XPOSURE'] = _da1(xposure, v)
    ds['CAL_VER'] = _da1(cal_ver, v)
    ds['CRDS_VER'] = _da1(crds_ver, v)
    ds['CRDS_CTX'] = _da1(crds_ctx, v)
    ds['PIPELINE'] = _da1(pipe_line, v)
    ds['PIPE_VER'] = _da1(pipe_ver, v)

    for key, val in r_from_fits.items():
        ds[key] = _da1(str(val), v)
    for key, val in s_from_fits.items():
        ds[key] = _da1(str(val), v)

    ds.attrs.update(
        {
            'HLSPVER': HLSPVER,
            'HLSPID': HLSPID,
            'HLSPNAME': HLSPNAME,
            'HLSP_PI': HLSP_PI,
            'HLSPLEAD': HLSPLEAD,
            'DOI': DOI,
            'STAR': STAR,
            'PLANET': PLANET,
            'HLSPTARG': PLANET,
            'OBSERVAT': observat,
            'TELESCOP': telescop,
            'INSTRUME': instrume,
            'RADESYS': RADESYS,
            'TIMESYS': TIMESYS,
            'TUNIT': TUNIT,
            'LICENSE': LICENSE,
            'LICENURL': LICENURL,
            'RA_TARG': ra_targ,
            'DEC_TARG': dec_targ,
            'PROPOSID': proposid,
            'LTT_CORR': int(bool(ltt_corr)),
        }
    )

    return ds


def _catalog_from_shared_fit(
    stage2_fits,
    stage3_specdata,
    stage4cal,
    samples,
    epf_params,
    ecf_config,
    visit,
    STAR,
    PLANET,
    SRC_DOI,
    HLSPVER,
    channel=None,
    eclipse_epoch=None,
):
    """
    Build one catalog row from shared samples plus EPF/ECF context.

    Parameters are the same as :func:`build_single_eclipse_from_arrays`,
    with additional shared-fit context.

    Returns
    -------
    xarray.Dataset
        Single-row catalog dataset.
    """
    v = int(visit)
    ch = _visit_channel(v, channel=channel)
    n_samples = _infer_n_samples(samples)
    compute_ltt = get_compute_ltt(ecf_config, default=True)

    fp_samples, _ = _resolve_visit_param(
        samples,
        epf_params,
        'fp',
        v,
        n_samples,
        channel=ch,
    )

    mjd_mid = _stage2_mjd_mid(stage2_fits)
    t_secondary = _resolve_secondary_time_samples(
        samples,
        epf_params,
        v,
        n_samples,
        mjd_mid=mjd_mid,
        channel=ch,
        eclipse_epoch=eclipse_epoch,
        compute_ltt=compute_ltt,
    )
    return build_single_eclipse_from_arrays(
        stage2_fits,
        stage3_specdata,
        stage4cal,
        fp_samples,
        t_secondary,
        v,
        STAR,
        PLANET,
        SRC_DOI,
        HLSPVER,
        ltt_corr=compute_ltt,
    )


def _visit_value(visit_info, *names, default=None, required=True):
    """
    Read a value from a visit-info dictionary using name aliases.

    Parameters
    ----------
    visit_info : dict
        Per-visit configuration dictionary.
    *names : str
        Candidate key names.
    default : object, optional
        Default value when no key is present.
    required : bool, optional
        If ``True``, raise an error when missing.

    Returns
    -------
    object
        Requested value.
    """
    lower = {str(key).lower(): value for key, value in visit_info.items()}
    for name in names:
        if name in visit_info:
            return visit_info[name]
        low = str(name).lower()
        if low in lower:
            return lower[low]
    if required:
        joined = ', '.join(names)
        raise KeyError(f'Missing required visit field; tried {joined}')
    return default


def build_single_eclipse_products(
    stage2_fits,
    stage3_specdata,
    stage4_lcdata,
    stage4cal,
    stage5_samples,
    stage5_fit,
    visit,
    STAR,
    PLANET,
    SRC_DOI,
    HLSPVER,
    out_dir='.',
    stage5_epf=None,
    stage5_ecf=None,
    channel=None,
    eclipse_epoch=None,
    make_fits=True,
):
    """
    Build and save all single-eclipse HLSP products.

    Parameters
    ----------
    stage2_fits, stage3_specdata, stage4_lcdata, stage4cal : str
        Per-eclipse Stage-2/3/4 products.
    stage5_samples : str
        Stage-5 posterior samples file. May contain direct ``t_secondary``
        samples or shared eccentricity/timing parameters.
    stage5_fit : str
        Stage-5 fit table. Used for the intermediate light-curve product.
    visit : int
        One-based visit number.
    STAR, PLANET, SRC_DOI, HLSPVER : str
        HLSP metadata.
    out_dir : str, optional
        Output directory. Created if needed.
    stage5_epf, stage5_ecf : str or None, optional
        Eureka parameter and control files used to resolve fixed values and
        model configuration.
    channel : int or None, optional
        Explicit zero-based channel override. By default, ``visit - 1``.
    eclipse_epoch : int or None, optional
        Explicit integer eclipse epoch override.
    make_fits : bool, optional
        Whether to also write the FITS catalog product.

    Returns
    -------
    outputs : dict
        Paths and in-memory datasets for the generated products.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    samples = _load_samples_dataset(stage5_samples)
    epf_params = parse_epf(stage5_epf)
    ecf_config = parse_ecf(stage5_ecf)

    cat_ds = _catalog_from_shared_fit(
        stage2_fits,
        stage3_specdata,
        stage4cal,
        samples,
        epf_params,
        ecf_config,
        visit,
        STAR,
        PLANET,
        SRC_DOI,
        HLSPVER,
        channel=channel,
        eclipse_epoch=eclipse_epoch,
    )

    cat_h5 = save_single_eclipse_hdf5(cat_ds, out_dir)
    cat_fits = hdf5_single_to_fits(cat_h5) if make_fits else None

    lc_ds = build_lightcurve_dataset(
        stage2_fits,
        stage3_specdata,
        stage4_lcdata,
        stage5_fit,
        visit,
        STAR,
        PLANET,
        SRC_DOI,
        HLSPVER,
    )
    lc_h5 = save_lightcurve_hdf5(lc_ds, out_dir)

    return {
        'catalog_dataset': cat_ds,
        'catalog_h5': cat_h5,
        'catalog_fits': cat_fits,
        'lightcurve_dataset': lc_ds,
        'lightcurve_h5': lc_h5,
    }


def build_checkpoint_products(
    visits,
    stage5_samples,
    stage5_fit,
    stage5_epf,
    stage5_ecf,
    checkpoint,
    STAR,
    PLANET,
    HLSPVER,
    out_dir='.',
    make_fits=True,
    make_checkpoint_plots=False,
    obs_ids=None,
):
    """
    Build and save all checkpoint HLSP products from a shared fit.

    Parameters
    ----------
    visits : list of dict
        Per-eclipse inputs. Each dictionary must contain ``visit``,
        ``stage2_fits``, ``stage3_specdata``, ``stage4_lcdata``,
        ``stage4cal``, and ``SRC_DOI``/``src_doi``. Optional keys are
        ``channel`` and ``eclipse_epoch``.
    stage5_samples : str
        Shared Stage-5 posterior samples file.
    stage5_fit : str
        Shared Stage-5 fit table containing all eclipses.
    stage5_epf, stage5_ecf : str or None
        Eureka parameter/control files for fixed values and configuration.
    checkpoint : int
        Checkpoint number used in output filenames.
    STAR, PLANET, HLSPVER : str
        HLSP metadata.
    out_dir : str, optional
        Output directory. Created if needed.
    make_fits : bool, optional
        Whether to also write the FITS catalog product.
    make_checkpoint_plots : bool, optional
        Whether to also write checkpoint-level report figures from the
        shared Stage-5 fit table.
    obs_ids : sequence or None, optional
        Explicit observation IDs for checkpoint observation-summary row
        labels. A flat sequence is interpreted as one ID per detected
        observation chunk; a nested sequence is interpreted as one sequence
        of IDs per plotted row.

    Returns
    -------
    outputs : dict
        Paths and in-memory datasets for the generated products.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    samples = _load_samples_dataset(stage5_samples)
    epf_params = parse_epf(stage5_epf)
    ecf_config = parse_ecf(stage5_ecf)

    cat_datasets = []
    lc_datasets = []

    for info in visits:
        visit = int(_visit_value(info, 'visit'))
        stage2_fits = _visit_value(info, 'stage2_fits')
        stage3_specdata = _visit_value(info, 'stage3_specdata')
        stage4_lcdata = _visit_value(info, 'stage4_lcdata')
        stage4cal = _visit_value(info, 'stage4cal')
        src_doi = _visit_value(
            info,
            'SRC_DOI',
            'src_doi',
            default='',
            required=False,
        )
        channel = _visit_value(
            info,
            'channel',
            default=None,
            required=False,
        )
        eclipse_epoch = _visit_value(
            info,
            'eclipse_epoch',
            default=None,
            required=False,
        )

        cat_ds = _catalog_from_shared_fit(
            stage2_fits,
            stage3_specdata,
            stage4cal,
            samples,
            epf_params,
            ecf_config,
            visit,
            STAR,
            PLANET,
            src_doi,
            HLSPVER,
            channel=channel,
            eclipse_epoch=eclipse_epoch,
        )
        cat_datasets.append(cat_ds)

        lc_ds = build_lightcurve_dataset(
            stage2_fits,
            stage3_specdata,
            stage4_lcdata,
            stage5_fit,
            visit,
            STAR,
            PLANET,
            src_doi,
            HLSPVER,
        )
        lc_datasets.append(lc_ds)

    cat_combined, cat_h5 = save_multi_eclipse_hdf5(
        cat_datasets,
        checkpoint,
        out_dir=out_dir,
        hlspver=HLSPVER,
    )
    cat_fits = hdf5_checkpoint_to_fits(cat_h5) if make_fits else None

    lc_tree = build_lightcurve_datatree(lc_datasets)
    lc_h5 = save_lightcurve_multi_hdf5(
        lc_tree,
        checkpoint,
        out_dir=out_dir,
        hlspver=HLSPVER,
        filtername=str(cat_combined['filter'].values[0]),
    )
    checkpoint_figures = None
    if make_checkpoint_plots:
        checkpoint_figures = make_checkpoint_figures(
            stage5_fit,
            stage5_samples,
            stage5_epf=stage5_epf,
            stage5_ecf=stage5_ecf,
            date_obs=cat_combined['date_obs'].values
            if 'date_obs' in cat_combined else None,
            checkpoint=checkpoint,
            planet=PLANET,
            out_dir=out_dir,
            obs_ids=obs_ids,
        )

    return {
        'catalog_dataset': cat_combined,
        'catalog_datasets': cat_datasets,
        'catalog_h5': cat_h5,
        'catalog_fits': cat_fits,
        'lightcurve_datatree': lc_tree,
        'lightcurve_datasets': lc_tree,
        'lightcurve_visit_datasets': lc_datasets,
        'lightcurve_h5': lc_h5,
        'checkpoint_figures': checkpoint_figures,
    }


def _load_fit_table(stage5_fit):
    """
    Load a Eureka Stage-5 fit table.

    Parameters
    ----------
    stage5_fit : str or pandas.DataFrame
        Path to a whitespace-delimited Eureka table, or an already-loaded
        dataframe.

    Returns
    -------
    pandas.DataFrame
        Fit table.
    """
    if isinstance(stage5_fit, pd.DataFrame):
        return stage5_fit.copy()
    return pd.read_csv(stage5_fit, comment='#', delimiter=r'\s+')


def _first_fit_column(df, candidates):
    """
    Return the first available fit-table column from a candidate list.

    Parameters
    ----------
    df : pandas.DataFrame
        Fit table.
    candidates : list of str
        Candidate column names.

    Returns
    -------
    str
        Matching column name.
    """
    for col in candidates:
        if col in df.columns:
            return col
    joined = ', '.join(candidates)
    raise KeyError(f'Could not find any of these columns: {joined}')


def _match_time_scale(values, reference_time):
    """
    Put time values onto the same offset convention as a reference axis.

    Parameters
    ----------
    values : array-like
        Time values, either full BJD-like or BJD/MJD minus ``timeOffset``.
    reference_time : array-like
        Reference time axis.

    Returns
    -------
    numpy.ndarray
        Time values in the reference convention.
    """
    arr = np.asarray(values, dtype=float)
    ref = np.asarray(reference_time, dtype=float)
    if arr.size == 0 or ref.size == 0:
        return arr

    arr_med = np.nanmedian(np.abs(arr))
    ref_med = np.nanmedian(np.abs(ref))
    if arr_med > 1000000.0 and ref_med < 1000000.0:
        return arr - timeOffset
    if arr_med < 1000000.0 and ref_med > 1000000.0:
        return arr + timeOffset
    return arr


def _checkpoint_timing_samples(
    samples,
    epf_params,
    ecf_config,
    fit_time,
    visit=1,
    channel=None,
):
    """
    Resolve transit, period, and secondary-eclipse samples for plotting.

    The fixed orbital quantities used by older plotting notebooks are read
    from the EPF when absent from the posterior samples.

    Parameters
    ----------
    samples : xarray.Dataset
        Shared Stage-5 posterior samples.
    epf_params : dict
        Parsed Stage-5 EPF parameters.
    ecf_config : dict
        Parsed Stage-5 ECF values.
    fit_time : array-like
        Stage-5 fit-table time axis whose offset convention should be used.
    visit : int, optional
        Visit/channel to use for shared timing parameters.
    channel : int or None, optional
        Explicit zero-based channel override.

    Returns
    -------
    timing : dict
        Arrays for ``t0``, ``per``, and ``t_secondary`` in the fit-table time
        convention.
    """
    n_samples = _infer_n_samples(samples)
    compute_ltt = get_compute_ltt(ecf_config, default=True)
    t0, _ = _resolve_visit_param(
        samples,
        epf_params,
        't0',
        visit,
        n_samples,
        channel=channel,
    )
    per, _ = _resolve_visit_param(
        samples,
        epf_params,
        'per',
        visit,
        n_samples,
        channel=channel,
    )
    t_secondary = _resolve_secondary_time_samples(
        samples,
        epf_params,
        visit,
        n_samples,
        channel=channel,
        eclipse_epoch=0,
        compute_ltt=compute_ltt,
    )
    return {
        't0': _match_time_scale(t0, fit_time),
        'per': np.asarray(per, dtype=float),
        't_secondary': _match_time_scale(t_secondary, fit_time),
    }


def _fit_components(fit_table):
    """
    Extract normalized flux, model, and cleaned-flux arrays from a fit table.

    Parameters
    ----------
    fit_table : pandas.DataFrame
        Stage-5 fit table.

    Returns
    -------
    dict
        Named numpy arrays and the selected model column names.
    """
    time = fit_table['time'].to_numpy(dtype=float)
    flux = fit_table['lcdata'].to_numpy(dtype=float)
    flux_err = fit_table['lcerr'].to_numpy(dtype=float)

    astro_col = _first_fit_column(
        fit_table,
        [
            'astrophysical model',
            'astro model',
            'astro_model',
            'starry',
            'batman',
            'eclipse',
            'transit',
            'planetary model',
        ],
    )
    astro_model = fit_table[astro_col].to_numpy(dtype=float)

    try:
        model_col = _first_fit_column(
            fit_table,
            ['model', 'full model', 'full_model'],
        )
        model = fit_table[model_col].to_numpy(dtype=float)
    except KeyError:
        model_col = None
        model = astro_model.copy()

    combined_correction = np.divide(
        model,
        astro_model,
        out=np.ones_like(model, dtype=float),
        where=np.isfinite(astro_model) & (astro_model != 0.0),
    )

    clean_flux = np.divide(
        flux,
        combined_correction,
        out=np.full_like(flux, np.nan, dtype=float),
        where=np.isfinite(combined_correction) & (combined_correction != 0.0),
    )
    clean_err = np.divide(
        flux_err,
        combined_correction,
        out=np.full_like(flux_err, np.nan, dtype=float),
        where=np.isfinite(combined_correction)
        & (combined_correction != 0.0),
    )
    if 'residuals' in fit_table:
        residuals = fit_table['residuals'].to_numpy(dtype=float)
    else:
        residuals = clean_flux - astro_model

    return {
        'time': time,
        'flux': flux,
        'flux_err': flux_err,
        'clean_flux': clean_flux,
        'clean_err': clean_err,
        'model': model,
        'astro_model': astro_model,
        'residuals': residuals,
        'model_col': model_col,
        'astro_col': astro_col,
    }


def _weighted_bins_by_count(x, y, yerr, target_bins=28):
    """
    Bin sorted data into roughly ``target_bins`` inverse-variance bins.

    Parameters
    ----------
    x, y, yerr : array-like
        Sorted data and uncertainties.
    target_bins : int, optional
        Approximate number of output bins.

    Returns
    -------
    tuple of numpy.ndarray
        Binned ``x``, ``y``, and uncertainty arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    nbin = max(1, len(x) // int(target_bins))
    n = len(x) // nbin * nbin
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    xb = x[:n].reshape(-1, nbin)
    yb = y[:n].reshape(-1, nbin)
    eb = yerr[:n].reshape(-1, nbin)
    finite = np.isfinite(yb) & np.isfinite(eb) & (eb > 0.0)
    weights = np.where(finite, 1.0 / eb**2, 0.0)
    weight_sum = np.sum(weights, axis=1)

    x_out = np.nanmean(xb, axis=1)
    y_out = np.divide(
        np.sum(weights * yb, axis=1),
        weight_sum,
        out=np.full(weight_sum.shape, np.nan, dtype=float),
        where=weight_sum > 0.0,
    )
    err_out = np.divide(
        1.0,
        np.sqrt(weight_sum),
        out=np.full(weight_sum.shape, np.nan, dtype=float),
        where=weight_sum > 0.0,
    )
    return x_out, y_out, err_out


def _phase_bins_by_histogram(phase, y, residuals, yerr, target_bins=28):
    """
    Bin phase-folded data using the original report-notebook convention.

    Parameters
    ----------
    phase, y, residuals, yerr : array-like
        Phase-sorted arrays.
    target_bins : int, optional
        Number of phase bins spanning the observed phase range.

    Returns
    -------
    tuple
        ``phase_bin``, ``flux_bin``, ``residual_bin``, ``err_bin``, and
        approximate bin size in phase units.
    """
    phase = np.asarray(phase, dtype=float)
    y = np.asarray(y, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    finite = (
        np.isfinite(phase)
        & np.isfinite(y)
        & np.isfinite(residuals)
        & np.isfinite(yerr)
    )
    phase = phase[finite]
    y = y[finite]
    residuals = residuals[finite]
    yerr = yerr[finite]
    if phase.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, np.nan

    counts, phase_edges = np.histogram(phase, int(target_bins))
    bin_edges = np.append(0, np.cumsum(counts))
    bin_width = float(np.nanmedian(np.diff(phase_edges)))

    phase_bin = []
    flux_bin = []
    residual_bin = []
    err_bin = []

    for i in range(int(target_bins)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if hi <= lo:
            continue

        p_temp = phase[lo:hi]
        y_temp = y[lo:hi]
        r_temp = residuals[lo:hi]
        e_temp = yerr[lo:hi]
        good = np.isfinite(e_temp) & (e_temp > 0.0)
        if not np.any(good):
            continue

        p_temp = p_temp[good]
        y_temp = y_temp[good]
        r_temp = r_temp[good]
        e_temp = e_temp[good]

        phase_bin.append(np.nanmedian(p_temp))
        flux_bin.append(np.sum(y_temp * e_temp) / np.sum(e_temp))
        residual_bin.append(np.sum(r_temp * e_temp) / np.sum(e_temp))
        err_bin.append(
            np.sum(e_temp * e_temp) / np.sum(e_temp) / np.sqrt(len(e_temp))
        )

    return (
        np.asarray(phase_bin),
        np.asarray(flux_bin),
        np.asarray(residual_bin),
        np.asarray(err_bin),
        bin_width,
    )


def _split_indices_by_gaps(time, min_gap_days):
    """
    Split a time series into sorted index groups at large gaps.

    Parameters
    ----------
    time : array-like
        Time axis.
    min_gap_days : float
        Minimum gap size that starts a new group.

    Returns
    -------
    list of numpy.ndarray
        Sorted integer-index groups.
    """
    time = np.asarray(time, dtype=float)
    if time.size == 0:
        return []
    order = np.argsort(time)
    gaps = np.diff(time[order])
    cuts = np.where(gaps > float(min_gap_days))[0] + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(order)]
    return [order[start:stop] for start, stop in zip(starts, stops)]


def _nearest_mid_eclipse_times(segments, time, t_secondary, period):
    """
    Find the predicted mid-eclipse closest to each observed segment.

    Parameters
    ----------
    segments : list of numpy.ndarray
        Observation index groups.
    time : array-like
        Fit-table time axis.
    t_secondary : array-like
        Secondary-eclipse timing samples.
    period : array-like
        Orbital-period samples.

    Returns
    -------
    numpy.ndarray
        Mid-eclipse times, one per segment.
    """
    tsec_ref = float(np.nanmedian(np.asarray(t_secondary, dtype=float)))
    period_ref = float(np.nanmedian(np.asarray(period, dtype=float)))
    mid_times = []
    for seg in segments:
        visit_center = float(np.nanmedian(time[np.asarray(seg)]))
        epoch = np.rint((visit_center - tsec_ref) / period_ref)
        mid_times.append(tsec_ref + epoch * period_ref)
    return np.asarray(mid_times, dtype=float)


def _observed_offsets_xlim(segments, time, mid_times, padding_fraction=0.02):
    """
    Return a common observed x-limit in hours from mid-eclipse.

    Parameters
    ----------
    segments : list of numpy.ndarray
        Observation index groups.
    time : array-like
        Fit-table time axis.
    mid_times : array-like
        Mid-eclipse time for each segment.
    padding_fraction : float, optional
        Fractional padding around the full observed envelope.

    Returns
    -------
    tuple
        ``(xmin, xmax)`` in hours.
    """
    offsets = []
    for seg, tmid in zip(segments, mid_times):
        offsets.append((time[np.asarray(seg)] - tmid) * 24.0)
    offsets = np.concatenate(offsets)

    xmin = float(np.nanmin(offsets))
    xmax = float(np.nanmax(offsets))
    width = xmax - xmin

    finite_time = np.sort(time[np.isfinite(time)])
    cadence = np.diff(finite_time)
    cadence = cadence[cadence > 0.0]
    cadence_pad = float(np.nanmedian(cadence)) * 24.0 if cadence.size else 0.0
    pad = max(cadence_pad, float(padding_fraction) * width)
    return xmin - pad, xmax + pad


def _padded_ylim(values, padding_fraction=0.05, fallback=(0.0, 1.0)):
    """
    Return finite data limits with symmetric fractional padding.

    Parameters
    ----------
    values : array-like or list of array-like
        Values that should be visible on the y-axis.
    padding_fraction : float, optional
        Fraction of the data span to add below and above the limits.
    fallback : tuple, optional
        Limits to use when there are no finite values.

    Returns
    -------
    tuple
        ``(ymin, ymax)`` padded around the finite values.
    """
    if isinstance(values, (list, tuple)):
        arrays = [
            np.asarray(value, dtype=float).ravel()
            for value in values
            if np.asarray(value).size > 0
        ]
        if not arrays:
            return fallback
        arr = np.concatenate(arrays)
    else:
        arr = np.asarray(values, dtype=float).ravel()

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return fallback

    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    span = ymax - ymin
    if span == 0.0:
        scale = max(abs(ymin), 1.0)
        span = max(scale * 1.0e-3, 1.0e-6)

    pad = float(padding_fraction) * span
    return ymin - pad, ymax + pad


def _zero_centered_sigma_ylim(values, nsigma=3.0):
    """
    Return zero-centered limits from a robust sigma estimate.

    Parameters
    ----------
    values : array-like or list of array-like
        Values used to estimate the plotted scatter.
    nsigma : float, optional
        Half-width in robust standard deviations.

    Returns
    -------
    tuple or None
        ``(-half_width, half_width)`` with half-width rounded up to a
        readable value, or ``None`` if finite limits cannot be inferred.
    """
    if isinstance(values, (list, tuple)):
        arrays = [
            np.asarray(value, dtype=float).ravel()
            for value in values
            if np.asarray(value).size > 0
        ]
        if not arrays:
            return None
        arr = np.concatenate(arrays)
    else:
        arr = np.asarray(values, dtype=float).ravel()

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    center = np.nanmedian(finite)
    mad = np.nanmedian(np.abs(finite - center))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = np.nanstd(finite)
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = np.nanmax(np.abs(finite))
    if not np.isfinite(sigma) or sigma == 0.0:
        return None

    half_width = float(nsigma) * float(sigma)
    scale = 10.0 ** np.floor(np.log10(half_width))
    fraction = half_width / scale
    for nice_fraction in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]:
        if fraction <= nice_fraction:
            half_width = nice_fraction * scale
            break
    else:
        half_width = 10.0 * scale

    return -half_width, half_width


def _observation_label(first_observation, n_observations):
    """
    Return a compact observation label from sequential observation numbers.

    Parameters
    ----------
    first_observation : int
        One-based number of the first observation represented by a row.
    n_observations : int
        Number of observation chunks represented by the row.

    Returns
    -------
    str
        Label such as ``'Observation 1'``, ``'Observations 4 & 5'``, or
        ``'Observations 4-6'``.
    """
    first = int(first_observation)
    count = int(n_observations)
    if count <= 1:
        return f'Observation {first}'
    last = first + count - 1
    if count == 2:
        return f'Observations {first} & {last}'
    return f'Observations {first}-{last}'


def _observation_label_from_ids(observation_ids):
    """
    Return a compact observation label from explicit observation IDs.

    Parameters
    ----------
    observation_ids : sequence
        Observation identifiers represented by one plotted row.

    Returns
    -------
    str
        Label such as ``'Observation 12'`` or ``'Observations 12 & 14'``.
    """
    labels = [str(obs_id) for obs_id in observation_ids]
    if len(labels) == 0:
        return 'Observation'
    if len(labels) == 1:
        return f'Observation {labels[0]}'
    if len(labels) == 2:
        return f'Observations {labels[0]} & {labels[1]}'
    return f"Observations {', '.join(labels[:-1])} & {labels[-1]}"


def _apply_observation_id_override(segment_rows, obs_ids):
    """
    Apply explicit observation IDs to checkpoint observation-row labels.

    Parameters
    ----------
    segment_rows : list of dict
        Row metadata dictionaries, each with ``n_observations`` and
        ``label`` keys.
    obs_ids : sequence or None
        If flat, one ID per detected observation chunk. If nested, one
        sequence of IDs per plotted row.

    Returns
    -------
    None
        ``segment_rows`` is modified in place.
    """
    if obs_ids is None:
        return

    rows = list(obs_ids)
    if len(rows) == len(segment_rows) and any(
        isinstance(row, (list, tuple, np.ndarray)) for row in rows
    ):
        for segment_row, row_ids in zip(segment_rows, rows):
            ids = list(row_ids)
            expected = segment_row['n_observations']
            if len(ids) != expected:
                raise ValueError(
                    'Nested obs_ids entries must match the detected '
                    f'observation chunks per row; expected {expected}, got '
                    f'{len(ids)}.'
                )
            segment_row['label'] = _observation_label_from_ids(ids)
        return

    total_observations = sum(row['n_observations'] for row in segment_rows)
    if len(rows) != total_observations:
        raise ValueError(
            'obs_ids must provide one ID per detected observation chunk '
            f'({total_observations} total), or nested IDs for each plotted '
            f'row ({len(segment_rows)} rows).'
        )

    start = 0
    for segment_row in segment_rows:
        stop = start + segment_row['n_observations']
        segment_row['label'] = _observation_label_from_ids(rows[start:stop])
        start = stop


def _date_obs_label(date_obs, index):
    """
    Return a compact label from a ``date_obs`` array.

    Parameters
    ----------
    date_obs : array-like
        Per-observation datetime values, typically from a catalog or
        light-curve dataset's ``date_obs`` variable.
    index : int
        Observation index to label.

    Returns
    -------
    str
        Formatted date label.
    """
    if date_obs is None:
        raise ValueError('date_obs is required for checkpoint row labels.')

    values = np.asarray(date_obs)
    if values.ndim == 0:
        if index != 0:
            raise IndexError('date_obs has one value but multiple rows need '
                             'labels.')
        value = values.item()
    elif index >= values.size:
        raise IndexError('date_obs has fewer values than observation rows.')
    else:
        value = values.ravel()[index]

    try:
        timestamp = pd.to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Could not parse date_obs value {value!r}.') from exc
    if pd.isna(timestamp):
        raise ValueError('date_obs contains NaT for an observation row.')
    return timestamp.strftime('%b %d, %Y')


def _checkpoint_style_axis(
    ax,
    x,
    ylim,
    xlim=None,
    max_major_ticks=6,
    y_major=None,
    y_minor=None,
    yfmt='%.3f',
    xfmt='%.1f',
    ticklen=4,
):
    """
    Apply report-style ticks and mirrored axes to a checkpoint subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style.
    x : array-like
        X data used to infer limits when ``xlim`` is not supplied.
    ylim : tuple
        Y-axis limits.
    xlim : tuple or None, optional
        X-axis limits.
    max_major_ticks : int, optional
        Approximate number of major ticks.
    y_major, y_minor : float or None, optional
        Explicit y-axis major/minor tick spacing.
    yfmt, xfmt : str, optional
        Tick-label format strings.
    ticklen : float, optional
        Minor tick length in points.
    """
    from matplotlib.ticker import (
        AutoMinorLocator,
        FormatStrFormatter,
        MaxNLocator,
        MultipleLocator,
    )

    x = np.asarray(x, dtype=float)
    if xlim is None:
        dx = np.nanmedian(np.diff(np.sort(x)))
        ax.set_xlim(np.nanmin(x) - dx, np.nanmax(x) + dx)
    else:
        ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    locator = MaxNLocator(
        nbins=max_major_ticks,
        steps=[1, 2, 2.5, 5, 10],
        min_n_ticks=3,
    )
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))

    if y_major is None:
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
        )
    else:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
    if y_minor is None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    else:
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))

    ax.tick_params(
        axis='both',
        which='major',
        direction='inout',
        length=2 * ticklen,
    )
    ax.tick_params(
        axis='both',
        which='minor',
        direction='inout',
        length=ticklen,
    )

    ax_top = ax.twiny()
    ax_right = ax.twinx()
    ax_top.set_xlim(ax.get_xlim())
    ax_right.set_ylim(ax.get_ylim())
    ax_top.xaxis.set_major_locator(
        MaxNLocator(
            nbins=max_major_ticks,
            steps=[1, 2, 2.5, 5, 10],
            min_n_ticks=3,
        )
    )
    ax_top.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_top.set_xticklabels([])

    if y_major is None:
        ax_right.yaxis.set_major_locator(
            MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
        )
    else:
        ax_right.yaxis.set_major_locator(MultipleLocator(y_major))
    if y_minor is None:
        ax_right.yaxis.set_minor_locator(AutoMinorLocator(5))
    else:
        ax_right.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax_right.set_yticklabels([])

    ax_top.tick_params(axis='x', which='both', direction='in')
    ax_right.tick_params(axis='y', which='both', direction='in')


def _checkpoint_plot_rc():
    """
    Return matplotlib rcParams for checkpoint report figures.

    Returns
    -------
    dict
        rcParams suitable for use with ``matplotlib.pyplot.rc_context``.
    """
    family = 'sans-serif'
    fontfamily = 'DejaVu Sans'
    fontsize = 14
    ticklen = 5
    return {
        'font.family': [family],
        f'font.{family}': fontfamily,
        'font.cursive': [fontfamily],
        'font.size': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.labelsize': fontsize,
        'legend.fontsize': fontsize,
        'xtick.major.size': ticklen,
        'ytick.major.size': ticklen,
        'xtick.minor.size': ticklen / 2,
        'ytick.minor.size': ticklen / 2,
        'figure.constrained_layout.use': True,
    }


def make_checkpoint_phase_folded_figure(
    stage5_fit,
    stage5_samples,
    stage5_epf=None,
    stage5_ecf=None,
    out_path=None,
    pdf_path=None,
    visit=1,
    channel=None,
    target_bins=28,
    ylim=None,
    zoom_ylim=None,
    zoom_ylim_residuals=None,
    unbinned_alpha=0.1,
):
    """
    Make the four-panel phase-folded checkpoint summary figure.

    Parameters
    ----------
    stage5_fit : str or pandas.DataFrame
        Shared Stage-5 fit table.
    stage5_samples : str or xarray.Dataset
        Shared Stage-5 posterior samples.
    stage5_epf, stage5_ecf : str or None, optional
        Eureka parameter/control files used to resolve fixed orbital
        quantities and ``compute_ltt``.
    out_path, pdf_path : str or None, optional
        Optional output paths for PNG/PDF (or any matplotlib-supported
        formats).
    visit : int, optional
        Visit/channel used to resolve shared timing parameters.
    channel : int or None, optional
        Explicit zero-based channel override.
    target_bins : int, optional
        Approximate number of phase bins.
    ylim : tuple or None, optional
        Y limits in ppm for the full-flux and full-residual panels. If
        ``None``, use zero-centered limits with a rounded 3-sigma half-width
        estimated from the unbinned points.
    zoom_ylim : tuple or None, optional
        Y limits in ppm for the binned-flux panel. If ``None``, use
        Matplotlib's automatic limits.
    zoom_ylim_residuals : tuple or None, optional
        Y limits in ppm for the binned-residual panel. If ``None``, use the
        binned-flux panel's max-min span, centered on zero.
    unbinned_alpha : float, optional
        Alpha for unbinned points.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    import matplotlib.pyplot as plt

    fit = _load_fit_table(stage5_fit)
    samples = _load_samples_dataset(stage5_samples)
    epf_params = parse_epf(stage5_epf)
    ecf_config = parse_ecf(stage5_ecf)
    comp = _fit_components(fit)
    timing = _checkpoint_timing_samples(
        samples,
        epf_params,
        ecf_config,
        comp['time'],
        visit=visit,
        channel=channel,
    )

    t0 = float(np.nanmedian(timing['t0']))
    period = float(np.nanmedian(timing['per']))
    phase = ((comp['time'] - t0) / period) % 1.0
    phase_eclipse = np.nanmedian(
        (timing['t_secondary'] - timing['t0']) / timing['per']
    )
    phase_eclipse = float(phase_eclipse % 1.0)

    order = np.argsort(phase)
    order = order[np.isfinite(phase[order])]
    phase = phase[order]
    clean = comp['clean_flux'][order]
    clean_err = comp['clean_err'][order]
    residuals = comp['residuals'][order]
    model = comp['astro_model'][order]
    clean_ppm = (clean - 1.0) * 1.0e6
    residuals_ppm = residuals * 1.0e6

    if ylim is None:
        ylim = _zero_centered_sigma_ylim([clean_ppm, residuals_ppm])

    pb, yb, residualsb, yerrb, phase_bin_width = _phase_bins_by_histogram(
        phase,
        clean,
        residuals,
        clean_err,
        target_bins=target_bins,
    )
    bin_size_minutes = int(np.round(phase_bin_width * period * 24.0 * 60.0))

    with plt.rc_context(_checkpoint_plot_rc()):
        fig, axs = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(12 * 0.8, 12 * 0.8),
            sharex=True,
            gridspec_kw={
                'hspace': 0,
                'height_ratios': [1.0, 0.85, 1.0, 0.85],
            },
        )

        axs[0].errorbar(
            phase,
            clean_ppm,
            yerr=clean_err * 1.0e6,
            fmt='.',
            color='grey',
            alpha=unbinned_alpha,
            label='Unbinned Data',
        )
        axs[0].errorbar(
            pb,
            (yb - 1.0) * 1.0e6,
            yerr=yerrb * 1.0e6,
            fmt='.',
            color='k',
            alpha=1,
            label=f'Binned Data ({bin_size_minutes} mins)',
        )
        axs[0].plot(
            phase,
            (model - 1.0) * 1.0e6,
            '-',
            c='r',
            zorder=np.inf,
            label='Eclipse Model',
        )
        axs[0].axvline(
            phase_eclipse,
            ls='dotted',
            c='red',
            label='Mid-Eclipse Timing',
        )
        if ylim is not None:
            axs[0].set_ylim(*ylim)

        axs[1].errorbar(
            pb,
            (yb - 1.0) * 1.0e6,
            yerr=yerrb * 1.0e6,
            fmt='.',
            color='k',
            alpha=1,
        )
        axs[1].plot(phase, (model - 1.0) * 1.0e6, '-', c='r',
                    zorder=np.inf)
        axs[1].axvline(phase_eclipse, ls='dotted', c='red')
        if zoom_ylim is not None:
            axs[1].set_ylim(*zoom_ylim)
        binned_flux_ylim = axs[1].get_ylim()

        axs[2].errorbar(
            phase,
            residuals_ppm,
            yerr=clean_err * 1.0e6,
            fmt='.',
            color='grey',
            alpha=unbinned_alpha,
        )
        axs[2].errorbar(
            pb,
            residualsb * 1.0e6,
            yerr=yerrb * 1.0e6,
            fmt='.',
            color='k',
            alpha=1,
        )
        axs[2].plot(phase, np.zeros_like(phase), '-', c='r', zorder=np.inf)
        axs[2].axvline(phase_eclipse, ls='dotted', c='red')
        if ylim is not None:
            axs[2].set_ylim(*ylim)

        axs[3].errorbar(
            pb,
            residualsb * 1.0e6,
            yerr=yerrb * 1.0e6,
            fmt='.',
            color='k',
            alpha=1,
        )
        axs[3].plot(phase, np.zeros_like(phase), '-', c='r', zorder=np.inf)
        axs[3].axvline(phase_eclipse, ls='dotted', c='red')
        if zoom_ylim_residuals is None:
            zoom_span = binned_flux_ylim[1] - binned_flux_ylim[0]
            zoom_ylim_residuals = (-0.5 * zoom_span, 0.5 * zoom_span)
        axs[3].set_ylim(*zoom_ylim_residuals)

        if phase.size > 2:
            phase_pad = phase[2] - phase[0]
        else:
            phase_pad = 0.01
        axs[0].set_xlim(phase[0] - phase_pad, phase[-1] + phase_pad)

        axs[0].set_ylabel('Planetary Flux\n(ppm)')
        axs[1].set_ylabel('Binned Flux\n(ppm)')
        axs[2].set_ylabel('Residuals\n(ppm)')
        axs[3].set_ylabel('Binned Residuals\n(ppm)')
        axs[3].set_xlabel('Orbital Phase')
        fig.align_ylabels(axs)

        handles, labels = axs[0].get_legend_handles_labels()
        handles = handles[2:] + handles[:2]
        labels = labels[2:] + labels[:2]
        axs[0].legend(handles, labels, loc=8, bbox_to_anchor=(0.5, 1),
                      ncol=2)

        for ax in axs:
            ax.minorticks_on()

        for ax in axs[:-1]:
            ax.tick_params(
                axis='x',
                which='both',
                direction='in',
                bottom=True,
                top=True,
            )
            ax.tick_params(axis='y', which='major', direction='inout',
                           left=True)
            ax.tick_params(axis='y', which='minor', direction='inout',
                           left=True)
        axs[-1].tick_params(axis='x', which='major', direction='inout',
                            bottom=True)
        axs[-1].tick_params(axis='x', which='minor', direction='inout',
                            bottom=True)
        axs[-1].tick_params(axis='y', which='major', direction='inout',
                            left=True)
        axs[-1].tick_params(axis='y', which='minor', direction='inout',
                            left=True)

        for ax in axs:
            ax_right = ax.twinx()
            ax_right.set_ylim(ax.get_ylim())
            ax_right.set_yticklabels([])
            ax_right.minorticks_on()
            ax_right.tick_params(axis='y', which='both', direction='in')
        ax_top = axs[-1].twiny()
        ax_top.set_xlim(axs[-1].get_xlim())
        ax_top.set_xticklabels([])
        ax_top.minorticks_on()
        ax_top.tick_params(axis='x', which='both', direction='in')

        if out_path is not None:
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
        if pdf_path is not None:
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        return fig


def make_checkpoint_observation_summary_figure(
    stage5_fit,
    stage5_samples,
    stage5_epf=None,
    stage5_ecf=None,
    out_path=None,
    pdf_path=None,
    visit=1,
    channel=None,
    target_bins=28,
    internal_gap_days=0.003,
    visit_gap_days=0.5,
    raw_ylim=None,
    clean_ylim=None,
    date_obs=None,
    obs_ids=None,
):
    """
    Make the per-observation checkpoint report summary figure.

    Parameters
    ----------
    stage5_fit : str or pandas.DataFrame
        Shared Stage-5 fit table.
    stage5_samples : str or xarray.Dataset
        Shared Stage-5 posterior samples.
    stage5_epf, stage5_ecf : str or None, optional
        Eureka parameter/control files used to resolve fixed orbital
        quantities and ``compute_ltt``.
    out_path, pdf_path : str or None, optional
        Optional output paths for PNG/PDF.
    visit : int, optional
        Visit/channel used to resolve shared timing parameters.
    channel : int or None, optional
        Explicit zero-based channel override.
    target_bins : int, optional
        Approximate number of binned points per continuous chunk.
    internal_gap_days : float, optional
        Gap threshold used to avoid drawing model lines across short breaks.
    visit_gap_days : float, optional
        Gap threshold used to split distinct observations.
    raw_ylim : tuple or None, optional
        Y-axis limits for raw flux panels. If ``None``, limits are computed
        from all plotted raw-flux points in the left column.
    clean_ylim : tuple or None, optional
        Y-axis limits for calibrated flux panels. If ``None``, limits are
        computed from the binned calibrated-flux points and their
        uncertainties only.
    date_obs : array-like
        Per-observation datetimes used for row labels.
    obs_ids : sequence or None, optional
        Explicit observation IDs for row labels. A flat sequence is
        interpreted as one ID per detected observation chunk. A nested
        sequence is interpreted as one sequence of IDs per plotted row.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    import matplotlib.pyplot as plt

    fit = _load_fit_table(stage5_fit)
    samples = _load_samples_dataset(stage5_samples)
    epf_params = parse_epf(stage5_epf)
    ecf_config = parse_ecf(stage5_ecf)
    comp = _fit_components(fit)
    timing = _checkpoint_timing_samples(
        samples,
        epf_params,
        ecf_config,
        comp['time'],
        visit=visit,
        channel=channel,
    )

    time = comp['time']
    segments = _split_indices_by_gaps(time, min_gap_days=visit_gap_days)
    mid_times = _nearest_mid_eclipse_times(
        segments,
        time,
        timing['t_secondary'],
        timing['per'],
    )
    common_xlim = _observed_offsets_xlim(segments, time, mid_times)

    segment_rows = []
    next_observation = 1
    for i, seg in enumerate(segments):
        seg = np.asarray(seg)
        seg = seg[np.argsort(time[seg])]
        chunks = _split_indices_by_gaps(
            time[seg],
            min_gap_days=internal_gap_days,
        )
        n_observations = max(1, len(chunks))
        colors = [
            f'C{(next_observation + j - 1) % 10}'
            for j in range(n_observations)
        ]
        segment_rows.append(
            {
                'seg': seg,
                'chunks': chunks,
                'label': _observation_label(
                    next_observation,
                    n_observations,
                ),
                'n_observations': n_observations,
                'colors': colors,
                'mid_time': mid_times[i],
            }
        )
        next_observation += n_observations
    _apply_observation_id_override(segment_rows, obs_ids)

    if raw_ylim is None:
        raw_ylim = _padded_ylim(
            [comp['flux'][row['seg']] for row in segment_rows],
            padding_fraction=0.04,
        )

    if clean_ylim is None:
        binned_clean = []
        for row in segment_rows:
            seg = row['seg']
            tmid = row['mid_time']
            for chunk in row['chunks']:
                ii = seg[chunk]
                xx = (time[ii] - tmid) * 24.0
                _, yb, eb = _weighted_bins_by_count(
                    xx,
                    comp['clean_flux'][ii],
                    comp['clean_err'][ii],
                    target_bins=target_bins,
                )
                binned_clean.append(yb)
                binned_clean.append(yb - eb)
                binned_clean.append(yb + eb)
        clean_ylim = _padded_ylim(binned_clean, padding_fraction=0.08)

    with plt.rc_context(_checkpoint_plot_rc()):
        nrows = len(segments)
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(11.2, 2.25 * nrows),
            gridspec_kw={'hspace': 0.0, 'wspace': 0.06},
            squeeze=False,
        )

        for i, row in enumerate(segment_rows):
            seg = row['seg']
            tmid = row['mid_time']
            x = (time[seg] - tmid) * 24.0

            label = row['label'] + '\n' + _date_obs_label(date_obs, i)

            for chunk, color in zip(row['chunks'], row['colors']):
                ii = seg[chunk]
                xx = (time[ii] - tmid) * 24.0
                axs[i, 0].plot(
                    xx,
                    comp['flux'][ii],
                    '.',
                    color=color,
                    alpha=0.8,
                    ms=2,
                    rasterized=True,
                )
                axs[i, 0].plot(
                    xx,
                    comp['model'][ii],
                    '-',
                    c='k',
                    lw=1.4,
                    zorder=10,
                )

                axs[i, 1].plot(
                    xx,
                    comp['clean_flux'][ii],
                    '.',
                    color=color,
                    alpha=0.06,
                    ms=2,
                    rasterized=True,
                )
                xb, yb, eb = _weighted_bins_by_count(
                    xx,
                    comp['clean_flux'][ii],
                    comp['clean_err'][ii],
                    target_bins=target_bins,
                )
                axs[i, 1].errorbar(
                    xb,
                    yb,
                    yerr=eb,
                    fmt='.',
                    color=color,
                    ms=8,
                    zorder=5,
                )
                axs[i, 1].plot(
                    xx,
                    comp['astro_model'][ii],
                    '-',
                    c='k',
                    lw=1.4,
                    zorder=10,
                )

            for ax in axs[i]:
                ax.axvline(0.0, ls='dotted', c='0.25', lw=1.2, zorder=20)

            axs[i, 0].set_ylabel(label)
            _checkpoint_style_axis(
                axs[i, 0],
                x,
                ylim=raw_ylim,
                xlim=common_xlim,
                y_major=0.002,
                y_minor=0.0005,
                yfmt='%.3f',
            )
            _checkpoint_style_axis(
                axs[i, 1],
                x,
                ylim=clean_ylim,
                xlim=common_xlim,
                y_major=0.0002,
                y_minor=0.00005,
                yfmt='%.4f',
            )

        axs[0, 0].set_title('Normalized Raw Flux')
        axs[0, 1].set_title('Normalized Calibrated Flux')
        axs[-1, 0].set_xlabel('Time from mid-eclipse (hours)')
        axs[-1, 1].set_xlabel('Time from mid-eclipse (hours)')

        if out_path is not None:
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
        if pdf_path is not None:
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        return fig


def make_checkpoint_figures(
    stage5_fit,
    stage5_samples,
    stage5_epf=None,
    stage5_ecf=None,
    checkpoint=None,
    planet=None,
    out_dir='.',
    prefix=None,
    date_obs=None,
    obs_ids=None,
):
    """
    Make and save the standard checkpoint report figures.

    Parameters
    ----------
    stage5_fit : str or pandas.DataFrame
        Shared Stage-5 fit table.
    stage5_samples : str or xarray.Dataset
        Shared Stage-5 posterior samples.
    stage5_epf, stage5_ecf : str or None, optional
        Eureka parameter/control files used to resolve fixed orbital
        quantities and ``compute_ltt``.
    checkpoint : int or None, optional
        Checkpoint number used in default filenames.
    planet : str or None, optional
        Planet name used in default filenames.
    out_dir : str, optional
        Output directory.
    prefix : str or None, optional
        Filename prefix. If ``None``, one is built from ``checkpoint`` and
        ``planet``.
    date_obs : array-like
        Per-observation datetimes used for observation-summary row labels.
    obs_ids : sequence or None, optional
        Explicit observation IDs for observation-summary row labels. A flat
        sequence is interpreted as one ID per detected observation chunk; a
        nested sequence is interpreted as one sequence of IDs per plotted row.

    Returns
    -------
    dict
        Paths and figure objects for the generated report figures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        pieces = []
        if checkpoint is not None:
            pieces.append(f'checkpoint{int(checkpoint):02d}')
        if planet:
            pieces.append(''.join(str(planet).lower().split()))
        prefix = '_'.join(pieces) if pieces else 'checkpoint'

    phase_png = str(out_dir / f'{prefix}_phase_folded_4panel.png')
    phase_pdf = str(out_dir / f'{prefix}_phase_folded_4panel.pdf')
    obs_png = str(out_dir / f'{prefix}_observations_summary.png')
    obs_pdf = str(out_dir / f'{prefix}_observations_summary.pdf')

    phase_fig = make_checkpoint_phase_folded_figure(
        stage5_fit,
        stage5_samples,
        stage5_epf=stage5_epf,
        stage5_ecf=stage5_ecf,
        out_path=phase_png,
        pdf_path=phase_pdf,
    )
    obs_fig = make_checkpoint_observation_summary_figure(
        stage5_fit,
        stage5_samples,
        stage5_epf=stage5_epf,
        stage5_ecf=stage5_ecf,
        out_path=obs_png,
        pdf_path=obs_pdf,
        date_obs=date_obs,
        obs_ids=obs_ids,
    )
    return {
        'phase_folded': phase_png,
        'phase_folded_pdf': phase_pdf,
        'observation_summary': obs_png,
        'observation_summary_pdf': obs_pdf,
        'phase_folded_figure': phase_fig,
        'observation_summary_figure': obs_fig,
    }


def make_lightcurve_fit_figure(ds, planet, out_path=None):
    """
    Make the fitted-light-curve summary figure for one visit.

    Parameters
    ----------
    ds : xarray.Dataset
        Single-visit light-curve dataset.
    planet : str
        Planet name for the title.
    out_path : str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clip

    t_offset = int(np.floor(np.nanmin(ds.time.values)))
    mosaic = 'A;B;C'
    fig, axs = plt.subplot_mosaic(
        mosaic,
        figsize=(10 * 0.8, 7.5 * 0.8),
        sharex=True,
        gridspec_kw={'hspace': 0.075},
    )
    time = ds.time - t_offset
    raw = ds.rawFlux[0]
    err = ds.rawFluxErr[0]

    axs['A'].errorbar(
        time,
        sigma_clip(raw, sigma=20, maxiters=None),
        err,
        fmt='.',
        c='k',
        alpha=0.1,
    )
    axs['A'].plot(
        time,
        ds.fullModel[0],
        '-',
        c='r',
        lw=1,
        label='Full Fitted Model',
        zorder=np.inf,
    )

    axs['B'].errorbar(
        time,
        ds.cleanedFlux[0],
        err,
        fmt='.',
        c='k',
        alpha=0.1,
    )
    axs['B'].plot(
        time,
        ds.astroModel[0],
        '-',
        c='r',
        lw=1,
        label='Fitted Eclipse Model',
        zorder=np.inf,
    )

    axs['C'].errorbar(
        time,
        (raw - ds.fullModel[0]) * 1.0e6,
        err * 1.0e6,
        fmt='.',
        c='k',
        alpha=0.1,
    )
    axs['C'].axhline(0, c='r', lw=1, zorder=np.inf)

    axs['A'].set_title(f'{planet} Fiducial Light Curve Fit')
    axs['A'].set_ylabel('Raw Flux')
    axs['B'].set_ylabel('Cleaned Flux')
    axs['C'].set_ylabel('Residuals (ppm)')
    axs['C'].set_xlabel(f'Time (BJD_TDB - {t_offset})')
    axs['A'].legend(loc=1)
    axs['B'].legend(loc=1)
    fig.align_ylabels([axs['A'], axs['B'], axs['C']])

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    return fig


def make_raw_measurements_figure(ds, planet, out_path=None):
    """
    Make the raw-measurements diagnostic figure for one visit.

    Parameters
    ----------
    ds : xarray.Dataset
        Single-visit light-curve dataset.
    planet : str
        Planet name for the title.
    out_path : str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    import matplotlib.pyplot as plt
    from astropy.stats import sigma_clip

    t_offset = int(np.floor(np.nanmin(ds.time.values)))
    fig, axs = plt.subplot_mosaic(
        'A;B;C;D;E',
        figsize=(10 * 0.8, 7.5 * 0.8),
        sharex=True,
        gridspec_kw={'hspace': 0.1},
    )

    series = [
        ds.rawFlux,
        ds.centroid_x,
        ds.centroid_y,
        ds.centroid_sx,
        ds.centroid_sy,
    ]
    for axname, var in zip(list(axs), series):
        data = np.ma.masked_where(
            ~np.isfinite(var[0]) + ~np.isfinite(ds.rawFlux[0]),
            var[0],
        )
        data = sigma_clip(data, sigma=20, maxiters=None)
        axs[axname].plot(ds.time - t_offset, data, '.', c='k', ms=1)

    axs['A'].set_title(f'{planet} Raw Measurements')
    axs['A'].set_ylabel('Flux')
    axs['B'].set_ylabel('$x$')
    axs['C'].set_ylabel('$y$')
    axs['D'].set_ylabel('$x$ Width')
    axs['E'].set_ylabel('$y$ Width')
    axs['E'].set_xlabel(f'Time (BJD_TDB - {t_offset})')
    fig.align_ylabels([axs['A'], axs['B'], axs['C'], axs['D'], axs['E']])

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    return fig


def make_lightcurve_figures(ds, planet, out_dir='.', prefix=''):
    """
    Make and save both standard light-curve figures for one visit.

    Parameters
    ----------
    ds : xarray.Dataset
        Single-visit light-curve dataset.
    planet : str
        Planet name for figure titles.
    out_dir : str, optional
        Output directory.
    prefix : str, optional
        Filename prefix, e.g. ``'ecl001_'``.

    Returns
    -------
    paths : dict
        Paths to the generated figures.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raw_path = str(Path(out_dir) / f'{prefix}Figure1.png')
    fit_path = str(Path(out_dir) / f'{prefix}Figure2.png')
    fig_raw = make_raw_measurements_figure(ds, planet, raw_path)
    fig_fit = make_lightcurve_fit_figure(ds, planet, fit_path)
    return {
        'raw_measurements': raw_path,
        'fit_summary': fit_path,
        'raw_measurements_figure': fig_raw,
        'fit_summary_figure': fig_fit,
    }
