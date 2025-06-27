"""Bias correction module (forecast)"""

import logging
import itertools
import functools
import collections
import multiprocessing
from typing import Literal, NamedTuple
from pathlib import Path

import numpy as np
import xarray as xr
import metpy.calc as mp
from metpy.units import units
from cmethods import adjust
from geoglue.types import Bbox

from .util import get_dart_root

logger = logging.getLogger(__name__)

# r = relative_humidity, q = specific_humidity
INSTANT_VARS = ["t2m", "d2m", "sp", "u10", "v10", "r", "q"]
INSTANT_EXTREME_VARS = ["mx2t24", "mxr24", "mxq24", "mn2t24", "mnr24", "mnq24"]
ACCUM_VARS = ["tp"]
SUPPORTED_VARS = INSTANT_VARS + INSTANT_EXTREME_VARS + ACCUM_VARS

BIAS_CORRECT_VARS = ["t2m", "r", "tp"]


def instant_vars(ds: xr.Dataset, include_extremes: bool = False) -> xr.Dataset:
    _vars = INSTANT_VARS + INSTANT_EXTREME_VARS if include_extremes else INSTANT_VARS
    return ds[[v for v in ds.data_vars if v in _vars]]


def accum_vars(ds: xr.Dataset) -> xr.Dataset:
    return ds[[v for v in ds.data_vars if v in ACCUM_VARS]]


def supported_vars(ds: xr.Dataset) -> xr.Dataset:
    return ds[[v for v in ds.data_vars if v in SUPPORTED_VARS]]


class Percentile(NamedTuple):
    low: int
    high: int

    @property
    def is_extreme(self) -> bool:
        "Whether this is an extremal percentile (covers less than 50 percentile range)"
        return self.high - self.low < 50


PERCENTILES: list[Percentile] = [
    Percentile(0, 5),
    Percentile(5, 10),
    Percentile(10, 90),
    Percentile(90, 95),
    Percentile(95, 100),
]

# n_quantiles value to use in adjust_wrapper_quantiles for is_extreme=True or
# is_extreme=False
ADJUST_N_QUANTILES: dict[bool, int] = {True: 10, False: 200}
ADJUST_KIND: dict[str, Literal["+", "*"]] = {"t2m": "+", "r": "+", "tp": "*"}


def ensure_corrected_forecast_notnull(corrected_forecast: xr.Dataset):
    err = []
    if corrected_forecast.t2m_bc.isnull().any():
        err.append("Corrected temperature (t2m_bc) field has NA values")
    if corrected_forecast.r_bc.isnull().any():
        err.append("Corrected relative humidity (r_bc) field has NA values")
    if corrected_forecast.tp_bc.isnull().any():
        err.append("Corrected precipitation (tp_bc) field has NA values")
    if err:
        raise ValueError("\n  " + "\n  ".join(err))


def adjust_wrapper_quantiles(
    method: Literal["quantile_mapping", "quantile_delta_mapping"],
    n_quantiles: int,
    obs: xr.Dataset | xr.DataArray,
    simh: xr.Dataset | xr.DataArray,
    simp: xr.Dataset | xr.DataArray,
    kind: Literal["+", "*"] = "+",
) -> xr.Dataset | xr.DataArray:
    """Function to correct extreme values located in the tails of the distribution

    Parameters
    ----------
    method
        Method to use for bias correction, one of *quantile_mapping* or
        *quantile_delta_mapping*
    n_quantiles
        Number of quantiles, passed as the `n_quantiles` parameter
        to cmethods.adjust"
    obs
        Historical ERA5 data that will be used as reference for
        quantile mapping correction
    simh
        Historical weather forecast data
    simp
        Real-time forecast data that we want to correct
    kind
        Type of quantile delta mapping "+" is additive
        (for temperature and humidity) whereas "*" is for precipitation.
        Default value is "+". This parameter is not used when method='quantile_mapping'

    See Also
    --------
    cmethods.adjust
        This is a thin wrapper around this bias correction method
    """
    match method:
        case "quantile_delta_mapping":
            return adjust(
                method="quantile_delta_mapping",
                obs=obs,
                simh=simh,
                simp=simp,
                n_quantiles=n_quantiles,  # Default number of quantiles for extreme correction
                kind=kind,  # "+"" for non tp and "*"" for tp
            )
        case "quantile_mapping":
            return adjust(
                method="quantile_mapping",
                obs=obs,
                simh=simh,
                simp=simp,
                n_quantiles=n_quantiles,  # Default number of quantiles for extreme correction
            )


def weekly_stats_era5(
    dataset1: xr.Dataset,
    initial_time: np.ndarray,
    timestep: int,
    agg: Literal["mean", "sum"],
):
    """
    Function to measure the mean/sum statistics of a dataset following specific
    intervals of starting dates

    Parameters
    ----------
    dataset1 : xr.Dataset
        Dataset with time, latitude and longitude dimensions
    initial_time : np.ndarray
        Array with the starting dates (in datetime64 format)
    timestep: int
        Time window to be considered for the temporal statistics (integer).
        For example, if initial_time is a vector with 2 values (1st and 4th of Jan
        of 2010 and timestep=7, the function will measure the temporal statistics from 1st-7th
        and 4-11th of January.
    agg: Literal['mean', 'sum']
        Type of statistic desired being "sum" for total accumulation or "mean" for mean statistics

    Returns
    -------
    xr.Dataset
        Temporally aggregated xarray Dataset
    """

    final_time = initial_time + np.timedelta64(timestep, "D")
    dataset_time = dataset1.where(
        (dataset1.time >= initial_time[0]) & (dataset1.time < final_time[0]), drop=True
    )
    dataset_time = getattr(dataset_time, agg)(dim="time")  # call mean or sum on dataset
    dataset_time_end = dataset_time.expand_dims(time=[initial_time[0]])

    for i in range(1, len(initial_time)):
        dataset_time = dataset1.where(
            (dataset1.time >= initial_time[i]) & (dataset1.time < final_time[i]),
            drop=True,
        )
        inter = getattr(dataset_time, agg)(dim="time")
        inter = inter.expand_dims(time=[initial_time[i]])
        dataset_time_end = xr.concat([dataset_time_end, inter], dim="time")
    return dataset_time_end


def get_weekly_forecast(data_raw_forecast: xr.Dataset) -> xr.Dataset:
    "Returns weekly aggregated forecast with derived metrics"

    # RH is between 0 to 1, so we multiply by 100 and use .values to remove units
    r = (
        mp.relative_humidity_from_dewpoint(
            np.array(data_raw_forecast.t2m) * units.kelvin,
            np.array(data_raw_forecast.d2m) * units.kelvin,
        )
        * 100
    ).magnitude
    q = mp.specific_humidity_from_dewpoint(
        np.array(data_raw_forecast.sp) * units.pascal,
        np.array(data_raw_forecast.d2m) * units.kelvin,
    ).to("kg/kg")

    # Now we will include these variables into the main dataset
    data_vars = {"r": r, "q": q}

    # Create the xarray Dataset
    inter = xr.Dataset(
        {
            var_name: xr.DataArray(
                data=data,
                dims=["step", "latitude", "longitude", "number"],
                coords={
                    "step": data_raw_forecast.step,
                    "latitude": data_raw_forecast.latitude,
                    "longitude": data_raw_forecast.longitude,
                    "number": data_raw_forecast.number,
                },
            )
            for var_name, data in data_vars.items()
        }
    )

    data_raw_forecast = xr.merge([data_raw_forecast, inter])

    # Daily aggregation
    data_raw_forecast = data_raw_forecast.rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    data_raw_forecast_accum = (
        accum_vars(
            data_raw_forecast
        )  # .resample(step="1D").sum(dim="step") # In the case of accumulated variables, the forecast give us the total accumulation of the variable per timestep. THis means that if we select precipitation in step 7-14 days, it would select the accumulation of precipitation since beginning of forecast until day 7 or 14
    )
    data_raw_forecast_inst = instant_vars(data_raw_forecast)

    # Weekly aggregation
    weekly_mean = data_raw_forecast_inst.resample(step="7D").mean(dim="step")[
        ["t2m", "r", "q"]
    ]
    weekly_max = (
        data_raw_forecast_inst.resample(step="1D")
        .max(dim="step")
        .resample(step="7D")
        .mean(dim="step")
        .rename_vars({"t2m": "mx2t24", "r": "mxr24", "q": "mxq24"})[
            ["mx2t24", "mxr24", "mxq24"]
        ]
    )
    weekly_min = (
        data_raw_forecast_inst.resample(step="1D")
        .min(dim="step")
        .resample(step="7D")
        .mean(dim="step")
        .rename_vars({"t2m": "mn2t24", "r": "mnr24", "q": "mnq24"})[
            ["mn2t24", "mnr24", "mnq24"]
        ]
    )

    weekly_raw_forecast = xr.merge([weekly_mean, weekly_max, weekly_min])

    # Drop step 0 of weekly_raw_forecast
    # TODO: Check if this is OK
    # NOTE: This is done to match the accumulated variables which have steps at 7 and 14 days
    weekly_raw_forecast = weekly_raw_forecast.sel(
        step=weekly_raw_forecast.step != np.timedelta64(0)
    )

    # For the weekly accumulation the forecast works differently, as the
    # forecast shows the full sum of the variables since the beginning of the
    # forecast. hence, for the first weekly stats we can just select the 7th
    # day of forecast.
    weekly_sum1 = data_raw_forecast_accum.sel(step="7.days")[ACCUM_VARS]

    # In order to get the accumulated sum for the second week, we need to
    # subtract results from day 14 minus the 7th day, (and then assign again
    # the step coordinates as after the subtraction the coordinates disappear

    weekly_sum2 = (
        data_raw_forecast_accum.sel(step="14.days")[ACCUM_VARS]
        - data_raw_forecast_accum.sel(step="7.days")[ACCUM_VARS]
    ).assign_coords(step=data_raw_forecast_accum.sel(step="14.days").step)

    weekly_sum = xr.concat([weekly_sum1, weekly_sum2], dim="step")

    logger.info("Assigning accumulative variables in weekly aggregation")
    # Assign the variables to the dataset
    for var in ACCUM_VARS:
        weekly_raw_forecast[var] = xr.DataArray(
            data=weekly_sum[var].values,
            dims=["step", "lat", "lon", "number"],
            coords={
                "step": weekly_raw_forecast.step,
                "lat": weekly_raw_forecast.lat,
                "lon": weekly_raw_forecast.lon,
                "number": weekly_raw_forecast.number,
            },
        )

    # When doing the resampling, step vector gets changed to 1 and 8 days,
    # which represent when the code starts measuring the weekly mean (this is,
    # mean/sum of day 1-7 and 8-14) hence we rename the dimensions to avoid
    # confusion, being the first step the start of the forecast (week1) and the
    # second step the start of second forecast weeks
    weekly_raw_forecast["step"] = [
        np.timedelta64(0, "D"),
        np.timedelta64(7, "D"),
    ]  # timestep of 7 days at 00

    # Now we will assigning the time coordinates to be the same as the start of
    # the forecast, and drop empty dimensions
    weekly_raw_forecast = (
        weekly_raw_forecast.assign_coords(
            time=(weekly_raw_forecast["time"] + weekly_raw_forecast["step"])
        )
        .drop_vars(["heightAboveGround", "surface"])
        .swap_dims({"step": "time"})
    )
    return weekly_raw_forecast


def sim_coords(ds: xr.DataArray | xr.Dataset, number: int) -> dict:
    return {"time": ds["time"], "number": number, "lat": ds["lat"], "lon": ds["lon"]}


def bias_correct_forecast(
    era5_hist: xr.Dataset,
    data_hist_forecast: xr.Dataset,
    data_raw_forecast: xr.Dataset,
    method: Literal["quantile_mapping", "quantile_delta_mapping"],
):
    dates = np.array(
        np.array(data_hist_forecast.time)
    )  # Dtes in which the historical forecast data of the ECMWF begin

    logger.info("Calculating weekly statistics for historical data")
    week1_mean = weekly_stats_era5(
        era5_hist, initial_time=dates, timestep=7, agg="mean"
    ).drop_vars("tp")  # measuring daily mean temperature and relative humidity
    week1_sum = weekly_stats_era5(
        era5_hist.tp, initial_time=dates, timestep=7, agg="sum"
    )  # weekly sum
    week1 = xr.merge([week1_mean, week1_sum])
    week2_mean = weekly_stats_era5(
        era5_hist,
        initial_time=dates + data_raw_forecast.step[1].item(),
        timestep=7,
        agg="mean",
    ).drop_vars("tp")  # measuring daily mean temperature and relative humidity
    week2_sum = weekly_stats_era5(
        era5_hist.tp,
        initial_time=dates + data_raw_forecast.step[1].item(),
        timestep=7,
        agg="sum",
    )  # weekly sum
    week2 = xr.merge([week2_mean, week2_sum])

    era_week1 = week1.rename({"latitude": "lat", "longitude": "lon"})
    era_week2 = week2.rename({"latitude": "lat", "longitude": "lon"})

    # We put the same time in era_week1 and era_week2 because the historial forecast is
    # stored in a xarray dataset with time equal to the start of the forecast,
    # and a 2D variable with the forecasted week (steps); for the bias
    # correction technique to work we need to have the same times.
    era_week2["time"] = era_week1["time"]

    logger.info("Computing weekly aggregated forecast with derived metrics")
    weekly_raw_forecast = get_weekly_forecast(data_raw_forecast)
    corrected_forecast = weekly_raw_forecast.copy(deep=True)
    corrected_forecast = corrected_forecast[BIAS_CORRECT_VARS]

    # bool_dataset keeps track of lat,lon grid points that have already been
    # corrected Correction takes place according to percentile values with
    # extreme values being corrected differently to non-extremal (10-90
    # percentile) values, see PERCENTILES array
    bool_dataset = xr.Dataset(
        {
            var: (
                corrected_forecast[var].dims,
                np.full(corrected_forecast[var].shape, False, dtype=bool),
            )
            for var in corrected_forecast.data_vars
        },
        coords=corrected_forecast.coords,
    )

    for step in [7, 14]:  # 2 weeks in advance
        logger.info("Correction at step=%d", step)
        # Use ensemble mean for correction
        forecast = data_hist_forecast.sel(step=f"{step}.days").mean(dim="number")
        weekly_raw_forecast_to_corr = weekly_raw_forecast.where(
            weekly_raw_forecast.time == weekly_raw_forecast.time[int(step / 7) - 1],
            drop=True,
        )

        # Selecting reanalysis data with the same starting weeks as the forecast
        reanalysis = {7: era_week1, 14: era_week2}[step]
        masks = {var: [] for var in BIAS_CORRECT_VARS}

        # Create masks to handle extreme values differently
        for p in PERCENTILES:
            for var in masks:
                low_quantile = reanalysis[var].quantile(p.low / 100, dim="time")
                high_quantile = reanalysis[var].quantile(p.high / 100, dim="time")
                masks[var].append(
                    (reanalysis[var] >= low_quantile)
                    & (reanalysis[var] < high_quantile)
                )

        for m, percentile in enumerate(PERCENTILES):
            logger.info("Starting correction at %r", percentile)
            for s in range(len(data_raw_forecast.number)):
                data_to_corr_or = weekly_raw_forecast_to_corr.sel(number=s)
                for var in masks:
                    inter_forecast = forecast[var].where(masks[var][m])
                    inter_reanalysis = reanalysis[var].where(masks[var][m])

                    data_to_corr = data_to_corr_or[var]
                    # Filtering data from future forecast outside the percentile range considered
                    #data_to_corr = data_to_corr.where(
                    #    (data_to_corr >= inter_forecast.min(dim="time"))
                    #    & (data_to_corr < inter_forecast.max(dim="time")),
                    #    drop=True,
                    #)
                   low_th=data_hist_forecast[var].where(masks[var][m]).sel(step=f"{step}.days").min(dim="time").min(dim="number") # we search for the lowest value 
                   high_th=data_hist_forecast[var].where(masks[var][m]).sel(step=f"{step}.days").max(dim="time").max(dim="number")

                   valid_data=data_to_corr_or[var].where((data_to_corr_or[var]>=low_th)&
                                                               (data_to_corr_or[var]<high_th),drop=True)   

                    valid_data = data_to_corr
                    if valid_data.size == 0:  # nothing to correct
                        continue

                    # If we have at least 1 point that felt into the percentile interval to correct
                    # select corresponding lat/lon points in inter_reanalysis and inter_forecast
                    # and check that datapoints selected were not already corrected

                    filtered_valid_data = valid_data.where(
                        ~bool_dataset[var].loc[sim_coords(valid_data, s)],
                        drop=True,
                    )
                    if filtered_valid_data.size == 0:
                        continue

                    # Select corresponding lat/lon points in inter_reanalysis and inter_forecast to performe the bias correction
                    inter_reanalysis_corr = inter_reanalysis.sel(
                        lat=filtered_valid_data["lat"], lon=filtered_valid_data["lon"]
                    )
                    inter_forecast_corr = inter_forecast.sel(
                        lat=filtered_valid_data["lat"], lon=filtered_valid_data["lon"]
                    )
                    corr_data = adjust_wrapper_quantiles(
                        method,
                        ADJUST_N_QUANTILES[percentile.is_extreme],
                        inter_reanalysis_corr,
                        inter_forecast_corr,
                        filtered_valid_data,
                        ADJUST_KIND[var],
                    )
                    int_bool_dataset = (
                        bool_dataset[var]
                        .loc[sim_coords(corr_data, s)]
                        .stack(all_coords=("lat", "lon", "time"))
                    )
                    non_nan_corr = corr_data.stack(all_coords=("lat", "lon", "time"))

                    # Now we are going to retain corrected data that was not corrected and that is not NaN
                    corr_data = non_nan_corr.where(
                        non_nan_corr.notnull() & ~int_bool_dataset, drop=True
                    )

                    # Update corrected_forecast and bool_dataset for all valid points simultaneously
                    corrected_forecast[var].loc[sim_coords(corr_data, s)] = corr_data[
                        var
                    ].where(
                        corr_data[var].notnull(),
                        corrected_forecast[var].loc[sim_coords(corr_data, s)],
                    )
                    bool_dataset[var].loc[sim_coords(corr_data, s)] = True
                    # print(bool_dataset[var].loc[sim_coords(corr_data, s)])
            logger.info("Finished correction at %r", percentile)

    # The bias correction method deletes units for relative humidity so we need to rewrite it
    corrected_forecast["r"].attrs["units"] = "percent"

    # Now we add the new corrected values to the main processed xarray
    weekly_raw_forecast = weekly_raw_forecast.assign(
        {
            "t2m_bc": corrected_forecast.t2m,
            "r_bc": corrected_forecast.r,
            "tp_bc": corrected_forecast.tp,
        }
    )

    for var in ["r", "r_bc"]:
        weekly_raw_forecast[var] = weekly_raw_forecast[var] * units("percent")
    ensure_corrected_forecast_notnull(weekly_raw_forecast)
    return weekly_raw_forecast

    # NOTE: resampling is not performed in dart-bias-correct, processing pipelines in
    #       DART-Pipeline should resample to target grid as appropriate


def get_forecast_dataset(ecmwf_forecast_iso3_date: str) -> xr.Dataset:
    dart_root = get_dart_root()
    parts = ecmwf_forecast_iso3_date.split("-")
    iso3 = parts[0]
    date = "-".join(parts[1:])
    ecmwf_root = dart_root / "sources" / iso3 / "ecmwf"
    accum_file = ecmwf_root / f"{iso3}-{date}-ecmwf.forecast.accum.nc"
    instant_file = ecmwf_root / f"{iso3}-{date}-ecmwf.forecast.instant.nc"
    if not accum_file.exists():
        raise FileNotFoundError(f"Accumulative variables file not found: {accum_file}")
    if not instant_file.exists():
        raise FileNotFoundError(
            f"Instantaneous variables file not found: {instant_file}"
        )
    instant = xr.open_dataset(instant_file, decode_timedelta=True)
    accum = xr.open_dataset(accum_file, decode_timedelta=True)
    ds = xr.merge([instant, accum])
    return supported_vars(ds)


def get_corrected_forecast_path(ecmwf_forecast_iso3_date: str, parallel: bool) -> Path:
    dart_root = get_dart_root()
    parts = ecmwf_forecast_iso3_date.split("-")
    iso3 = parts[0]
    date = "-".join(parts[1:])
    ecmwf_root = dart_root / "sources" / iso3 / "ecmwf"
    _parallel = "_parallel" if parallel else ""
    return ecmwf_root / f"{iso3}-{date}-ecmwf.forecast.corrected{_parallel}.nc"


def print_dataset(ds: xr.Dataset | xr.DataArray, name: str):
    print("-" * (79 - len(name)) + " " + name)
    print(ds)


def crop(ds: xr.Dataset, bbox: Bbox) -> xr.Dataset:
    if "lat" in ds.coords:
        return ds.sel(lat=bbox.lat_slice, lon=bbox.lon_slice)
    else:
        return ds.sel(latitude=bbox.lat_slice, longitude=bbox.lon_slice)


def bias_correct_forecast_from_paths(
    era5_hist_path: Path,
    data_hist_forecast_path: Path,
    ecmwf_forecast_iso3_date: str,
    bbox: str | Bbox | None = None,
    method: Literal["quantile_mapping", "quantile_delta_mapping"] = "quantile_mapping",
) -> Path:
    if isinstance(bbox, str):
        bbox = Bbox.from_string(bbox)
    logger.info("Starting bias correct forecast using %s", method)
    logger.info("Reading historical observational data: %s", era5_hist_path)
    logger.info("Reading historical forecast data: %s", data_hist_forecast_path)
    era5_hist = xr.open_dataset(era5_hist_path)
    data_hist_forecast = xr.open_dataset(data_hist_forecast_path, decode_timedelta=True)
    if "rh" in era5_hist.variables:
        era5_hist = era5_hist.rename_vars({"rh": "r"})
    if "rh" in data_hist_forecast.variables:
        data_hist_forecast = data_hist_forecast.rename_vars({"rh": "r"})

    # Use era5_hist bounds if no bbox supplied
    bbox = bbox or Bbox.from_xarray(era5_hist)
    logger.info("Cropping datasets to %r", bbox)
    era5_hist = crop(supported_vars(era5_hist), bbox)
    data_hist_forecast = crop(supported_vars(data_hist_forecast), bbox)

    # TODO: Assert shape equal
    logger.info("Reading ECMWF forecast data for: %s", ecmwf_forecast_iso3_date)
    data_raw_forecast = get_forecast_dataset(ecmwf_forecast_iso3_date)

    data_raw_forecast = crop(data_raw_forecast, bbox)

    output_path = get_corrected_forecast_path(ecmwf_forecast_iso3_date, parallel=False)
    logger.info("Expected output path on successful correction: %s", output_path)
    ds = bias_correct_forecast(era5_hist, data_hist_forecast, data_raw_forecast, method)
    ds.to_netcdf(output_path)
    logger.info("Correction complete, file saved at: %s", output_path)
    return output_path
