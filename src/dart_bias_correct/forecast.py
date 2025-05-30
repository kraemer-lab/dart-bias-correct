"""Bias correction module (forecast)"""

import logging
import itertools
from typing import Literal, NamedTuple
from pathlib import Path

import numpy as np
import xarray as xr
import metpy.calc as mp
from metpy.units import units
from cmethods import adjust

from .util import get_dart_root

logger = logging.getLogger(__name__)

# TODO: This is temporary, should be fixed to use geoglue.region
LATITUDE_SLICE = slice(11.25, 10)
LONGITUDE_SLICE = slice(106, 107.25)


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


def adjust_wrapper_quantiles(
    n_quantiles: int,
    obs: xr.Dataset | xr.DataArray,
    simh: xr.Dataset | xr.DataArray,
    simp: xr.Dataset | xr.DataArray,
    kind: Literal["+", "*"],
) -> xr.Dataset | xr.DataArray:
    """Function to correct extreme values located in the tails of the distribution

    Parameters
    ----------
    n_quantiles
        Number of quantiles, passed as the `n_quantiles` parameter
        to cmethods.adjust
    obs
        Historical ERA5 data that will be used as reference for
        quantile mapping correction
    simh
        Historical weather forecast data
    simp
        Real-time forecast data that we want to correct
    kind
        Type of quantile delta mapping "+" is additive
        (for temperature and humidity) whereas "*" is for precipitation

    See Also
    --------
    cmethods.adjust
        This is a thin wrapper around this bias correction method
    """
    return adjust(
        method="quantile_delta_mapping",
        obs=obs,
        simh=simh,
        simp=simp,
        n_quantiles=n_quantiles,  # Default number of quantiles for extreme correction
        kind=kind,  # "+"" for non tp and "*"" for tp
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
    rh = (
        mp.relative_humidity_from_dewpoint(
            np.array(data_raw_forecast.t2m) * units.kelvin,
            np.array(data_raw_forecast.d2m) * units.kelvin,
        )
        * 100
    ).magnitude
    sh = mp.specific_humidity_from_dewpoint(
        np.array(data_raw_forecast.sp) * units.pascal,
        np.array(data_raw_forecast.d2m) * units.kelvin,
    ).to("kg/kg")
    ws = mp.wind_speed(
        np.array(data_raw_forecast.u10) * units.meters / units.second,
        np.array(data_raw_forecast.v10) * units.meters / units.second,
    ).to("m/s")

    # Now we will include these variables into the main dataset
    data_vars = {"rh": rh, "sh": sh, "ws": ws}

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
        data_raw_forecast[["tp", "ssrd"]].resample(step="1D").sum(dim="step")
    )  # getting accumulated variables
    data_raw_forecast_inst = (
        data_raw_forecast.drop_vars(["tp", "ssrd"]).resample(step="1D").mean(dim="step")
    )  # getting instanteous variables

    # Weekly aggregation
    # TODO: Note that this is different from aggregation in the primary pipeline (DART-Pipeline)
    #       where maximum is taken over the day and overall mean is calculated
    weekly_mean = data_raw_forecast_inst.resample(step="7D").mean(dim="step")[
        ["t2m", "rh", "sh", "ws"]
    ]
    weekly_max = (
        data_raw_forecast_inst.resample(step="7D")
        .max(dim="step")
        .rename_vars({"t2m": "mx2t24", "rh": "mxrh24", "sh": "mxsh24"})[
            ["mx2t24", "mxrh24", "mxsh24"]
        ]
    )
    weekly_min = (
        data_raw_forecast_inst.resample(step="7D")
        .min(dim="step")
        .rename_vars({"t2m": "mn2t24", "rh": "mnrh24", "sh": "mnsh24"})[
            ["mn2t24", "mnrh24", "mnsh24"]
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
    weekly_sum1 = data_raw_forecast_accum.sel(step="7.days")[["tp", "ssrd"]]

    # In order to get the accumulated sum for the second week, we need to
    # subtract results from day 14 minus the 7th day, (and then assign again
    # the step coordinates as after the subtraction the coordinates disappear

    weekly_sum2 = (
        data_raw_forecast_accum.sel(step="14.days")[["tp", "ssrd"]]
        - data_raw_forecast_accum.sel(step="7.days")[["tp", "ssrd"]]
    ).assign_coords(step=data_raw_forecast_accum.sel(step="14.days").step)

    weekly_sum = xr.concat([weekly_sum1, weekly_sum2], dim="step")

    logger.info("Assigning accumulative variables in weekly aggregation")
    # Assign the variables to the dataset
    for var in ["tp", "ssrd"]:
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


def bias_correct_forecast(
    era5_hist: xr.Dataset,
    data_hist_forecast: xr.Dataset,
    data_raw_forecast: xr.Dataset,
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
    corrected_forecast = corrected_forecast[["t2m", "rh", "tp"]]

    # bool_dataset keeps track of lat,lon grid points that have already been
    # corrected Correction takes place according to percentile values with
    # extreme values being corrected differently to non-extremal (10-90
    # percentile) values, see PERCENTILES array
    bool_dataset = xr.Dataset(
        {
            var: (
                corrected_forecast[var].dims,
                np.full(corrected_forecast[var].shape, 0, dtype=int),
            )
            for var in corrected_forecast.data_vars
        },
        coords=corrected_forecast.coords,
    )

    for step in [7, 14]:  # 2 weeks in advance
        logger.info("Correction at step=%d", step)
        # Use ensemble mean for correction
        forecast = data_hist_forecast.sel(step=f"{step}.days").mean(dim="number")

        # Selecting reanalysis data with the same starting weeks as the forecast
        reanalysis = {7: era_week1, 14: era_week2}[step]
        masks = {var: [] for var in ["t2m", "rh", "tp"]}

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
            grid = itertools.product(
                range(len(forecast.lat)),
                range(len(forecast.lon)),
                range(len(data_raw_forecast.number)),
            )
            for la, lo, s in grid:
                data_to_corr_or = weekly_raw_forecast.sel(number=s)
                for var in masks:
                    kind = "*" if var == "tp" else "+"

                    # Identify the lat and lon point where we have reanalysis
                    # data between a specific threshold and locate forecast
                    # data in the same timestamp in the forecast
                    inter_forecast = (
                        forecast[var]
                        .where(masks[var][m])
                        .sel(lat=forecast.lat[la], lon=forecast.lon[lo])
                        .dropna(dim="time")
                    )
                    inter_reanalysis = (
                        reanalysis[var]
                        .where(masks[var][m])
                        .sel(lat=reanalysis.lat[la], lon=reanalysis.lon[lo])
                        .dropna(dim="time")
                    )

                    # Get the same latitudinal and longitudinal point for the data to correct
                    data_to_corr = data_to_corr_or[var].sel(
                        lat=data_to_corr_or.lat[la], lon=data_to_corr_or.lon[lo]
                    )

                    # Now, we will search if in the forecast data exists values
                    # of the variable  to correct that are contained inside the
                    # percentile threshold considered in the historical
                    # forecast data (inter_forecast)
                    data_to_corr = data_to_corr.where(
                        (data_to_corr >= inter_forecast.min())
                        & (data_to_corr < inter_forecast.max()),
                        drop=True,
                    )

                    if data_to_corr.size != 0:
                        corr_data = adjust_wrapper_quantiles(
                            ADJUST_N_QUANTILES[percentile.is_extreme],
                            inter_reanalysis,
                            inter_forecast,
                            data_to_corr,
                            kind=kind,
                        )
                    else:
                        continue

                    # Now, in the following lines we are checking in the dummy
                    # copy of the dataset if, in the same coordinates the
                    # number is different from 0 if it is, it means that one of
                    # the values that we corrected was already processed, hence
                    # we will delete the repeated value from the correction
                    selected_data = bool_dataset[var].loc[
                        dict(
                            time=data_to_corr.time,
                            number=s,
                            lat=data_to_corr.lat,
                            lon=data_to_corr.lon,
                        )
                    ]
                    filtered_data = selected_data.where(selected_data < 1, drop=True)

                    if filtered_data.shape != corr_data[var].shape:
                        # if filtered_data has a different shape from
                        # corr_data, it means that one of the values was
                        # already processed, hence we will filter the repeated
                        # value from the corrected dataset
                        corr_data = corr_data.sel(time=filtered_data.time)

                    # Now, we will include the corrected values into the xarray dataset
                    corrected_forecast[var].loc[
                        dict(
                            time=corr_data.time,
                            number=s,
                            lat=corr_data.lat,
                            lon=corr_data.lon,
                        )
                    ] = corr_data[var].values

                    # We increase the dummy dataset values by one in the processed coordinates to mark that the specified coordinates was already preprocessed
                    bool_dataset[var].loc[
                        dict(
                            time=corr_data.time,
                            number=s,
                            lat=corr_data.lat,
                            lon=corr_data.lon,
                        )
                    ] += 1
            logger.info("Finished correction at %r", percentile)

    # The bias correction method deletes units for relative humidity so we need to rewrite it
    corrected_forecast["rh"] = corrected_forecast["rh"] / 100
    corrected_forecast["rh"].attrs["units"] = "percent"
    corrected_forecast["rh"] = corrected_forecast["rh"].metpy.quantify()
    corrected_forecast["rh"] = corrected_forecast["rh"].metpy.convert_units(
        units.percent
    )

    # Now we add the new corrected values to the main processed xarray
    weekly_raw_forecast = weekly_raw_forecast.assign(
        {
            "t2m_bc": corrected_forecast.t2m,
            "rh_bc": corrected_forecast.rh,
            "tp_bc": corrected_forecast.tp,
        }
    )

    for var in ["rh", "rh_bc"]:
        weekly_raw_forecast[var] = weekly_raw_forecast[var] * units("percent")
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
            f"Accumulative variables file not found: {instant_file}"
        )
    instant = xr.open_dataset(instant_file, decode_timedelta=True)
    accum = xr.open_dataset(accum_file, decode_timedelta=True)
    return xr.merge([instant, accum])


def get_corrected_forecast_path(ecmwf_forecast_iso3_date: str) -> Path:
    dart_root = get_dart_root()
    parts = ecmwf_forecast_iso3_date.split("-")
    iso3 = parts[0]
    date = "-".join(parts[1:])
    ecmwf_root = dart_root / "sources" / iso3 / "ecmwf"
    return ecmwf_root / f"{iso3}-{date}-ecmwf.forecast.corrected.nc"


def print_dataset(ds: xr.Dataset | xr.DataArray, name: str):
    print("-" * (79 - len(name)) + " " + name)
    print(ds)


def crop(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords:
        return ds.sel(lat=LATITUDE_SLICE, lon=LONGITUDE_SLICE)
    else:
        return ds.sel(latitude=LATITUDE_SLICE, longitude=LONGITUDE_SLICE)


def bias_correct_forecast_from_paths(
    era5_hist_path: Path,
    data_hist_forecast_path: Path,
    ecmwf_forecast_iso3_date: str,
) -> Path:
    logger.info("Starting bias correct forecast")
    logger.info("Reading historical observational data: %s", era5_hist_path)
    logger.info("Reading historical forecast data: %s", data_hist_forecast_path)
    era5_hist = xr.open_dataset(era5_hist_path)
    if "r" in era5_hist.variables:
        era5_hist = era5_hist.rename_vars({"r": "rh"})
    data_hist_forecast = xr.open_dataset(data_hist_forecast_path, decode_timedelta=True)

    data_hist_forecast = crop(data_hist_forecast)
    era5_hist = crop(era5_hist)

    # TODO: Assert shape equal
    logger.info("Reading ECMWF forecast data for: %s", ecmwf_forecast_iso3_date)
    data_raw_forecast = get_forecast_dataset(ecmwf_forecast_iso3_date)

    data_raw_forecast = crop(data_raw_forecast)

    output_path = get_corrected_forecast_path(ecmwf_forecast_iso3_date)
    logger.info("Expected output path on successful correction: %s", output_path)
    ds = bias_correct_forecast(era5_hist, data_hist_forecast, data_raw_forecast)
    ds.to_netcdf(output_path)
    logger.info("Correction complete, file saved at: %s", output_path)
    return output_path
