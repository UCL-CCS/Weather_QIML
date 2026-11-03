"""
WeatherBench 2 ERA5 data loading and preparation.

Loads ERA5 variables used by experiments, including:
  - geopotential at 500 hPa (z500)
  - vorticity at 500 hPa (vo500)
  - temperature at 850 hPa (t850)
  - 10 m zonal wind (u10)
  - vertical velocity at 500 hPa (w500)
  - total precipitation at 6-hour accumulation (tp6h)
from the WB2 Zarr store on Google Cloud Storage (public, no authentication
required). Data is 6-hourly at 1.5 deg resolution (120 lat x 240 lon,
cropped from 121 for patch-size compatibility).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

# WB2 public GCS paths
ERA5_ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
CLIMATOLOGY_ZARR = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)
GCS_OPTS = dict(token="anon")

HOURS_PER_STEP = 6  # WB2 temporal resolution

# Variables to load: (zarr_variable_name, pressure_level or None for surface/derived)
# None level means surface variable (no level dimension)
VARIABLES: List[Tuple[str, Optional[int]]] = [
    ("geopotential", 500),   # Z500
    ("vorticity", 500),      # VO500 (relative vorticity)
    ("temperature", 850),    # T850
    ("10m_u_component_of_wind", None),  # U10 (surface)
    ("vertical_velocity", 500),  # W500
    ("total_precipitation_6hr", None),  # TP6H accumulated precipitation
]
VAR_NAMES = ["z500", "vo500", "t850", "u10", "w500", "tp6h"]  # human-readable names matching VARIABLES


def load_era5_data(
    train_years: Tuple[int, int] = (1979, 2015),
    val_years: Tuple[int, int] = (2016, 2017),
    cache_dir: str = "cache",
    variables: List[Tuple[str, Optional[int]]] = VARIABLES,
    var_names: List[str] = VAR_NAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ERA5 data from WB2, split by year ranges.

    Downloads from Google Cloud on first call, caches locally as .npy files.

    Returns:
        data_train: float32 array (T_train, C, 120, 240).
        data_val:   float32 array (T_val,   C, 120, 240).
        lat: float64 array (120,) latitude values in degrees.
        lon: float64 array (240,) longitude values in degrees.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    var_tag = "_".join(var_names)
    train_file = cache_path / f"train_{var_tag}_{train_years[0]}_{train_years[1]}.npy"
    val_file   = cache_path / f"val_{var_tag}_{val_years[0]}_{val_years[1]}.npy"
    coords_file = cache_path / "coords.npz"

    n_vars = len(variables)
    EXPECTED_SHAPE = (n_vars, 120, 240)  # (C, lat, lon)

    if train_file.exists() and val_file.exists() and coords_file.exists():
        data_train = np.load(train_file)
        if data_train.shape[1:] == EXPECTED_SHAPE:
            print(f"Loading cached data from {cache_dir}/")
            data_val = np.load(val_file)
            coords = np.load(coords_file)
            lat, lon = coords["lat"], coords["lon"]
            T, C, H, W = data_train.shape
            print(
                f"Loaded {C} variables ({', '.join(var_names)}): "
                f"train={T} steps ({train_years[0]}-{train_years[1]}), "
                f"val={data_val.shape[0]} steps ({val_years[0]}-{val_years[1]}), "
                f"grid {H}x{W}"
            )
            return data_train, data_val, lat, lon
        else:
            print(
                f"Cache shape mismatch: got {data_train.shape[1:]}, "
                f"expected {EXPECTED_SHAPE}. Re-downloading..."
            )

    print("Downloading ERA5 data from WeatherBench 2...")
    print("(This may take several minutes on first run — data is cached afterwards)")
    ds = xr.open_zarr(ERA5_ZARR, chunks={"time": 200}, storage_options=GCS_OPTS)

    lat_arr, lon_arr = None, None
    train_channels, val_channels = [], []

    for var_name, level in variables:
        da = ds[var_name].sel(level=level) if level is not None else ds[var_name]
        da = da.transpose("time", "latitude", "longitude")

        if lat_arr is None:
            lat_arr = da["latitude"].values
            lon_arr = da["longitude"].values

        print(f"  Fetching {var_name} @ {level} hPa — train ({train_years[0]}-{train_years[1]})...")
        z_tr = (
            da.sel(time=slice(f"{train_years[0]}-01-01", f"{train_years[1]}-12-31"))
            .compute().values.astype(np.float32)
        )
        print(f"  Fetching {var_name} @ {level} hPa — val ({val_years[0]}-{val_years[1]})...")
        z_val = (
            da.sel(time=slice(f"{val_years[0]}-01-01", f"{val_years[1]}-12-31"))
            .compute().values.astype(np.float32)
        )

        # Crop latitude from 121 to 120 so H is divisible by 32.
        train_channels.append(z_tr[:, :120, :])
        val_channels.append(z_val[:, :120, :])

    lat_arr = lat_arr[:120]

    # Stack channels: (T, C, H, W)
    data_train = np.stack(train_channels, axis=1)
    data_val   = np.stack(val_channels,   axis=1)

    if not np.isfinite(data_train).all() or not np.isfinite(data_val).all():
        raise ValueError("Data contains NaN or Inf values.")

    np.save(train_file, data_train)
    np.save(val_file,   data_val)
    np.savez(coords_file, lat=lat_arr, lon=lon_arr)

    T, C, H, W = data_train.shape
    print(
        f"Loaded {C} variables ({', '.join(var_names)}): "
        f"train={T} steps, val={data_val.shape[0]} steps, grid {H}x{W}"
    )
    print(f"Cached to {cache_dir}/")
    return data_train, data_val, lat_arr, lon_arr


def load_climatology() -> xr.Dataset:
    """Load WB2 pre-computed ERA5 hourly climatology (needed for ACC)."""
    print("Loading WB2 climatology...")
    return xr.open_zarr(CLIMATOLOGY_ZARR, chunks=None, storage_options=GCS_OPTS)


def build_pairs(
    data: np.ndarray, lead_hours: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised (input, target) pairs with the given lead time.

    Args:
        data: Array of shape (T, C, H, W).
        lead_hours: Forecast lead time in hours (must be multiple of 6).

    Returns:
        x: float32 array (N, C, H, W) inputs.
        y: float32 array (N, C, H, W) targets.
    """
    assert lead_hours % HOURS_PER_STEP == 0, (
        f"lead_hours={lead_hours} must be a multiple of {HOURS_PER_STEP}"
    )
    lead_steps = lead_hours // HOURS_PER_STEP
    x = data[:-lead_steps]   # (N, C, H, W)
    y = data[lead_steps:]
    return x, y


class TemporalForecastDataset(Dataset):
    """Lazy (history, future) windows over a (T, n_vars, H, W) ERA5 array."""

    def __init__(self, data: np.ndarray, lead_hours: int, history_frames: int = 1, future_frames: int = 4):
        if lead_hours % HOURS_PER_STEP != 0:
            raise ValueError(f"lead_hours={lead_hours} must be multiple of {HOURS_PER_STEP}")
        self.data = data
        self.lead_steps = lead_hours // HOURS_PER_STEP
        self.history_frames = int(history_frames)
        self.future_frames = int(future_frames)
        self.t_total, self.n_vars, self.h, self.w = data.shape
        self.start_t = self.history_frames - 1
        self.end_t = self.t_total - self.lead_steps - self.future_frames
        if self.end_t < self.start_t:
            raise ValueError(
                "Not enough timesteps for requested framing. "
                f"T={self.t_total}, history_frames={self.history_frames}, "
                f"future_frames={self.future_frames}, lead_steps={self.lead_steps}"
            )
        self.n = self.end_t - self.start_t + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        t = self.start_t + int(idx)
        x_seq = self.data[t - self.history_frames + 1 : t + 1]
        y_seq = self.data[t + self.lead_steps : t + self.lead_steps + self.future_frames]
        x = x_seq.reshape(self.history_frames * self.n_vars, self.h, self.w).astype(np.float32, copy=False)
        y = y_seq.reshape(self.future_frames * self.n_vars, self.h, self.w).astype(np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def get_lat_weights(lat: np.ndarray) -> torch.Tensor:
    """Compute cosine-of-latitude area weights, normalized to mean 1."""
    weights = np.cos(np.deg2rad(lat)).astype(np.float32)
    weights = weights / weights.mean()
    return torch.from_numpy(weights)


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Reverse z-score normalization: x * std + mean."""
    return x * std + mean
