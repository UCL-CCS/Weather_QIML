"""Local NetCDF loader used by the optional ``--data-source local_nc`` Q-Prior path."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr


def load_era5_z500(data_dir: str, years: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load z500 NetCDF files for the specified years, sort by time, deduplicate."""
    data_dir = Path(data_dir)
    z_chunks, t_chunks = [], []

    for year in sorted(years):
        pattern = f"geopotential_500hPa_{year}_5.625deg.nc"
        matches = list(data_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file found for year {year} in {data_dir}")
        for fpath in matches:
            with xr.open_dataset(fpath) as ds:
                da = ds["z"].transpose("time", "lat", "lon")
                z_chunks.append(np.asarray(da.values, dtype=np.float32))
                t_chunks.append(np.asarray(da["time"].values))

    z = np.concatenate(z_chunks, axis=0)
    times = np.concatenate(t_chunks, axis=0)

    order = np.argsort(times)
    z, times = z[order], times[order]
    _, unique_idx = np.unique(times, return_index=True)
    z, times = z[np.sort(unique_idx)], times[np.sort(unique_idx)]

    if not np.isfinite(z).all():
        raise ValueError("Data contains NaN or Inf values.")

    print(
        f"Loaded z500: {z.shape[0]} timesteps, grid {z.shape[1]}x{z.shape[2]}, "
        f"years {min(years)}-{max(years)}"
    )
    return z, times


def build_pairs(z: np.ndarray, lead_steps: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised (input, target) pairs with the given lead time."""
    x = z[:-lead_steps, None, :, :]
    y = z[lead_steps:, None, :, :]
    return x, y
