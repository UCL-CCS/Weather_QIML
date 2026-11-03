"""
WeatherBench 2-style evaluation metrics.

Provides per-variable latitude-weighted RMSE and Anomaly Correlation
Coefficient (ACC) for multi-variable forecasts, following WB2 conventions.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from data import HOURS_PER_STEP, VARIABLES, VAR_NAMES, build_pairs, denormalize


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Run model on all inputs and return predictions on CPU."""
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(device)
        preds.append(model(xb).cpu())
    return torch.cat(preds, dim=0)


def lat_weighted_rmse(
    pred: torch.Tensor,
    truth: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted RMSE per channel, in physical units.

    Args:
        pred:  (N, C, H, W) predictions in physical units.
        truth: (N, C, H, W) targets in physical units.
        lat_weights: (H,) cosine-latitude weights, mean-normalized to 1.

    Returns:
        Tensor of shape (C,) with RMSE per variable.
    """
    se = (pred - truth).pow(2)                   # (N, C, H, W)
    w = lat_weights.view(1, 1, -1, 1)             # (1, 1, H, 1)
    # Mean over N, H, W (weighted); shape -> (C,)
    return (se * w).mean(dim=(0, 2, 3)).sqrt()


def anomaly_correlation(
    pred: torch.Tensor,
    truth: torch.Tensor,
    clim: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    """Anomaly Correlation Coefficient per channel.

    Args:
        pred:  (N, C, H, W) predictions in physical units.
        truth: (N, C, H, W) targets in physical units.
        clim:  (N, C, H, W) climatology values for each target timestep.
        lat_weights: (H,) weights.

    Returns:
        Tensor of shape (C,) with ACC per variable.
    """
    w = lat_weights.view(1, 1, -1, 1)

    pred_anom  = pred  - clim
    truth_anom = truth - clim

    # Sum over spatial dims per sample, then average over time
    num = (w * pred_anom * truth_anom).sum(dim=(2, 3))         # (N, C)
    dp  = (w * pred_anom.pow(2)).sum(dim=(2, 3)).sqrt()         # (N, C)
    dt  = (w * truth_anom.pow(2)).sum(dim=(2, 3)).sqrt()        # (N, C)

    acc_per_sample = num / (dp * dt + 1e-8)                     # (N, C)
    return acc_per_sample.mean(dim=0)                            # (C,)


def get_climatology_for_targets(
    lead_hours: int,
    years: tuple,
    clim_ds: xr.Dataset,
    variables: list = VARIABLES,
) -> torch.Tensor:
    """Extract climatology values matching target timesteps (vectorized).

    Returns:
        Tensor of shape (N, C, 120, 240).
    """
    lead_steps = lead_hours // HOURS_PER_STEP

    start = pd.Timestamp(f"{years[0]}-01-01")
    end   = pd.Timestamp(f"{years[1]}-12-31T18:00:00")
    all_times    = pd.date_range(start, end, freq="6h")
    target_times = all_times[lead_steps:]

    doys  = xr.DataArray(target_times.dayofyear.values, dims="sample")
    hours = xr.DataArray(target_times.hour.values,      dims="sample")

    channels = []
    for var_name, level in variables:
        da = clim_ds[var_name].sel(level=level) if level is not None else clim_ds[var_name]
        da = da.transpose("dayofyear", "hour", "latitude", "longitude")
        arr = da.sel(dayofyear=doys, hour=hours).values.astype(np.float32)  # (N, 121, 240)
        arr = arr[:, :120, :]                                                # crop lat
        channels.append(arr)

    clim_arr = np.stack(channels, axis=1)   # (N, C, 120, 240)
    return torch.from_numpy(clim_arr)


@torch.no_grad()
def evaluate_normalized_mse(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Per-element MSE averaged over all channels and spatial dims."""
    model.eval()
    loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )
    total_se, total_elements = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total_se       += (pred - yb).pow(2).sum().item()
        total_elements += yb.numel()
    return total_se / total_elements


@torch.no_grad()
def evaluate_wb2(
    model: torch.nn.Module,
    data_val_raw: np.ndarray,
    lead_hours: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    lat_weights: torch.Tensor,
    device: torch.device,
    val_years: tuple,
    clim_ds: xr.Dataset,
    batch_size: int = 64,
    variables: list = VARIABLES,
    var_names: list = VAR_NAMES,
) -> dict:
    """Full WB2-style evaluation at a given lead time.

    Returns dict with normalized_mse and per-variable rmse + acc.
    """
    model.eval()

    x_np, y_np = build_pairs(data_val_raw, lead_hours)
    x_raw = torch.from_numpy(np.ascontiguousarray(x_np))
    y_raw = torch.from_numpy(np.ascontiguousarray(y_np))

    x_norm    = (x_raw - mean) / std
    pred_norm = predict_all(model, x_norm, device, batch_size)
    pred_phys = denormalize(pred_norm, mean, std)

    y_norm   = (y_raw - mean) / std
    norm_mse = (pred_norm - y_norm).pow(2).mean().item()

    n_model_ch = int(pred_phys.shape[1])
    rmse_per_ch = lat_weighted_rmse(pred_phys, y_raw, lat_weights)  # (C_model,)

    clim = get_climatology_for_targets(lead_hours, val_years, clim_ds, variables)
    n_clim_ch = int(clim.shape[1])

    if n_clim_ch == n_model_ch:
        acc_per_ch = anomaly_correlation(pred_phys, y_raw, clim, lat_weights)
    else:
        acc_per_ch = torch.full((n_model_ch,), float("nan"))
        for ci, (var_name_full, level) in enumerate(variables):
            matching_name = var_names[ci] if ci < len(var_names) else None
            if matching_name is None:
                continue
            model_ci = var_names.index(matching_name) if matching_name in var_names else -1
            if model_ci < 0 or model_ci >= n_model_ch:
                continue
            clim_1ch = clim[:, ci : ci + 1]
            pred_1ch = pred_phys[:, model_ci : model_ci + 1]
            y_1ch = y_raw[:, model_ci : model_ci + 1]
            acc_per_ch[model_ci] = anomaly_correlation(pred_1ch, y_1ch, clim_1ch, lat_weights).squeeze()

    result = {
        "lead_hours": lead_hours,
        "normalized_mse": round(norm_mse, 6),
    }
    for j, name in enumerate(var_names):
        if j >= n_model_ch:
            break
        result[f"rmse_{name}"] = round(rmse_per_ch[j].item(), 4)
        result[f"acc_{name}"] = round(acc_per_ch[j].item(), 4) if not acc_per_ch[j].isnan() else None
    return result


VAR_UNITS = {"z500": "m²/s²", "t850": "K", "u10": "m/s"}


def _load_cartopy():
    """Return Cartopy modules when available; otherwise use plain Matplotlib."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ModuleNotFoundError:
        return None, None
    return ccrs, cfeature


def _add_cyclic_longitude(lon: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close the longitude seam for global map plotting."""
    if lon.ndim != 1 or field.shape[-1] != lon.size or lon.size < 2:
        return lon, field
    lon_span = float(lon[-1] - lon[0])
    if lon_span >= 359.9:
        return lon, field
    lon_step = float(np.median(np.diff(lon)))
    cyclic_lon = np.concatenate([lon, [lon[-1] + lon_step]])
    cyclic_field = np.concatenate([field, field[..., :1]], axis=-1)
    return cyclic_lon, cyclic_field


def _plot_geo_panel(
    ax,
    fig,
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    ccrs=None,
    cfeature=None,
):
    """Plot one lat/lon field, using Cartopy coastlines when available."""
    plot_lon, plot_field = _add_cyclic_longitude(lon, field)
    if ccrs is not None:
        im = ax.pcolormesh(
            plot_lon,
            lat,
            plot_field,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        ax.set_global()
        ax.coastlines(linewidth=0.6)
        if cfeature is not None:
            ax.add_feature(cfeature.BORDERS, linewidth=0.25, alpha=0.5)
        grid = ax.gridlines(draw_labels=True, linewidth=0.25, alpha=0.4)
        grid.top_labels = False
        grid.right_labels = False
    else:
        im = ax.pcolormesh(
            plot_lon,
            lat,
            plot_field,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            shading="auto",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)


def plot_world_map_comparison(
    truth: np.ndarray,
    predictions: dict,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    var_names: list = VAR_NAMES,
    model_label: str | None = None,
    lead_hours: int | None = None,
    sample_idx: int | None = None,
    save_path: str,
    metric_text=None,
    use_cartopy: bool = True,
):
    """Plot ground truth, predictions, and errors on world maps.

    Args:
        truth: Array of shape (C, H, W), in physical units.
        predictions: Mapping from model name to array of shape (C, H, W).
        lat/lon: 1D coordinate arrays matching H/W.
        save_path: Output PNG path.
    """
    if truth.ndim != 3:
        raise ValueError(f"truth must have shape (C, H, W), got {truth.shape}")
    if not predictions:
        raise ValueError("predictions must contain at least one model.")

    clean_predictions = {}
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        if pred.shape != truth.shape:
            raise ValueError(
                f"Prediction {name!r} has shape {pred.shape}; expected {truth.shape}"
            )
        clean_predictions[str(name)] = pred

    ccrs, cfeature = _load_cartopy() if use_cartopy else (None, None)
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}

    n_vars = len(var_names)
    n_models = len(clean_predictions)
    n_cols = 1 + 2 * n_models
    fig_width = max(16, 5.5 * n_cols)
    fig, axes = plt.subplots(
        n_vars,
        n_cols,
        figsize=(fig_width, 4.7 * n_vars),
        subplot_kw=subplot_kw,
    )
    axes = np.asarray(axes)
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    for row, vn in enumerate(var_names):
        unit = VAR_UNITS.get(vn, "")
        truth_field = truth[row]
        pred_values = list(clean_predictions.values())
        value_min = min([truth_field.min()] + [pred[row].min() for pred in pred_values])
        value_max = max([truth_field.max()] + [pred[row].max() for pred in pred_values])
        error_abs = max(
            max(abs((pred[row] - truth_field).min()), abs((pred[row] - truth_field).max()))
            for pred in pred_values
        )

        _plot_geo_panel(
            axes[row, 0],
            fig,
            lon,
            lat,
            truth_field,
            title=f"Ground Truth - {vn.upper()}",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
            colorbar_label=unit,
            ccrs=ccrs,
            cfeature=cfeature,
        )

        col = 1
        for name, pred in clean_predictions.items():
            pred_field = pred[row]
            error_field = pred_field - truth_field
            _plot_geo_panel(
                axes[row, col],
                fig,
                lon,
                lat,
                pred_field,
                title=f"{name} Prediction - {vn.upper()}",
                cmap="viridis",
                vmin=value_min,
                vmax=value_max,
                colorbar_label=unit,
                ccrs=ccrs,
                cfeature=cfeature,
            )
            _plot_geo_panel(
                axes[row, col + 1],
                fig,
                lon,
                lat,
                error_field,
                title=f"{name} Error - {vn.upper()}",
                cmap="RdBu_r",
                vmin=-error_abs,
                vmax=error_abs,
                colorbar_label=unit,
                ccrs=ccrs,
                cfeature=cfeature,
            )
            col += 2

    title_parts = []
    if model_label:
        title_parts.append(model_label)
    if lead_hours is not None:
        title_parts.append(f"{lead_hours}h forecast")
    if sample_idx is not None:
        title_parts.append(f"sample {sample_idx}")
    title = " - ".join(title_parts) if title_parts else "Forecast comparison"
    if metric_text:
        title = f"{title}\n{metric_text}"
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved world map comparison: {save_path}")


@torch.no_grad()
def plot_forecast_maps(
    model: torch.nn.Module,
    data_val_raw: np.ndarray,
    lead_hours: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    device: torch.device,
    model_name: str,
    var_names: list = VAR_NAMES,
    sample_idx: int = 0,
    save_dir: str = "results",
    metric_text=None,
):
    """Plot truth / prediction / error maps for a single sample.

    Saves a PNG with one row per variable and 3 columns (Truth, Pred, Error).
    Uses Cartopy coastlines when Cartopy is installed, otherwise falls back to
    Matplotlib lat/lon axes.
    """
    model.eval()
    x_np, y_np = build_pairs(data_val_raw, lead_hours)
    x_sample = torch.from_numpy(np.ascontiguousarray(x_np[sample_idx : sample_idx + 1]))
    y_sample = torch.from_numpy(np.ascontiguousarray(y_np[sample_idx : sample_idx + 1]))

    x_norm = (x_sample - mean) / std
    pred_norm = model(x_norm.to(device)).cpu()
    pred_phys = denormalize(pred_norm, mean, std)

    truth = y_sample[0].numpy()   # (C, H, W)
    pred  = pred_phys[0].numpy()  # (C, H, W)

    ch_idx = [VAR_NAMES.index(vn) for vn in var_names]
    truth_selected = truth[ch_idx]
    pred_selected = pred[ch_idx]

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_{lead_hours}h_forecast.png")
    plot_world_map_comparison(
        truth=truth_selected,
        predictions={model_name.upper(): pred_selected},
        lat=lat,
        lon=lon,
        var_names=var_names,
        model_label=model_name.upper(),
        lead_hours=lead_hours,
        sample_idx=sample_idx,
        save_path=path,
        metric_text=metric_text,
    )
