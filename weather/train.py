"""Train the ERA5 Koopman rollout: QIML (``--qprior-on``) or the matched baseline.

The network predicts one physical step and is unrolled autoregressively in
training. Shared CLI defaults live in ``koopman.parse_args()``.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, Dataset

import koopman as base
from data import (
    HOURS_PER_STEP,
    VAR_NAMES,
    VARIABLES,
    TemporalForecastDataset,
    build_pairs,
    denormalize,
    get_lat_weights,
    load_climatology,
    load_era5_data,
)
from evaluate import anomaly_correlation, get_climatology_for_targets, lat_weighted_rmse


def _experiment_suffix(tag: str) -> str:
    t = (tag or "").strip()
    if not t:
        return ""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in t)
    return f"_{safe}"


@torch.no_grad()
def estimate_norm_stats_xy_separate(
    dataset: Dataset,
    device: torch.device,
    max_batches: int = 64,
    batch_size: int = 8,
):
    """
    Per-channel mean/std for x and y separately (no broadcast sum when Cx != Cy).

    x: (B, Cx, H, W), y: (B, Cy, H, W) — required for history_frames=1,future=4 with n_vars=1, etc.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sum_x = sum_x_sq = None
    sum_y = sum_y_sq = None
    count_x = 0
    count_y = 0
    for bi, (xb, yb) in enumerate(loader):
        if bi >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        b, _, h, w = xb.shape
        sx = xb.sum(dim=(0, 2, 3), keepdim=True)
        sy = yb.sum(dim=(0, 2, 3), keepdim=True)
        sqx = (xb * xb).sum(dim=(0, 2, 3), keepdim=True)
        sqy = (yb * yb).sum(dim=(0, 2, 3), keepdim=True)
        if sum_x is None:
            sum_x = sx
            sum_x_sq = sqx
            sum_y = sy
            sum_y_sq = sqy
        else:
            sum_x = sum_x + sx
            sum_x_sq = sum_x_sq + sqx
            sum_y = sum_y + sy
            sum_y_sq = sum_y_sq + sqy
        count_x += b * h * w
        count_y += b * h * w
    mean_x = sum_x / max(count_x, 1)
    var_x = (sum_x_sq / max(count_x, 1)) - mean_x * mean_x
    std_x = torch.sqrt(torch.clamp(var_x, min=1e-6))
    mean_y = sum_y / max(count_y, 1)
    var_y = (sum_y_sq / max(count_y, 1)) - mean_y * mean_y
    std_y = torch.sqrt(torch.clamp(var_y, min=1e-6))
    return mean_x.cpu(), std_x.cpu(), mean_y.cpu(), std_y.cpu()


def _rollout_train_step(
    koopman_core: nn.Module,
    xb_n: torch.Tensor,
    yb_n: torch.Tensor,
    point_loss_fn,
    args,
    n_vars: int,
    future_frames: int,
    qprior_cfg=None,
    mean=None,
    std=None,
):
    """Rollout loss for one minibatch (normalized space)."""
    width = xb_n.shape[-1]
    current_n = xb_n
    total_fwd = torch.tensor(0.0, device=xb_n.device)
    total_id = torch.tensor(0.0, device=xb_n.device)
    total_bwd = torch.tensor(0.0, device=xb_n.device)
    total_qp = torch.tensor(0.0, device=xb_n.device)
    preds_n = []

    for k in range(future_frames):
        x_small = base._to_small_lon(current_n, args.lon_downsample)
        h_small, w_small = x_small.shape[-2], x_small.shape[-1]
        x_koop = base._resize_to_square(x_small, args.koopman_input_size)
        x_hwc = x_koop.permute(0, 2, 3, 1).contiguous()

        out_pred, out_identity = koopman_core(x_hwc, mode="forward")
        pred_small = nn.functional.interpolate(out_pred[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
        id_small = nn.functional.interpolate(out_identity[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
        pred_full_n = base._to_full_lon(pred_small, width, args.lon_downsample)
        preds_n.append(pred_full_n)

        yk_n = yb_n[:, k * n_vars : (k + 1) * n_vars]
        yk_small = base._to_small_lon(yk_n, args.lon_downsample)
        total_fwd = total_fwd + point_loss_fn(pred_small, yk_small)
        total_id = total_id + point_loss_fn(id_small, x_small)

        if args.koopman_steps_back > 0:
            yk_koop = base._resize_to_square(yk_small, args.koopman_input_size)
            yk_hwc = yk_koop.permute(0, 2, 3, 1).contiguous()
            out_back, _ = koopman_core(yk_hwc, mode="backward")
            back_small = nn.functional.interpolate(out_back[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
            total_bwd = total_bwd + point_loss_fn(back_small, x_small)

        if qprior_cfg is not None:
            total_qp = total_qp + base.koopman_qprior_ns_style_loss(
                koopman_core=koopman_core,
                x_bchw=current_n,
                mean=mean,
                std=std,
                qprior_cfg=qprior_cfg,
                lon_downsample=args.lon_downsample,
                koopman_input_size=args.koopman_input_size,
            )

        current_n = pred_full_n

    if future_frames > 0:
        total_qp = total_qp / future_frames
    pred_stack_n = torch.cat(preds_n, dim=1)
    return pred_stack_n, total_fwd, total_id, total_bwd, total_qp


@torch.no_grad()
def evaluate_normalized_mse_lazy_ar(
    koopman_core: nn.Module,
    dataset,
    mean_x: torch.Tensor,
    std_x: torch.Tensor,
    mean_y: torch.Tensor,
    std_y: torch.Tensor,
    args,
    n_vars: int,
    future_frames: int,
    device: torch.device,
    batch_size: int = 8,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mean_x_d = mean_x.to(device)
    std_x_d = std_x.to(device)
    mean_y_d = mean_y.to(device)
    std_y_d = std_y.to(device)
    mx_n = mean_x_d[:, -n_vars:, :, :]
    sx_n = std_x_d[:, -n_vars:, :, :]
    total_se = 0.0
    total_el = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        xb_n = (xb[:, -n_vars:, :, :] - mx_n) / sx_n
        yb_n = (yb - mean_y_d) / std_y_d
        pred_n, _, _, _, _ = _rollout_train_step(
            koopman_core=koopman_core,
            xb_n=xb_n,
            yb_n=yb_n,
            point_loss_fn=base.LegacyLpLoss(size_average=False),
            args=args,
            n_vars=n_vars,
            future_frames=future_frames,
            qprior_cfg=None,
            mean=mx_n,
            std=sx_n,
        )
        total_se += (pred_n - yb_n).pow(2).sum().item()
        total_el += yb_n.numel()
    return total_se / max(total_el, 1)


@torch.no_grad()
def evaluate_wb2_1in4out_ar(
    model: torch.nn.Module,
    data_val_raw: np.ndarray,
    lead_hours: int,
    train_lead_hours: int,
    mean_x: torch.Tensor,
    std_x: torch.Tensor,
    mean_y: torch.Tensor,
    std_y: torch.Tensor,
    lat_weights: torch.Tensor,
    device: torch.device,
    val_years: tuple,
    clim_ds,
    batch_size: int = 64,
    variables: list = VARIABLES,
    var_names: list = VAR_NAMES,
    n_vars: int = 1,
) -> dict:
    if lead_hours % train_lead_hours != 0:
        return {
            "lead_hours": lead_hours,
            "normalized_mse": None,
            "eval_method": "1in4out_ar_rollout",
            "eval_scope": "free_rollout_from_truth_initial_state",
            "eval_note": f"Skipped: lead_hours {lead_hours} not divisible by train_lead_hours {train_lead_hours}.",
        }

    steps = lead_hours // train_lead_hours
    x_np, y_np = build_pairs(data_val_raw, lead_hours)
    x_raw = torch.from_numpy(np.ascontiguousarray(x_np))
    y_raw = torch.from_numpy(np.ascontiguousarray(y_np))
    mean_x_cpu = mean_x.cpu()
    std_x_cpu = std_x.cpu()
    mean_y0 = mean_y[:, :n_vars, :, :].cpu()
    std_y0 = std_y[:, :n_vars, :, :].cpu()
    mx_in = mean_x_cpu[:, -n_vars:, :, :]
    sx_in = std_x_cpu[:, -n_vars:, :, :]
    clim = get_climatology_for_targets(lead_hours, val_years, clim_ds, variables)

    preds = []
    model.eval()
    for i in range(0, len(x_raw), batch_size):
        xb = x_raw[i : i + batch_size]
        xb_n = (xb - mx_in) / sx_in
        cur = xb_n.to(device)
        expected_c = int(xb_n.shape[1])
        for _ in range(steps):
            cur = model(cur)
            if cur.shape[1] != expected_c:
                raise RuntimeError(
                    f"1in4out_ar expects one-step output channels={expected_c}, got {cur.shape[1]}"
                )
        preds.append(cur.cpu())
    pred_n = torch.cat(preds, dim=0)
    pred_phys = denormalize(pred_n, mean_y0, std_y0)
    y_n = (y_raw - mean_y0) / std_y0

    rmse_per_var = lat_weighted_rmse(pred_phys, y_raw, lat_weights)
    acc_per_var = anomaly_correlation(pred_phys, y_raw, clim, lat_weights)
    norm_mse = (pred_n - y_n).pow(2).mean().item()

    result = {
        "lead_hours": lead_hours,
        "normalized_mse": round(norm_mse, 6),
        "eval_method": "1in4out_ar_rollout",
        "eval_scope": "free_rollout_from_truth_initial_state",
        "rollout_steps": int(steps),
    }
    # rmse/acc are (C,) in dataset channel order (len == len(var_names)), not VAR_NAMES order.
    for j, name in enumerate(var_names):
        result[f"rmse_{name}"] = round(rmse_per_var[j].item(), 4)
        result[f"acc_{name}"] = round(acc_per_var[j].item(), 4)
    return result


def main():
    args = base.parse_args()
    base.set_seed(args.seed)

    history_frames = 1
    future_frames = 4

    if args.eval_var == "all":
        raise ValueError("This trainer supports --eval-var z500/t850/u10/w500/tp6h only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = torch.device("cuda")

    if args.qprior_on and args.qprior_channel < 0:
        args.qprior_channel = VAR_NAMES.index(args.eval_var)

    eval_idx = VAR_NAMES.index(args.eval_var)
    eval_variables = [VARIABLES[eval_idx]]
    eval_var_names = [VAR_NAMES[eval_idx]]

    data_train_raw, data_val_raw, lat, _ = load_era5_data(
        train_years=tuple(args.train_years),
        val_years=tuple(args.val_years),
        cache_dir=args.cache_dir,
        variables=eval_variables,
        var_names=eval_var_names,
    )
    n_vars = int(data_train_raw.shape[1])

    train_ds = TemporalForecastDataset(data_train_raw, lead_hours=args.train_lead_hours, history_frames=history_frames, future_frames=future_frames)
    val_ds = TemporalForecastDataset(data_val_raw, lead_hours=args.train_lead_hours, history_frames=history_frames, future_frames=future_frames)

    _x0, _y0 = train_ds[0]
    _xin = int(_x0.shape[0])
    _yin = int(_y0.shape[0])
    if _xin % n_vars != 0:
        raise ValueError(f"Input channels {_xin} not divisible by n_vars={n_vars}")
    _hist_eff = _xin // n_vars
    print(
        f"Data: train={len(train_ds)}, val={len(val_ds)}, n_vars={n_vars}, "
        f"x_ch={_xin} y_ch={_yin} (history_frames_eff={_hist_eff}; rollout uses last frame only, in_channel=n_vars)"
    )

    mean_x, std_x, _, _ = estimate_norm_stats_xy_separate(
        train_ds, device=device, max_batches=64, batch_size=args.batch_size
    )

    qprior_cfg = base.load_qprior_config(args, device)
    if qprior_cfg is not None:
        # one-step output has n_vars channels; constrain selected base variable each recurrent step
        qprior_cfg["channel"] = int(args.qprior_channel)

    if args.koopman_input_size % 32 != 0:
        raise ValueError(f"--koopman-input-size must be divisible by 32, got {args.koopman_input_size}")
    latent_hw = args.koopman_input_size // 32
    koopman_core = base.KoopmanAE2DTransFlexible(
        in_channel=n_vars,
        out_channel=n_vars,
        dim=args.koopman_dim,
        num_blocks=args.koopman_num_blocks,
        heads=args.koopman_heads,
        ffn_expansion_factor=args.koopman_ffn_expansion_factor,
        bias=False,
        LayerNorm_type="WithBias",
        steps=args.koopman_steps,
        steps_back=args.koopman_steps_back,
        grid_info=args.koopman_grid_info,
        latent_hw=latent_hw,
    ).to(device)
    model = base.KoopmanWeatherAdapter(
        koopman_core=koopman_core,
        lon_downsample=args.lon_downsample,
        koopman_input_size=args.koopman_input_size,
    ).to(device)

    point_loss_fn = base.LegacyLpLoss(size_average=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best_val = float("inf")
    best_state = None
    history = []
    epochs_no_improve = 0
    mean_x_dev = mean_x.to(device)
    std_x_dev = std_x.to(device)
    mean_x_narrow = mean_x_dev[:, -n_vars:, :, :]
    std_x_narrow = std_x_dev[:, -n_vars:, :, :]
    # Same z-score per variable as xb_n (not separate per-future-slot stats from xy_separate).
    mean_y_z_same_as_state = mean_x_narrow.repeat(1, future_frames, 1, 1).contiguous()
    std_y_z_same_as_state = std_x_narrow.repeat(1, future_frames, 1, 1).contiguous()
    mean_y_ckpt = mean_y_z_same_as_state.detach().cpu().clone()
    std_y_ckpt = std_y_z_same_as_state.detach().cpu().clone()
    t0 = base.time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = base.time.time()
        model.train()
        tr_tot = tr_fwd = tr_id = tr_bwd = tr_qp = 0.0
        for xb, yb in tqdm.tqdm(train_loader, desc=f"Train epoch {epoch:03d}", leave=False):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_n = (xb[:, -n_vars:, :, :] - mean_x_narrow) / std_x_narrow
            yb_n = (yb - mean_y_z_same_as_state) / std_y_z_same_as_state

            optimizer.zero_grad(set_to_none=True)
            pred_n, loss_fwd, loss_id, loss_bwd, loss_qp = _rollout_train_step(
                koopman_core=koopman_core,
                xb_n=xb_n,
                yb_n=yb_n,
                point_loss_fn=point_loss_fn,
                args=args,
                n_vars=n_vars,
                future_frames=future_frames,
                qprior_cfg=qprior_cfg if qprior_cfg is not None else None,
                mean=mean_x_narrow,
                std=std_x_narrow,
            )
            loss = (
                loss_fwd
                + args.koopman_identity_lambda * loss_id
                + args.koopman_backward_lambda * loss_bwd
                + loss_qp
            )
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = xb.size(0)
            tr_tot += loss.item() * bs
            tr_fwd += loss_fwd.item() * bs
            tr_id += loss_id.item() * bs
            tr_bwd += loss_bwd.item() * bs
            tr_qp += loss_qp.item() * bs

        ntr = len(train_loader.dataset)
        tr_tot /= ntr
        tr_fwd /= ntr
        tr_id /= ntr
        tr_bwd /= ntr
        tr_qp /= ntr

        model.eval()
        va_tot = va_fwd = va_id = va_bwd = va_qp = 0.0
        with torch.no_grad():
            for xb, yb in tqdm.tqdm(val_loader, desc=f"Val epoch {epoch:03d}", leave=False):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb_n = (xb[:, -n_vars:, :, :] - mean_x_narrow) / std_x_narrow
                yb_n = (yb - mean_y_z_same_as_state) / std_y_z_same_as_state
                _, loss_fwd, loss_id, loss_bwd, loss_qp = _rollout_train_step(
                    koopman_core=koopman_core,
                    xb_n=xb_n,
                    yb_n=yb_n,
                    point_loss_fn=point_loss_fn,
                    args=args,
                    n_vars=n_vars,
                    future_frames=future_frames,
                    qprior_cfg=qprior_cfg if qprior_cfg is not None else None,
                    mean=mean_x_narrow,
                    std=std_x_narrow,
                )
                loss = (
                    loss_fwd
                    + args.koopman_identity_lambda * loss_id
                    + args.koopman_backward_lambda * loss_bwd
                    + loss_qp
                )
                bs = xb.size(0)
                va_tot += loss.item() * bs
                va_fwd += loss_fwd.item() * bs
                va_id += loss_id.item() * bs
                va_bwd += loss_bwd.item() * bs
                va_qp += loss_qp.item() * bs

        nva = len(val_loader.dataset)
        va_tot /= nva
        va_fwd /= nva
        va_id /= nva
        va_bwd /= nva
        va_qp /= nva
        scheduler.step(va_tot)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train_total={tr_tot:.6f} | val_total={va_tot:.6f} | "
            f"val_fwd={va_fwd:.6f} | val_id={va_id:.6f} | val_back={va_bwd:.6f} | val_qprior={va_qp:.6f} | lr={lr:.2e} | "
            f"{base.time.time()-ep_start:.1f}s"
        )

        history.append(
            dict(
                epoch=epoch,
                train_total=round(tr_tot, 8),
                train_forward=round(tr_fwd, 8),
                train_identity=round(tr_id, 8),
                train_backward=round(tr_bwd, 8),
                train_qprior=round(tr_qp, 8),
                val_total=round(va_tot, 8),
                val_forward=round(va_fwd, 8),
                val_identity=round(va_id, 8),
                val_backward=round(va_bwd, 8),
                val_qprior=round(va_qp, 8),
                lr=lr,
            )
        )

        if va_tot < best_val - 1e-6:
            best_val = va_tot
            best_state = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val_total={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_mse = evaluate_normalized_mse_lazy_ar(
        koopman_core=koopman_core,
        dataset=val_ds,
        mean_x=mean_x,
        std_x=std_x,
        mean_y=mean_y_ckpt,
        std_y=std_y_ckpt,
        args=args,
        n_vars=n_vars,
        future_frames=future_frames,
        device=device,
        batch_size=args.batch_size,
    )

    lat_weights = get_lat_weights(lat)
    clim_ds = load_climatology()
    wb2_eval = {}
    run_tag = "with_qprior" if args.qprior_on else "no_qprior"
    exp_sfx = _experiment_suffix(getattr(args, "experiment_tag", ""))
    model_tag = "koopman1x4ar" if args.koopman_input_size == 64 else f"koopman{args.koopman_input_size}_1x4ar"
    if exp_sfx:
        print(f"Saving with filename suffix {exp_sfx!r} (--experiment-tag)")

    for lead_h in args.lead_hours:
        m = evaluate_wb2_1in4out_ar(
            model=model,
            data_val_raw=data_val_raw,
            lead_hours=int(lead_h),
            train_lead_hours=int(args.train_lead_hours),
            mean_x=mean_x,
            std_x=std_x,
            mean_y=mean_y_ckpt,
            std_y=std_y_ckpt,
            lat_weights=lat_weights,
            device=device,
            val_years=tuple(args.val_years),
            clim_ds=clim_ds,
            batch_size=args.batch_size,
            variables=eval_variables,
            var_names=eval_var_names,
            n_vars=n_vars,
        )
        wb2_eval[f"{model_tag}_{lead_h}h"] = m

    os.makedirs(args.plots_dir, exist_ok=True)
    base.plot_acc_decay(
        wb2_eval=wb2_eval,
        model_names=[model_tag],
        lead_hours=list(args.lead_hours),
        var_names=eval_var_names,
        save_dir=args.plots_dir,
        run_tag=f"{model_tag}_{args.eval_var}_{run_tag}{exp_sfx}",
    )

    framing_meta = {
        "history_frames": history_frames,
        "history_frames_effective_from_data": int(_hist_eff),
        "future_frames": future_frames,
        "n_vars_per_timestep": n_vars,
        "channel_layout": "time_major_then_var",
        "recurrent_rollout": True,
        "input_uses_last_history_frame_only": True,
        "model_step_hours": int(args.train_lead_hours),
        "target_zscore_same_as_state_per_var": True,
    }
    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": cpu_state,
            "config": vars(args),
            "framing": framing_meta,
            "best_val_mse": val_mse,
            "normalization_mean_x": mean_x,
            "normalization_std_x": std_x,
            "normalization_mean_y": mean_y_ckpt,
            "normalization_std_y": std_y_ckpt,
        },
        ckpt_dir / f"{model_tag}_{args.eval_var}_baseline_{run_tag}{exp_sfx}.pt",
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "run_tag": run_tag,
        "experiment_tag": getattr(args, "experiment_tag", "") or None,
        "qprior_enabled": args.qprior_on,
        "args": vars(args),
        "framing": framing_meta,
        "koopman_history": history,
        "final": {
            model_tag: {
                "val_mse": val_mse,
                "params": base.count_params(model),
                "total_time_s": round(base.time.time() - t0, 2),
            }
        },
        "wb2_evaluation": wb2_eval,
    }
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    baseline_path = (
        Path(args.results_dir) / f"baseline_results_{model_tag}_{args.eval_var}_{run_tag}{exp_sfx}.json"
    )
    baseline_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {baseline_path}")

    if args.qprior_on:
        lam_str = base._format_sci(float(args.qprior_lambda))
        model_file = (
            Path(args.results_dir)
            / f"{model_tag.upper()}_{args.eval_var}_results_with_qprior_{lam_str}{exp_sfx}.json"
        )
    else:
        model_file = (
            Path(args.results_dir) / f"{model_tag.upper()}_{args.eval_var}_results_no_qprior{exp_sfx}.json"
        )
    model_file.write_text(json.dumps(results, indent=2))
    print(f"Per-model results saved to {model_file}")


if __name__ == "__main__":
    main()

