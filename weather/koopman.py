"""Koopman backbone used by ``train.py`` (model, CLI, Q-Prior loss).

This module is not a separate experiment. Train QIML or the matched Koopman
baseline with ``python train.py``.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset as TorchTensorDataset

from data import VAR_NAMES

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_legacy_root(start_dir: Path) -> Path:
    """
    Find the project root that contains legacy Koopman dependencies.
    """
    for candidate in [start_dir] + list(start_dir.parents):
        if (candidate / "lib" / "pfnn_consist_2d.py").exists() and (candidate / "utilities.py").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing lib/pfnn_consist_2d.py and utilities.py")


LEGACY_ROOT = _find_legacy_root(SCRIPT_DIR)
LIB_DIR = LEGACY_ROOT / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))
if str(LEGACY_ROOT) not in sys.path:
    sys.path.append(str(LEGACY_ROOT))

RESULT_BASE = SCRIPT_DIR
default_qprior_path = RESULT_BASE / "results" / "qprior" / "qprior_pdf_z500_10q_lbfgs.npz"
if not default_qprior_path.exists():
    default_qprior_path = LEGACY_ROOT / "results" / "qprior" / "qprior_pdf_z500_10q_lbfgs.npz"
default_qprior_vort_path = RESULT_BASE / "results" / "qprior" / "qprior_pdf_vo500_10q_lbfgs.npz"
if not default_qprior_vort_path.exists():
    default_qprior_vort_path = LEGACY_ROOT / "results" / "qprior" / "qprior_pdf_vo500_10q_lbfgs.npz"

from pfnn_consist_2d import KoopmanAE_2d_trans  # noqa: E402
from vae_base import dynamics, dynamics_back  # noqa: E402


class LegacyLpLoss(object):
    """Standalone LpLoss to avoid importing utilities.py optional deps."""

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y, std=False):
        del std  # keep signature compatible with legacy usage
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, std=False):
        return self.rel(x, y, std)


class KoopmanAE2DTransFlexible(KoopmanAE_2d_trans):
    """
    Legacy Koopman with configurable latent spatial size.
    For 64-input path latent_hw=2; for 128-input path latent_hw=4.
    """

    def __init__(self, *args, latent_hw: int = 2, init_scale: float = 1.0, **kwargs):
        super().__init__(*args, init_scale=init_scale, **kwargs)
        self.latent_hw = latent_hw
        dyn_dim = self.latent_dim * self.latent_hw * self.latent_hw
        self.dynamics = dynamics(dyn_dim, init_scale)
        self.backdynamics = dynamics_back(dyn_dim, self.dynamics)

    def forward(self, x, mode='forward'):
        out = []
        out_id = []
        out_back = []
        out_back_id = []

        if self.grid_dim > 0:
            grid = self.get_grid(x.shape[1], x.shape[0], x.device)
            x_grid = torch.cat((x, grid.permute(0, 2, 3, 1)), dim=-1)
            x_grid = x_grid.permute(0, 3, 1, 2)
        else:
            x_grid = x.permute(0, 3, 1, 2)

        z = self.encoder(x_grid.contiguous())
        qt = z.contiguous().reshape(z.size(0), -1)

        if mode == 'forward':
            for _ in range(self.steps):
                q_t1 = self.dynamics(qt)
                q_t1 = q_t1.view(q_t1.size(0), self.latent_dim, self.latent_hw, self.latent_hw)
                out.append(self.decoder(q_t1))
            out_id.append(self.decoder(z.contiguous()))
            return out, out_id

        if mode == 'backward':
            for _ in range(self.steps_back):
                q_1t = self.backdynamics(qt)
                q_1t = q_1t.view(q_1t.size(0), self.latent_dim, self.latent_hw, self.latent_hw)
                out_back.append(self.decoder(q_1t))
            out_back_id.append(self.decoder(z.contiguous()))
            return out_back, out_back_id

        raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    p = argparse.ArgumentParser(description="Koopman weather experiment (ERA5).")
    # Data
    p.add_argument("--train-years", type=int, nargs=2, default=[1979, 2015])
    p.add_argument("--val-years", type=int, nargs=2, default=[2016, 2017])
    p.add_argument("--cache-dir", type=str, default=str(RESULT_BASE / "cache"))
    p.add_argument("--checkpoints-dir", type=str, default=str(RESULT_BASE / "checkpoints"))
    p.add_argument("--results-dir", type=str, default=str(RESULT_BASE / "results"))
    p.add_argument("--plots-dir", type=str, default=str(RESULT_BASE / "plots"))
    p.add_argument(
        "--train-vars",
        type=str,
        nargs="+",
        default=["all"],
        choices=list(VAR_NAMES) + ["all"],
        help="Variables to load/train (channel order follows this list). Default: all.",
    )
    # Lead times
    p.add_argument("--train-lead-hours", type=int, default=24)
    p.add_argument("--lead-hours", type=int, nargs="+", default=[24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480])
    p.add_argument(
        "--eval-var",
        type=str,
        choices=list(VAR_NAMES) + ["all"],
        default="z500",
        help="Metrics/maps focus (model still uses all channels from data.VARIABLES).",
    )
    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    # Koopman loss weights
    p.add_argument("--koopman-identity-lambda", type=float, default=1.0)
    p.add_argument("--koopman-backward-lambda", type=float, default=0.1)
    p.add_argument("--koopman-point-loss", type=str, choices=["lp"], default="lp")
    # Koopman architecture (legacy network, no structural changes)
    p.add_argument("--koopman-dim", type=int, default=8)
    p.add_argument("--koopman-num-blocks", type=int, nargs=4, default=[2, 2, 2, 2])
    p.add_argument("--koopman-heads", type=int, nargs=4, default=[1, 2, 4, 8])
    p.add_argument("--koopman-ffn-expansion-factor", type=float, default=2.66)
    p.add_argument("--koopman-steps", type=int, default=1)
    p.add_argument("--koopman-steps-back", type=int, default=1)
    p.add_argument("--koopman-grid-info", dest="koopman_grid_info", action="store_true")
    p.add_argument("--no-koopman-grid-info", dest="koopman_grid_info", action="store_false")
    p.set_defaults(koopman_grid_info=True)
    p.add_argument("--koopman-input-size", type=int, default=64)
    # Shape adapter
    p.add_argument("--lon-downsample", type=int, default=1, choices=[1, 2, 3, 4])
    # QPrior regularization
    p.add_argument("--qprior-on", action="store_true")
    p.add_argument(
        "--qprior-path",
        type=str,
        default=str(default_qprior_path),
    )
    # Master scale on the whole NS-style prior block (L2 legacy 'MMD' + KL).
    p.add_argument("--qprior-lambda", type=float, default=1.0)
    p.add_argument(
        "--qprior-channel",
        type=int,
        default=-1,
        help="Channel index for QPrior on training channels. Default -1: use --eval-var channel when eval-var != all.",
    )
    # Match train_q_NS_final_2_peak.py: multi-step rollout, then mean over steps.
    p.add_argument("--qprior-rollout-steps", type=int, default=1)
    # Soft assignment of z500 pixels to learned-PDF bins (analogous to NS temperature on logits).
    p.add_argument("--qprior-hist-tau", type=float, default=0.0, help=">0 fixes tau; 0 => auto from bin spacing")
    # Same weights as legacy NS script (gamma_2 MMD/L2, gamma_3 KL).
    p.add_argument("--qprior-gamma-l2", type=float, default=5.0)
    p.add_argument("--qprior-gamma-kl", type=float, default=5.0)
    # Dual scalar Q-Prior: independent priors/lambdas/channels for z500 and vo500.
    p.add_argument("--qprior-dual-on", action="store_true")
    p.add_argument("--qprior-z500-path", type=str, default=str(default_qprior_path))
    p.add_argument("--qprior-vort-path", type=str, default=str(default_qprior_vort_path))
    p.add_argument("--qprior-lambda-z500", type=float, default=1.0)
    p.add_argument("--qprior-lambda-vort", type=float, default=1.0e4)
    p.add_argument("--qprior-z500-channel", type=int, default=0, help="Channel index for z500 prior (default 0).")
    p.add_argument("--qprior-vort-channel", type=int, default=1, help="Channel index for vo500 prior (default 1).")
    p.add_argument(
        "--experiment-tag",
        type=str,
        default="",
        help="Optional suffix for saved checkpoints/JSON/plot names (avoid overwriting). Example: lam5 or qp_strong.",
    )
    p.add_argument(
        "--save-forecast-maps",
        action="store_true",
        help="If set, also save per-lead forecast map PNGs. Default: disabled.",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_single_qprior_config(
    *,
    prior_path: Path,
    lam: float,
    channel: int,
    rollout_steps: int,
    hist_tau_arg: float,
    gamma_l2: float,
    gamma_kl: float,
    device: torch.device,
    label: str,
):
    if not prior_path.exists():
        raise FileNotFoundError(f"QPrior file not found: {prior_path}")

    prior_data = np.load(prior_path)
    bin_centers = prior_data["bin_centers"].astype(np.float64)
    probs = prior_data["learned_smooth"].astype(np.float64) if "learned_smooth" in prior_data else prior_data["learned_pdf"].astype(np.float64)
    value_transform = str(prior_data["value_transform"]) if "value_transform" in prior_data else "identity"
    log1p_scale = float(prior_data["log1p_scale"]) if "log1p_scale" in prior_data else -1.0
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum()
    if len(bin_centers) > 1:
        dc = float(np.median(np.diff(np.sort(bin_centers))))
    else:
        dc = 1.0
    hist_tau = float(hist_tau_arg) if hist_tau_arg > 0 else max(dc, 1e-6)

    centers_t = torch.tensor(bin_centers, dtype=torch.float32, device=device)
    prior_t = torch.tensor(probs, dtype=torch.float32, device=device)

    cfg = {
        "label": label,
        "path": str(prior_path),
        "lambda": float(lam),
        "channel": int(channel),
        "rollout_steps": int(rollout_steps),
        "bin_centers": centers_t,
        "prior_probs": prior_t,
        "hist_tau": hist_tau,
        "gamma_l2": float(gamma_l2),
        "gamma_kl": float(gamma_kl),
        "value_transform": value_transform,
        "log1p_scale": log1p_scale,
        "standardize_mean": float(prior_data["standardize_mean"]) if "standardize_mean" in prior_data else np.nan,
        "standardize_std": float(prior_data["standardize_std"]) if "standardize_std" in prior_data else np.nan,
    }
    print(
        f"QPrior[{cfg['label']}] (NS-style: rollout mean hist + L2 + KL): path={prior_path}, channel={cfg['channel']}, "
        f"master_lambda={cfg['lambda']}, rollout_steps={cfg['rollout_steps']}, hist_tau={hist_tau:.6g}, "
        f"gamma_l2={cfg['gamma_l2']}, gamma_kl={cfg['gamma_kl']}, "
        f"value_transform={cfg['value_transform']}"
    )
    return cfg


def load_qprior_config(args, device: torch.device):
    if not args.qprior_on:
        return None
    return _load_single_qprior_config(
        prior_path=Path(args.qprior_path),
        lam=args.qprior_lambda,
        channel=args.qprior_channel,
        rollout_steps=args.qprior_rollout_steps,
        hist_tau_arg=args.qprior_hist_tau,
        gamma_l2=args.qprior_gamma_l2,
        gamma_kl=args.qprior_gamma_kl,
        device=device,
        label="single",
    )


def load_qprior_configs(args, device: torch.device):
    cfgs = []
    if args.qprior_on:
        cfgs.append(load_qprior_config(args, device))
    if args.qprior_dual_on:
        cfgs.append(
            _load_single_qprior_config(
                prior_path=Path(args.qprior_z500_path),
                lam=args.qprior_lambda_z500,
                channel=args.qprior_z500_channel,
                rollout_steps=args.qprior_rollout_steps,
                hist_tau_arg=args.qprior_hist_tau,
                gamma_l2=args.qprior_gamma_l2,
                gamma_kl=args.qprior_gamma_kl,
                device=device,
                label="z500",
            )
        )
        cfgs.append(
            _load_single_qprior_config(
                prior_path=Path(args.qprior_vort_path),
                lam=args.qprior_lambda_vort,
                channel=args.qprior_vort_channel,
                rollout_steps=args.qprior_rollout_steps,
                hist_tau_arg=args.qprior_hist_tau,
                gamma_l2=args.qprior_gamma_l2,
                gamma_kl=args.qprior_gamma_kl,
                device=device,
                label="vo500",
            )
        )
    return [c for c in cfgs if c is not None]


def soft_histogram_1d(values: torch.Tensor, centers: torch.Tensor, tau: float, chunk_size: int = 16384) -> torch.Tensor:
    """Differentiable soft histogram with chunking to reduce peak memory.

    Each pixel assigns mass to climate bins via a softmax over squared distances.
    We compute the same result as the naive implementation, but split pixels into
    value chunks so we never materialize a full (n_pixels, n_bins) logits matrix.
    """
    values_flat = values.reshape(-1)
    n_bins = int(centers.numel())
    c = centers.reshape(1, -1)  # (1, n_bins)

    hist_sum = torch.zeros(n_bins, device=values_flat.device, dtype=values_flat.dtype)
    count = 0

    denom = float(tau + 1e-12)
    for start in range(0, values_flat.numel(), int(chunk_size)):
        end = min(values_flat.numel(), start + int(chunk_size))
        v_chunk = values_flat[start:end].reshape(-1, 1)  # (m, 1)
        logits = -((v_chunk - c) / denom) ** 2  # (m, n_bins)
        w = torch.softmax(logits, dim=1)  # (m, n_bins)
        hist_sum = hist_sum + w.sum(dim=0)  # (n_bins,)
        count += w.shape[0]

    return hist_sum / max(count, 1)


def koopman_qprior_ns_style_loss(
    koopman_core: nn.Module,
    x_bchw: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    qprior_cfg: dict,
    lon_downsample: int,
    koopman_input_size: int,
) -> torch.Tensor:
    """
    Mirrors train_q_NS_final_2_peak.py (without peak term): autoregressive rollout, average
    per-step distributions, then L2 to prior (legacy script calls this MMD) and KL. The
    distribution is a soft histogram on climate PDF bins for physical z500 (channel).

    **Contract (same as single-step Koopman training):**
    ``x_bchw`` must be **z-scored** with the same ``mean``/``std`` as ``koopman_core`` sees in
    training. Core outputs are treated as normalized; ``pred * std + mean`` recovers physical
    units for the histogram. Rollout sets the next input to the full predicted stack, so any
    ``in_channel == out_channel`` head (including 4-in-4-out time stacks) is shape-valid; the
    channel index ``qprior_cfg["channel"]`` must name the **output** slot to monitor (e.g. last
    future frame of z500 for 4×4).
    """
    steps = max(1, int(qprior_cfg["rollout_steps"]))
    width = x_bchw.shape[-1]
    ch = qprior_cfg["channel"]
    centers = qprior_cfg["bin_centers"]
    prior = qprior_cfg["prior_probs"]
    tau = float(qprior_cfg["hist_tau"])

    current_small = _to_small_lon(x_bchw, lon_downsample)
    h_small, w_small = current_small.shape[-2], current_small.shape[-1]
    per_step_hists = []

    for _ in range(steps):
        current_koop = _resize_to_square(current_small, koopman_input_size)
        current_hwc = current_koop.permute(0, 2, 3, 1).contiguous()
        out_pred, _ = koopman_core(current_hwc, mode="forward")
        pred_koop = out_pred[0]
        pred_small = F.interpolate(pred_koop, size=(h_small, w_small), mode="bilinear", align_corners=False)
        pred_full = _to_full_lon(pred_small, width, lon_downsample)
        pred_phys = pred_full[:, ch : ch + 1] * std[:, ch : ch + 1] + mean[:, ch : ch + 1]
        transform_name = qprior_cfg.get("value_transform", "identity")
        if transform_name == "log1p":
            s = float(qprior_cfg.get("log1p_scale", 1.0e-3))
            if s <= 0:
                raise ValueError(f"Invalid qprior log1p_scale={s}")
            pred_for_hist = torch.log1p(torch.clamp(pred_phys, min=0.0) / s)
        elif transform_name == "standardize":
            mu = float(qprior_cfg.get("standardize_mean", np.nan))
            sigma = float(qprior_cfg.get("standardize_std", np.nan))
            if (not np.isfinite(mu)) or (not np.isfinite(sigma)) or sigma <= 0:
                raise ValueError(
                    f"QPrior standardize needs finite positive std; got mean={mu}, std={sigma}, path={qprior_cfg.get('path', '?')}"
                )
            pred_for_hist = (pred_phys - mu) / sigma
        else:
            pred_for_hist = pred_phys
        step_hist = soft_histogram_1d(pred_for_hist.reshape(-1), centers, tau)
        per_step_hists.append(step_hist)
        current_small = pred_small

    avg_hist = torch.stack(per_step_hists, dim=0).mean(dim=0)

    loss_l2 = torch.norm(avg_hist - prior, p=2)
    eps = 1e-8
    p = avg_hist + eps
    p = p / p.sum()
    q = prior + eps
    q = q / q.sum()
    loss_kl = F.kl_div(p.log(), q, reduction="batchmean")

    return qprior_cfg["lambda"] * (
        qprior_cfg["gamma_l2"] * loss_l2 + qprior_cfg["gamma_kl"] * loss_kl
    )


def plot_acc_decay(
    wb2_eval: dict,
    model_names: list,
    lead_hours: list,
    var_names: list,
    save_dir: str,
    run_tag: str,
):
    os.makedirs(save_dir, exist_ok=True)
    sorted_leads = sorted(set(lead_hours))
    for var_name in var_names:
        fig = plt.figure(figsize=(8, 5))
        has_curve = False
        for model_name in model_names:
            acc_values, lead_values = [], []
            for lead_h in sorted_leads:
                key = f"{model_name}_{lead_h}h"
                if key not in wb2_eval:
                    continue
                metric_key = f"acc_{var_name}"
                if metric_key not in wb2_eval[key]:
                    continue
                lead_values.append(lead_h)
                acc_values.append(wb2_eval[key][metric_key])
            if lead_values:
                has_curve = True
                plt.plot(lead_values, acc_values, marker="o", linewidth=2, label=model_name.upper())
        if not has_curve:
            plt.close(fig)
            continue
        plt.xlabel("Lead Time (hours)")
        plt.ylabel(f"ACC ({var_name.upper()})")
        plt.title(f"ACC Decay vs Lead Time ({var_name.upper()}, {run_tag})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = Path(save_dir) / f"acc_decay_{var_name}_{run_tag}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved ACC decay plot: {out_path}")


def _to_small_lon(x: torch.Tensor, downsample: int) -> torch.Tensor:
    if downsample <= 1:
        return x
    return x[:, :, :, ::downsample]


def _resize_to_square(x: torch.Tensor, size: int) -> torch.Tensor:
    """Resize BCHW fields to square size for legacy Koopman."""
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def _to_full_lon(x_small: torch.Tensor, width: int, downsample: int) -> torch.Tensor:
    """Map coarse-longitude tensors back to full WB2 width (240).

    Training loss is computed on the coarse grid; this path is for full-res API.
    Use bilinear upsampling so maps/GIFs are not dominated by repeat_interleave stripes.
    """
    if downsample <= 1:
        return x_small
    h = x_small.shape[-2]
    w = int(width)
    return F.interpolate(x_small, size=(h, w), mode="bilinear", align_corners=False)


class KoopmanWeatherAdapter(nn.Module):
    """
    Full-resolution BCHW wrapper around the legacy HWC Koopman core.

    Applies lon downsampling, bilinear resize to ``koopman_input_size`` square, then ``permute``
    to BHWC for the core. **Channel axis is never reshuffled** beyond the usual BCHW↔BHWC layout
    change; ``koopman_core`` decides ``in_channel`` / ``out_channel`` (equal for 1-step weather,
    different for N-time stacks such as 4-in-4-out).
    """

    def __init__(self, koopman_core: nn.Module, lon_downsample: int, koopman_input_size: int):
        super().__init__()
        self.koopman_core = koopman_core
        self.lon_downsample = lon_downsample
        self.koopman_input_size = koopman_input_size

    def forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C_in, H, W)`` → ``(B, C_out, H, W)`` at full longitude.

        ``C_in`` / ``C_out`` come from ``koopman_core`` (e.g. both ``n_vars`` for 1-step, or
        ``4*n_vars`` for 4-in-4-out). **No** slicing to a single frame: the full decoder head
        output is interpolated back to the input grid.
        """
        width = x_bchw.shape[-1]
        x_small = _to_small_lon(x_bchw, self.lon_downsample)
        h_small, w_small = x_small.shape[-2], x_small.shape[-1]
        x_koop = _resize_to_square(x_small, self.koopman_input_size)
        x_hwc = x_koop.permute(0, 2, 3, 1).contiguous()
        out_pred, _ = self.koopman_core(x_hwc, mode="forward")
        pred_koop = out_pred[0]
        pred_small = F.interpolate(pred_koop, size=(h_small, w_small), mode="bilinear", align_corners=False)
        return _to_full_lon(pred_small, width, self.lon_downsample)


def train_koopman(args, device, x_train, y_train, x_val, y_val, mean, std, qprior_cfgs=None):
    print("\n" + "=" * 60)
    print("  TRAINING KOOPMAN (legacy architecture)")
    print("=" * 60)

    lkw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        lkw["persistent_workers"] = True
    train_loader = DataLoader(TorchTensorDataset(x_train, y_train), shuffle=True, **lkw)
    val_loader = DataLoader(TorchTensorDataset(x_val, y_val), shuffle=False, **lkw)

    n_vars = x_train.shape[1]
    if args.koopman_input_size % 32 != 0:
        raise ValueError(f"--koopman-input-size must be divisible by 32, got {args.koopman_input_size}")
    latent_hw = args.koopman_input_size // 32
    koopman_core = KoopmanAE2DTransFlexible(
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
    model = KoopmanWeatherAdapter(
        koopman_core=koopman_core,
        lon_downsample=args.lon_downsample,
        koopman_input_size=args.koopman_input_size,
    ).to(device)

    n_params = count_params(model)
    print(f"Koopman params: {n_params:,}")

    # Keep optimization consistent with legacy Koopman scripts: LpLoss for all Koopman terms.
    point_loss_fn = LegacyLpLoss(size_average=False)
    id_loss_fn = LegacyLpLoss(size_average=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()
    mean_dev = mean.to(device)
    std_dev = std.to(device)

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        model.train()
        train_total_sum = train_forward_sum = train_identity_sum = train_back_sum = train_qprior_sum = 0.0
        train_iter = tqdm.tqdm(train_loader, desc=f"Train epoch {epoch:03d}", leave=False)
        for xb, yb in train_iter:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            width = xb.shape[-1]
            xb_small = _to_small_lon(xb, args.lon_downsample)
            yb_small = _to_small_lon(yb, args.lon_downsample)
            h_small, w_small = xb_small.shape[-2], xb_small.shape[-1]
            xb_koop = _resize_to_square(xb_small, args.koopman_input_size)
            yb_koop = _resize_to_square(yb_small, args.koopman_input_size)
            xb_hwc = xb_koop.permute(0, 2, 3, 1).contiguous()
            yb_hwc = yb_koop.permute(0, 2, 3, 1).contiguous()

            optimizer.zero_grad(set_to_none=True)
            out_pred, out_identity = koopman_core(xb_hwc, mode="forward")
            pred_small = F.interpolate(out_pred[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
            id_small = F.interpolate(out_identity[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
            loss_fwd = point_loss_fn(pred_small, yb_small)
            loss_id = id_loss_fn(id_small, xb_small)

            loss_bwd = torch.tensor(0.0, device=device)
            if args.koopman_steps_back > 0:
                out_back, _ = koopman_core(yb_hwc, mode="backward")
                back_small = F.interpolate(out_back[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
                loss_bwd = point_loss_fn(back_small, xb_small)

            loss_qprior = torch.tensor(0.0, device=device)
            if qprior_cfgs:
                for cfg in qprior_cfgs:
                    loss_qprior = loss_qprior + koopman_qprior_ns_style_loss(
                        koopman_core=koopman_core,
                        x_bchw=xb,
                        mean=mean_dev,
                        std=std_dev,
                        qprior_cfg=cfg,
                        lon_downsample=args.lon_downsample,
                        koopman_input_size=args.koopman_input_size,
                    )

            loss = (
                loss_fwd
                + args.koopman_identity_lambda * loss_id
                + args.koopman_backward_lambda * loss_bwd
            )
            if qprior_cfgs:
                loss = loss + loss_qprior

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = xb.size(0)
            train_total_sum += loss.item() * bs
            train_forward_sum += loss_fwd.item() * bs
            train_identity_sum += loss_id.item() * bs
            train_back_sum += loss_bwd.item() * bs
            train_qprior_sum += loss_qprior.item() * bs

        train_total = train_total_sum / len(train_loader.dataset)
        train_forward = train_forward_sum / len(train_loader.dataset)
        train_identity = train_identity_sum / len(train_loader.dataset)
        train_backward = train_back_sum / len(train_loader.dataset)
        train_qprior = train_qprior_sum / len(train_loader.dataset)

        model.eval()
        val_total_sum = val_forward_sum = val_identity_sum = val_back_sum = val_qprior_sum = 0.0
        with torch.no_grad():
            val_iter = tqdm.tqdm(val_loader, desc=f"Val epoch {epoch:03d}", leave=False)
            for xb, yb in val_iter:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                width = xb.shape[-1]
                xb_small = _to_small_lon(xb, args.lon_downsample)
                yb_small = _to_small_lon(yb, args.lon_downsample)
                h_small, w_small = xb_small.shape[-2], xb_small.shape[-1]
                xb_koop = _resize_to_square(xb_small, args.koopman_input_size)
                yb_koop = _resize_to_square(yb_small, args.koopman_input_size)
                xb_hwc = xb_koop.permute(0, 2, 3, 1).contiguous()
                yb_hwc = yb_koop.permute(0, 2, 3, 1).contiguous()

                out_pred, out_identity = koopman_core(xb_hwc, mode="forward")
                pred_small = F.interpolate(out_pred[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
                id_small = F.interpolate(out_identity[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
                loss_fwd = point_loss_fn(pred_small, yb_small)
                loss_id = id_loss_fn(id_small, xb_small)

                loss_bwd = torch.tensor(0.0, device=device)
                if args.koopman_steps_back > 0:
                    out_back, _ = koopman_core(yb_hwc, mode="backward")
                    back_small = F.interpolate(out_back[0], size=(h_small, w_small), mode="bilinear", align_corners=False)
                    loss_bwd = point_loss_fn(back_small, xb_small)

                loss_qprior = torch.tensor(0.0, device=device)
                if qprior_cfgs:
                    for cfg in qprior_cfgs:
                        loss_qprior = loss_qprior + koopman_qprior_ns_style_loss(
                            koopman_core=koopman_core,
                            x_bchw=xb,
                            mean=mean_dev,
                            std=std_dev,
                            qprior_cfg=cfg,
                            lon_downsample=args.lon_downsample,
                            koopman_input_size=args.koopman_input_size,
                        )

                loss = (
                    loss_fwd
                    + args.koopman_identity_lambda * loss_id
                    + args.koopman_backward_lambda * loss_bwd
                )
                if qprior_cfgs:
                    loss = loss + loss_qprior

                bs = xb.size(0)
                val_total_sum += loss.item() * bs
                val_forward_sum += loss_fwd.item() * bs
                val_identity_sum += loss_id.item() * bs
                val_back_sum += loss_bwd.item() * bs
                val_qprior_sum += loss_qprior.item() * bs

        val_total = val_total_sum / len(val_loader.dataset)
        val_forward = val_forward_sum / len(val_loader.dataset)
        val_identity = val_identity_sum / len(val_loader.dataset)
        val_backward = val_back_sum / len(val_loader.dataset)
        val_qprior = val_qprior_sum / len(val_loader.dataset)

        scheduler.step(val_total)
        ep_time = time.time() - ep_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:03d}/{args.epochs} | "
            f"train_total={train_total:.6f} | val_total={val_total:.6f} | "
            f"val_fwd={val_forward:.6f} | val_id={val_identity:.6f} | "
            f"val_back={val_backward:.6f} | val_qprior={val_qprior:.6f} | "
            f"lr={lr:.2e} | {ep_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_total": round(train_total, 8),
            "train_forward": round(train_forward, 8),
            "train_identity": round(train_identity, 8),
            "train_backward": round(train_backward, 8),
            "train_qprior": round(train_qprior, 8),
            "val_total": round(val_total, 8),
            "val_forward": round(val_forward, 8),
            "val_identity": round(val_identity, 8),
            "val_backward": round(val_backward, 8),
            "val_qprior": round(val_qprior, 8),
            "lr": lr,
            "time_s": round(ep_time, 2),
        })

        if val_total < best_val - 1e-6:
            best_val = val_total
            best_state = {
                k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}. Best val_total={best_val:.6f}")
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, n_params, elapsed, history


def _format_sci(x: float) -> str:
    s = f"{x:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")
