"""Plot Z500 world maps from QIML and matched Koopman checkpoints."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

import koopman as kwm
from data import HOURS_PER_STEP, VARIABLES, VAR_NAMES, build_pairs, load_era5_data
from evaluate import plot_world_map_comparison


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Plot QIML / Koopman checkpoint predictions on world maps.")
    p.add_argument("--checkpoint-dir", type=str, default=str(SCRIPT_DIR / "checkpoints"))
    p.add_argument(
        "--checkpoint-glob",
        action="append",
        default=None,
        help="Glob inside checkpoint-dir. Can be repeated. Default: *.pt",
    )
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Explicit .pt files. If supplied, checkpoint-dir/globs are ignored.",
    )
    p.add_argument("--max-checkpoints", type=int, default=0, help="0 means no limit.")
    p.add_argument(
        "--models-per-figure",
        type=int,
        default=2,
        help="Split large comparisons into readable chunks. 0 means put all models in one figure.",
    )
    p.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR / "plots" / "checkpoint_world_maps"))
    p.add_argument("--cache-dir", type=str, default=str(SCRIPT_DIR / "cache"))
    p.add_argument("--train-years", type=int, nargs=2, default=[1979, 2015])
    p.add_argument("--val-years", type=int, nargs=2, default=[2016, 2017])
    p.add_argument("--variable", type=str, choices=VAR_NAMES, default="z500")
    p.add_argument("--lead-hours", type=int, nargs="+", default=[48, 144, 240])
    p.add_argument(
        "--sample-index",
        type=int,
        default=3,
        help="Forecast-origin time index in validation data.",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Skip checkpoints that cannot be loaded/predicted.",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def discover_checkpoints(args) -> list[Path]:
    if args.checkpoints:
        paths = [Path(p).expanduser().resolve() for p in args.checkpoints]
    else:
        ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
        patterns = args.checkpoint_glob or ["*.pt"]
        seen = set()
        paths = []
        for pattern in patterns:
            for path in sorted(ckpt_dir.glob(pattern)):
                if path.is_file() and path not in seen:
                    paths.append(path)
                    seen.add(path)
    if args.max_checkpoints and args.max_checkpoints > 0:
        paths = paths[: args.max_checkpoints]
    if not paths:
        raise FileNotFoundError("No checkpoints found. Check --checkpoint-dir / --checkpoints.")
    return paths


def build_koopman(args: SimpleNamespace, in_ch: int, out_ch: int, device: torch.device) -> torch.nn.Module:
    if args.koopman_input_size % 32 != 0:
        raise ValueError(f"koopman_input_size must be divisible by 32, got {args.koopman_input_size}")
    latent_hw = args.koopman_input_size // 32
    core = kwm.KoopmanAE2DTransFlexible(
        in_channel=in_ch,
        out_channel=out_ch,
        dim=args.koopman_dim,
        num_blocks=tuple(args.koopman_num_blocks),
        heads=tuple(args.koopman_heads),
        ffn_expansion_factor=args.koopman_ffn_expansion_factor,
        bias=False,
        LayerNorm_type="WithBias",
        steps=args.koopman_steps,
        steps_back=args.koopman_steps_back,
        grid_info=args.koopman_grid_info,
        latent_hw=latent_hw,
    ).to(device)
    return kwm.KoopmanWeatherAdapter(
        koopman_core=core,
        lon_downsample=args.lon_downsample,
        koopman_input_size=args.koopman_input_size,
    ).to(device)


def load_koopman_weights(model: torch.nn.Module, state_dict: dict) -> None:
    keys = list(state_dict.keys())
    if any(k.startswith("koopman_core.") for k in keys):
        model.load_state_dict(state_dict, strict=True)
    else:
        model.koopman_core.load_state_dict(state_dict, strict=True)


def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def infer_base_var_names(ckpt: dict, ns: SimpleNamespace, n_base_vars: int, requested_var: str) -> list[str]:
    train_var_names = ckpt.get("train_var_names")
    if isinstance(train_var_names, (list, tuple)) and len(train_var_names) == n_base_vars:
        return list(train_var_names)
    eval_var = getattr(ns, "eval_var", None)
    if n_base_vars == 1 and eval_var in VAR_NAMES:
        return [eval_var]
    if n_base_vars == 1:
        return [requested_var]
    return list(VAR_NAMES[:n_base_vars])


def load_data_for_vars(args, var_names: list[str], data_cache: dict):
    key = tuple(var_names)
    if key in data_cache:
        return data_cache[key]
    variables = [VARIABLES[VAR_NAMES.index(name)] for name in var_names]
    _, data_val, lat, lon = load_era5_data(
        train_years=tuple(args.train_years),
        val_years=tuple(args.val_years),
        cache_dir=args.cache_dir,
        variables=variables,
        var_names=var_names,
    )
    data_cache[key] = (data_val, lat, lon)
    return data_cache[key]


def build_model_from_checkpoint(ckpt: dict, device: torch.device):
    ns = SimpleNamespace(**ckpt["config"])
    framing = ckpt.get("framing") or {}
    if not framing.get("recurrent_rollout"):
        raise ValueError("This plot script only loads the paper Koopman / QIML rollout checkpoints.")
    n_vars = int(framing["n_vars_per_timestep"])
    model = build_koopman(ns, in_ch=n_vars, out_ch=n_vars, device=device)
    load_koopman_weights(model, ckpt["model_state_dict"])
    model.eval()
    return model, ns, framing


def predict_ar_rollout(model, data_val, mean_x, std_x, ns, framing, lead_h: int, sample_idx: int, ch: int, device):
    train_lead_h = int(ns.train_lead_hours)
    if lead_h % train_lead_h != 0:
        raise ValueError(f"{lead_h}h is not divisible by train_lead_hours={train_lead_h}.")
    n_vars = int(framing["n_vars_per_timestep"])
    mean_step = mean_x[:, -n_vars:, :, :]
    std_step = std_x[:, -n_vars:, :, :]

    x_np, y_np = build_pairs(data_val, lead_h)
    if sample_idx >= len(x_np):
        raise IndexError(f"sample_index={sample_idx} is out of range for lead {lead_h}h.")
    x = torch.from_numpy(np.ascontiguousarray(x_np[sample_idx : sample_idx + 1])).to(device)
    cur = (x - mean_step) / std_step
    with torch.no_grad():
        for _ in range(lead_h // train_lead_h):
            cur = model(cur)
    pred = denorm(cur, mean_step, std_step)[0, ch].detach().cpu().numpy()
    truth = y_np[sample_idx, ch].astype(np.float32, copy=False)
    return truth, pred


def predict_checkpoint(path: Path, args, lead_h: int, device: torch.device, data_cache: dict):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model, ns, framing = build_model_from_checkpoint(ckpt, device)
    mean_x = ckpt.get("normalization_mean_x", ckpt.get("normalization_mean")).to(device)
    std_x = ckpt.get("normalization_std_x", ckpt.get("normalization_std")).to(device)
    n_vars = int(framing["n_vars_per_timestep"])
    var_names = infer_base_var_names(ckpt, ns, n_vars, args.variable)
    if args.variable not in var_names:
        raise ValueError(f"{args.variable!r} is not in checkpoint variables {var_names}")
    ch = var_names.index(args.variable)
    data_val, lat, lon = load_data_for_vars(args, var_names, data_cache)
    truth, pred = predict_ar_rollout(
        model, data_val, mean_x, std_x, ns, framing, lead_h, args.sample_index, ch, device
    )
    del model, ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return truth, pred, lat, lon


def label_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "with_qprior" in stem or "qprior" in stem:
        return "QIML"
    if "no_qprior" in stem or "noq" in stem:
        return "Koopman"
    return path.stem


def main():
    args = parse_args()
    for lead in args.lead_hours:
        if lead % HOURS_PER_STEP != 0:
            raise ValueError(f"lead-hours must be multiples of {HOURS_PER_STEP}; got {lead}")

    device = resolve_device(args.device)
    ckpt_paths = discover_checkpoints(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoints selected: {len(ckpt_paths)}")
    for path in ckpt_paths:
        print(f"  - {path.name}")

    data_cache = {}
    for lead_h in args.lead_hours:
        predictions = {}
        truth_ref = None
        lat_ref = lon_ref = None
        skipped = []

        print(f"\n=== Lead {lead_h}h ===")
        for path in ckpt_paths:
            try:
                truth, pred, lat, lon = predict_checkpoint(path, args, lead_h, device, data_cache)
                label = label_from_path(path)
                predictions[label] = pred[np.newaxis, ...]
                if truth_ref is None:
                    truth_ref = truth[np.newaxis, ...]
                    lat_ref, lon_ref = lat, lon
                print(f"  OK   {path.name} -> {label}")
            except Exception as exc:
                msg = f"{path.name}: {type(exc).__name__}: {exc}"
                skipped.append(msg)
                print(f"  SKIP {msg}")
                if not args.continue_on_error:
                    raise

        if not predictions:
            print(f"No valid predictions for lead {lead_h}h; skipping plot.")
            continue

        metric_text = None
        if skipped:
            metric_text = f"Skipped {len(skipped)} checkpoint(s); see terminal output."
        items = list(predictions.items())
        chunk_size = len(items) if args.models_per_figure <= 0 else int(args.models_per_figure)
        n_parts = int(np.ceil(len(items) / chunk_size))
        for part_idx in range(n_parts):
            chunk = dict(items[part_idx * chunk_size : (part_idx + 1) * chunk_size])
            part_suffix = "" if n_parts == 1 else f"_part{part_idx + 1:02d}"
            out_path = output_dir / (
                f"{args.variable}_sample{args.sample_index}_{lead_h}h_world_map_compare{part_suffix}.png"
            )
            plot_world_map_comparison(
                truth=truth_ref,
                predictions=chunk,
                lat=lat_ref,
                lon=lon_ref,
                var_names=[args.variable],
                model_label=f"{args.variable.upper()} checkpoint comparison"
                + (f" ({part_idx + 1}/{n_parts})" if n_parts > 1 else ""),
                lead_hours=lead_h,
                sample_idx=args.sample_index,
                save_path=str(out_path),
                metric_text=metric_text,
            )

    print(f"\nDone. Plots are in: {output_dir}")


if __name__ == "__main__":
    main()
