"""Train the 1-D Z500 Q-Prior used by the ERA5 QIML forecast."""
import numpy as np
import torch
import torch.nn as nn
import cudaq
import matplotlib.pyplot as plt
import time
import os
import argparse
import sys
from pathlib import Path

# -------------------------------
# 0. Parse command line arguments
# -------------------------------
parser = argparse.ArgumentParser(
    description="Train the 1-D ERA5 Q-Prior used by QIML."
)
parser.add_argument("--n_qubits", type=int, default=10,
                    help="Number of qubits (default: 10, i.e. 2^10 bins)")
parser.add_argument("--epochs", type=int, default=500,
                    help="Number of training epochs (default: 500)")
parser.add_argument("--lr", type=float, default=0.002,
                    help="Learning rate for gradient/SPSA optimizers (default: 0.002)")
parser.add_argument("--optimizer", type=str, default="lbfgs",
                    choices=["gradient", "spsa", "lbfgs"],
                    help="Optimization method (default: lbfgs)")
parser.add_argument("--spsa_c", type=float, default=0.01, help="SPSA perturbation magnitude")
parser.add_argument("--spsa_a", type=float, default=0.01, help="SPSA step size scale")
parser.add_argument("--spsa_groups", type=int, default=3, help="G-SPSA parameter groups")
parser.add_argument("--lbfgs_eps", type=float, default=1e-5, help="L-BFGS finite-diff epsilon")
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
default_data_dir = project_dir / "era5-5.625deg" / "geopotential_500"
parser.add_argument("--data-dir", type=str, default=str(default_data_dir),
                    help="Path to ERA5 z500 NetCDF files (local_nc mode)")
parser.add_argument("--years", type=int, nargs="+", default=[2015, 2016, 2017],
                    help="Years to load, e.g. --years 2015 2016 2017")
parser.add_argument("--data-source", type=str, default="wb2_cache",
                    choices=["local_nc", "wb2_cache"],
                    help="local_nc: read 5.625deg NetCDF files; wb2_cache: stream WB2 via data.py and cache (default)")
parser.add_argument("--cache-dir", type=str, default=str(project_dir / "cache"),
                    help="Cache dir used when --data-source wb2_cache")
parser.add_argument("--model-dir", type=str, default=str(project_dir / "models" / "model_climate_cudaq"),
                    help="Directory to save trained QPrior model")
parser.add_argument("--results-dir", type=str, default=str(project_dir / "results" / "qprior"),
                    help="Directory to save QPrior artifacts (pdfs, plots, metadata)")
parser.add_argument("--lead-steps", type=int, default=24,
                    help="Lead steps used when building training pairs (default: 24)")
parser.add_argument("--train-frac", type=float, default=0.8,
                    help="Chronological train split fraction")
parser.add_argument("--prior-source", type=str, default="x_train",
                    choices=["raw", "x_train"],
                    help="Use the raw field or the training-pair input distribution")
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))
from data import VAR_NAMES, VARIABLES
parser.add_argument("--variable", type=str, default="z500", choices=VAR_NAMES,
                    help="Variable to build scalar climate prior for (e.g. z500, vo500).")
parser.add_argument(
    "--bin-layout",
    type=str,
    default="auto",
    choices=["auto", "data_minmax_linear", "symmetric_tail_dense"],
    help="Histogram bin placement: auto uses tail-dense symmetric bins for vo500, else data min/max linear.",
)
parser.add_argument(
    "--bin-abs-min",
    type=float,
    default=1e-7,
    help="For symmetric_tail_dense: smallest |zeta| edge scale (s^-1) on the positive half-axis.",
)
parser.add_argument(
    "--bin-abs-max",
    type=float,
    default=5e-4,
    help="For symmetric_tail_dense: largest |zeta| edge (s^-1); default 5e-4 and expanded to data 99.9pct tail if needed.",
)
parser.add_argument(
    "--bin-tail-steepness",
    type=float,
    default=4.0,
    help="For symmetric_tail_dense: exp-steepness k>0 packs more bin edges toward larger |x| (tail emphasis).",
)
parser.add_argument(
    "--prior-standardize",
    action="store_true",
    help="If set, histogram / bins are built on z-scored samples; npz stores standardize_mean/std so Koopman can match in the same space (similar λ scale across variables).",
)
parser.add_argument(
    "--zscore-bin-half-width",
    type=float,
    default=6.0,
    help="With --prior-standardize, linear bins on [-W, W] in σ units (total range 2W).",
)
args = parser.parse_args()

# -------------------------------
# 1. Load z500 data
# -------------------------------
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
from era5_baseline_dataset import build_pairs, load_era5_z500

if args.data_source == "local_nc":
    if args.variable != "z500":
        raise ValueError("local_nc mode currently supports only --variable z500. Use --data-source wb2_cache for other vars.")
    print(f"Loading {args.variable} from {args.data_dir}, years={args.years}")
    values, times = load_era5_z500(args.data_dir, args.years)
else:
    # WeatherBench 2 loader with local cache.
    from data import load_era5_data

    train_start, train_end = min(args.years), max(args.years)
    var_idx = VAR_NAMES.index(args.variable)
    var_spec = VARIABLES[var_idx]
    print(
        f"Loading {args.variable} from WB2 with cache_dir={args.cache_dir}, "
        f"train_years=({train_start}, {train_end})"
    )
    data_train, _, _, _ = load_era5_data(
        train_years=(train_start, train_end),
        val_years=(train_start, train_end),
        cache_dir=args.cache_dir,
        variables=[var_spec],
        var_names=[args.variable],
    )
    values = data_train[:, 0, :, :]  # (T, H, W)
    times = np.arange(values.shape[0], dtype=np.int64)

# Align prior statistics with the training split when requested.
if args.prior_source == "x_train":
    x_np, _ = build_pairs(values, args.lead_steps)
    n_train = int(len(x_np) * args.train_frac)
    values_for_prior = x_np[:n_train, 0, :, :]
    source_desc = (
        f"x_train (pairs, lead_steps={args.lead_steps}, train_frac={args.train_frac})"
    )
else:
    values_for_prior = values
    source_desc = f"raw {args.variable} field"

# Flatten to 1D: (..., lat, lon) -> (N,)
values_flat = values_for_prior.ravel()
# Drop NaNs if any
values_flat = values_flat[~np.isnan(values_flat)]
global_min = float(np.min(values_flat))
global_max = float(np.max(values_flat))
print(
    f"Variable {args.variable}: source={source_desc}, raw_shape={values.shape}, "
    f"timesteps={len(times)}, range [{global_min}, {global_max}], N = {len(values_flat)}"
)

n_qubits = args.n_qubits
num_bins = 2 ** n_qubits


def _symmetric_tail_dense_edges(num_bins: int, abs_min: float, abs_max: float, k: float) -> np.ndarray:
    """Symmetric bins on [-abs_max, abs_max]; exp-spaced positive edges emphasize the tails."""
    if num_bins % 2 != 0:
        raise ValueError(f"symmetric_tail_dense needs even num_bins, got {num_bins}")
    if abs_min <= 0 or abs_max <= abs_min or k <= 0:
        raise ValueError("Need abs_min>0, abs_max>abs_min, tail k>0")
    n_half = num_bins // 2
    t = np.linspace(0.0, 1.0, n_half + 1, dtype=np.float64)
    pos = abs_max * (np.exp(k * t) - 1.0) / (np.exp(k) - 1.0)
    pos[0] = abs_min
    pos[-1] = abs_max
    neg = -pos[::-1]
    bin_edges = np.concatenate([neg[:-1], pos])
    return bin_edges


bin_layout = args.bin_layout
if bin_layout == "auto":
    bin_layout = "symmetric_tail_dense" if args.variable == "vo500" else "data_minmax_linear"

prior_standardize = bool(args.prior_standardize)
standardize_mean = standardize_std = None
values_for_hist = values_flat.astype(np.float64, copy=False)
value_transform_tag = "identity"

if prior_standardize:
    standardize_mean = float(np.mean(values_for_hist))
    standardize_std = float(np.std(values_for_hist))
    if standardize_std <= 0:
        standardize_std = 1.0
    values_for_hist = (values_for_hist - standardize_mean) / standardize_std
    value_transform_tag = "standardize"
    w = float(args.zscore_bin_half_width)
    bin_edges = np.linspace(-w, w, num_bins + 1)
    print(
        f"prior_standardize: mean={standardize_mean:.6g}, std={standardize_std:.6g}, "
        f"bins linear on [-{w}, {w}] (z-score space)"
    )
elif bin_layout == "data_minmax_linear":
    gmin = float(np.min(values_for_hist))
    gmax = float(np.max(values_for_hist))
    bin_edges = np.linspace(gmin, gmax, num_bins + 1)
    print(f"bin_layout=data_minmax_linear over data range [{gmin}, {gmax}]")
elif bin_layout == "symmetric_tail_dense":
    amax = float(args.bin_abs_max)
    amin = float(args.bin_abs_min)
    k = float(args.bin_tail_steepness)
    data_p99 = float(np.percentile(np.abs(values_for_hist), 99.9))
    amax = max(amax, data_p99 * 1.05)
    print(
        f"bin_layout=symmetric_tail_dense: abs in [{amin:.3e}, {amax:.3e}] s^-1, k={k}, "
        f"(99.9% |data|={data_p99:.3e})"
    )
    bin_edges = _symmetric_tail_dense_edges(num_bins, amin, amax, k)
else:
    raise ValueError(f"Unknown bin_layout {bin_layout}")

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# -------------------------------
# 2. CUDA-Q kernel (same as qprior_wf)
# -------------------------------
try:
    cudaq.set_target("nvidia")
    print("Using NVIDIA GPU backend")
except Exception as e:
    print(f"NVIDIA backend failed, using default: {e}")
    cudaq.set_target("default")


def create_QPrior_kernel(num_qubits, num_layers):
    @cudaq.kernel
    def QPrior_circuit_kernel(params: list[float]):
        q = cudaq.qvector(num_qubits)
        param_idx = 0
        for l in range(num_layers):
            for i in range(num_qubits):
                ry(params[param_idx], q[i])
                param_idx += 1
            for i in range(num_qubits):
                rz(params[param_idx], q[i])
                param_idx += 1
            for i in range(num_qubits):
                rx(params[param_idx], q[i])
                param_idx += 1
            for i in range(num_qubits - 1):
                x.ctrl(q[i], q[i + 1])
    return QPrior_circuit_kernel


num_layers = 8
QPrior_circuit_kernel = create_QPrior_kernel(n_qubits, num_layers)
print(f"Created QPrior kernel: {n_qubits} qubits, {num_layers} layers")


def get_probs_from_params(kernel_func, params_list):
    state = cudaq.get_state(kernel_func, params_list)
    return np.abs(state) ** 2


def mmd_loss_np(target, generated, smoothness_weight=0.01):
    eps = 1e-10
    generated_safe = np.abs(generated) + eps
    generated_safe = generated_safe / generated_safe.sum()
    target_safe = np.abs(target) + eps
    target_safe = target_safe / target_safe.sum()
    # Keep MMD definition aligned with qcbm_tcf_final.py:
    # simplified direct form via L2 distance between probability vectors.
    mmd = np.sum((generated_safe - target_safe) ** 2)
    diff = generated_safe[1:] - generated_safe[:-1]
    smooth_penalty = smoothness_weight * np.mean(diff ** 2)
    return mmd + smooth_penalty


class CUDAQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params_tensor, kernel_func):
        params_np = params_tensor.detach().cpu().numpy().tolist()
        state = cudaq.get_state(kernel_func, params_np)
        probs = np.abs(state) ** 2
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=params_tensor.device)
        ctx.save_for_backward(params_tensor)
        ctx.kernel_func = kernel_func
        return probs_tensor

    @staticmethod
    def backward(ctx, grad_output):
        params_tensor, = ctx.saved_tensors
        kernel_func = ctx.kernel_func
        params_np = params_tensor.detach().cpu().numpy()
        grad_input = torch.zeros_like(params_tensor)
        eps = 1e-4
        for i in range(len(params_np)):
            params_plus = params_np.copy()
            params_plus[i] += eps
            state_plus = cudaq.get_state(kernel_func, params_plus.tolist())
            probs_plus = np.abs(state_plus) ** 2
            params_minus = params_np.copy()
            params_minus[i] -= eps
            state_minus = cudaq.get_state(kernel_func, params_minus.tolist())
            probs_minus = np.abs(state_minus) ** 2
            grad_i = (probs_plus - probs_minus) / (2 * eps)
            grad_input[i] = torch.tensor(
                (grad_output.cpu().numpy() * grad_i).sum(),
                dtype=torch.float32, device=params_tensor.device
            )
        return grad_input, None


def QPrior_circuit(params_tensor, kernel_func):
    return CUDAQFunction.apply(params_tensor, kernel_func)


class QPrior(nn.Module):
    def __init__(self, kernel_func):
        super(QPrior, self).__init__()
        self.params = nn.Parameter(torch.randn(num_layers * 3 * n_qubits))
        self.kernel_func = kernel_func

    def forward(self):
        return QPrior_circuit(self.params, self.kernel_func)


def train_QPrior_on_target(
    target_pdf,
    num_epochs=200,
    lr=0.002,
    optimizer_type="gradient",
    spsa_c=0.01,
    spsa_a=0.01,
    spsa_groups=3,
    lbfgs_eps=1e-5,
):
    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device_torch}, optimizer: {optimizer_type}")
    target_pdf_tensor = torch.tensor(target_pdf, dtype=torch.float32, device=device_torch)
    model = QPrior(QPrior_circuit_kernel).to(device_torch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    def mmd_loss(target, generated, smoothness_weight=0.01):
        eps = 1e-10
        generated_safe = torch.abs(generated) + eps
        generated_safe = generated_safe / generated_safe.sum()
        target_safe = torch.abs(target) + eps
        target_safe = target_safe / target_safe.sum()
        mmd = torch.sum((generated_safe - target_safe) ** 2)
        diff = generated_safe[1:] - generated_safe[:-1]
        smooth_penalty = (diff ** 2).mean()
        return mmd + smoothness_weight * smooth_penalty

    n_params = model.params.numel()

    if optimizer_type == "lbfgs":
        try:
            from scipy.optimize import minimize
        except Exception as e:
            raise ImportError(
                "L-BFGS optimizer requires scipy with a compatible libstdc++ runtime. "
                "Try setting LD_LIBRARY_PATH to your conda env lib, or use --optimizer gradient."
            ) from e
        x0 = model.params.detach().cpu().numpy().astype(np.float64).copy()
        kernel_func = QPrior_circuit_kernel
        eps_fd = lbfgs_eps

        def objective_and_grad(x):
            probs = get_probs_from_params(kernel_func, x.tolist())
            probs = probs / (probs.sum() or 1e-12)
            loss = mmd_loss_np(target_pdf, probs)
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps_fd
                x_minus = x.copy()
                x_minus[i] -= eps_fd
                p_plus = get_probs_from_params(kernel_func, x_plus.tolist())
                p_minus = get_probs_from_params(kernel_func, x_minus.tolist())
                p_plus = p_plus / (p_plus.sum() or 1e-12)
                p_minus = p_minus / (p_minus.sum() or 1e-12)
                grad[i] = (mmd_loss_np(target_pdf, p_plus) - mmd_loss_np(target_pdf, p_minus)) / (2.0 * eps_fd)
            return loss, grad

        n_iter = [0]

        def callback(xk):
            n_iter[0] += 1
            if n_iter[0] % 50 == 0 or n_iter[0] == 1:
                probs = get_probs_from_params(kernel_func, xk.tolist())
                probs = probs / (probs.sum() or 1e-12)
                loss = mmd_loss_np(target_pdf, probs)
                print(f"Epoch {n_iter[0]}: Loss = {loss}")

        result = minimize(
            objective_and_grad, x0, method="L-BFGS-B", jac=True,
            callback=callback, options={"maxiter": num_epochs, "disp": False}
        )
        probs_final = get_probs_from_params(kernel_func, result.x.tolist())
        probs_final = probs_final / (probs_final.sum() or 1e-12)
        learned_pdf = probs_final
        model.params.data = torch.tensor(result.x.astype(np.float32), dtype=torch.float32, device=device_torch)
    else:
        if optimizer_type == "spsa":
            num_groups = max(1, min(spsa_groups, n_params))
            group_indices = np.array_split(np.arange(n_params), num_groups)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            if optimizer_type == "spsa":
                c_k = spsa_c / ((epoch + 1) ** 0.101)
                params_np = model.params.detach().cpu().numpy()
                grad_estimate = np.zeros(n_params, dtype=np.float32)
                loss_sum = 0.0
                n_meas = 0
                for g_idx in range(len(group_indices)):
                    inds = group_indices[g_idx]
                    delta = np.zeros(n_params, dtype=np.float64)
                    delta[inds] = np.random.choice([-1.0, 1.0], size=len(inds))
                    params_plus = params_np + c_k * delta
                    params_minus = params_np - c_k * delta
                    probs_plus = get_probs_from_params(QPrior_circuit_kernel, params_plus.tolist())
                    probs_minus = get_probs_from_params(QPrior_circuit_kernel, params_minus.tolist())
                    probs_plus = probs_plus / (probs_plus.sum() or 1e-12)
                    probs_minus = probs_minus / (probs_minus.sum() or 1e-12)
                    p_plus = torch.tensor(probs_plus, dtype=torch.float32, device=device_torch)
                    p_minus = torch.tensor(probs_minus, dtype=torch.float32, device=device_torch)
                    L_plus = mmd_loss(target_pdf_tensor, p_plus)
                    L_minus = mmd_loss(target_pdf_tensor, p_minus)
                    grad_estimate[inds] = (L_plus.item() - L_minus.item()) / (2.0 * c_k) * delta[inds].astype(np.float32)
                    loss_sum += L_plus.item() + L_minus.item()
                    n_meas += 2
                model.params.grad = torch.tensor(grad_estimate, dtype=torch.float32, device=device_torch)
                optimizer.step()
                loss = torch.tensor(loss_sum / n_meas, dtype=torch.float32, device=device_torch)
            else:
                probs = model()
                loss = mmd_loss(target_pdf_tensor, probs)
                loss.backward()
                optimizer.step()
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
        learned_pdf = model().detach().cpu().numpy()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"qprior_climate_{args.variable}_{n_qubits}q.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    learned_pdf = learned_pdf / (learned_pdf.sum() or 1e-12)
    return learned_pdf


# -------------------------------
# 3. Build target PDF from selected variable (single distribution over all loaded data)
# -------------------------------
hist, _ = np.histogram(values_for_hist, bins=bin_edges)
if hist.sum() == 0:
    raise ValueError("No data in bins; check variable and range.")
target_pdf = hist.astype(np.float64) / hist.sum()
print(f"Target PDF: {num_bins} bins, sum = {target_pdf.sum():.6f}")

# -------------------------------
# 4. Train QPrior
# -------------------------------
print(f"Training QPrior for {args.variable} ({n_qubits} qubits, {args.epochs} epochs, {args.optimizer})")
t0 = time.time()
learned_pdf = train_QPrior_on_target(
    target_pdf,
    num_epochs=args.epochs,
    lr=args.lr,
    optimizer_type=args.optimizer,
    spsa_c=args.spsa_c,
    spsa_a=args.spsa_a,
    spsa_groups=args.spsa_groups,
    lbfgs_eps=args.lbfgs_eps,
)
print(f"Training time: {time.time() - t0:.2f} s")

# -------------------------------
# 5. Optional smoothing and plot
# -------------------------------
def smooth_pdf(pdf, window=5):
    if window < 2 or len(pdf) < window:
        return pdf
    pad = window // 2
    padded = np.pad(pdf, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed = smoothed / smoothed.sum() * pdf.sum()
    return smoothed


learned_smooth = smooth_pdf(learned_pdf, window=7)
results_dir = Path(args.results_dir)
results_dir.mkdir(parents=True, exist_ok=True)

# Persist PDFs for downstream integration.
pdf_path = results_dir / f"qprior_pdf_{args.variable}_{n_qubits}q_{args.optimizer}.npz"
np.savez(
    pdf_path,
    bin_centers=bin_centers,
    target_pdf=target_pdf,
    learned_pdf=learned_pdf,
    learned_smooth=learned_smooth,
    value_transform=value_transform_tag,
    standardize_mean=(standardize_mean if standardize_mean is not None else np.nan),
    standardize_std=(standardize_std if standardize_std is not None else np.nan),
    bin_layout=bin_layout,
    variable=args.variable,
)
print(f"PDF artifacts saved to {pdf_path}")

plt.figure(figsize=(10, 5))
plt.plot(bin_centers, target_pdf, alpha=0.7, label=f"Target ({args.variable})")
plt.plot(bin_centers, learned_smooth, label="QPrior learned (smoothed)")
plt.xlabel(args.variable)
plt.ylabel("Probability density")
plt.title(f"Climate prior: {args.variable} ({n_qubits} qubits, {args.optimizer})")
plt.legend()
out_png = results_dir / f"QPrior_climate_{args.variable}_{n_qubits}_{args.optimizer}.png"
plt.savefig(out_png)
plt.show()
print(f"Plot saved to {out_png}")
