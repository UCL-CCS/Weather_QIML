# Weather QIML

Code for the ERA5 case study in:

**[Quantum Statistical Memory Advantage Reveals Predictive Structure in Chaotic Invariant Measures](https://arxiv.org/abs/2606.13422)**  
Maida Wang, Xiao Xue, Minh Chung, Peter V. Coveney (2026). [arXiv:2606.13422](https://arxiv.org/abs/2606.13422)

---

This repository contains one forecast comparison: **QIML** (Koopman rollout + Q-Prior) against the **matched Koopman baseline** (same network, same data, no Q-Prior). Both are trained with `weather/train.py`.

The paper’s remaining case studies (turbulent channel flow, hardware Bell read-out, planetary-wave phase sector) are described in the manuscript and are not part of this snapshot. The theoretical results do not involve this code.

## Dependencies

Python ≥ 3.10 with:

- [CUDA-Q](https://nvidia.github.io/cuda-quantum/) — Q-Prior generator
- PyTorch — Koopman trainer
- NumPy, SciPy, matplotlib
- xarray, zarr, gcsfs, netCDF4 — WeatherBench 2 / ERA5 I/O
- cartopy — world-map figures (optional)

## Data

ERA5 reanalysis via [WeatherBench 2](https://weatherbench2.readthedocs.io/), originally from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels):

- Variable: 500 hPa geopotential ($Z_{500}$)
- Grid: 1.5° equiangular with poles (240 × 121)
- Cadence: 6-hourly
- Training years: 1979–2015
- Held-out evaluation: 2016–2017
- Default training lead: 24 h; evaluation leads up to 20 days

The loaders in `weather/data.py` read the public WeatherBench 2 Zarr stores.

## Hardware

Training and evaluation in the paper were run on NVIDIA H100 GPUs at the [LRZ BEAST](https://www.lrz.de/services/compute/special_systems/beast/) cluster. A single H100 is sufficient to reproduce the ERA5 forecasts in this repository.

## Layout

```
lib/            Koopman / Restormer backbone
weather/
  train.py      QIML (`--qprior-on`) or matched Koopman baseline
  koopman.py    shared model / CLI / Q-Prior loss
  qprior/       1-D climate Q-Prior ($Z_{500}$) generator
```

Run weather scripts from `weather/`. Checkpoints, caches, and result JSON are written locally under `weather/checkpoints`, `weather/cache`, and `weather/results` (gitignored).

## Usage

Train the $Z_{500}$ Q-Prior (writes `results/qprior/qprior_pdf_z500_10q_lbfgs.npz`):

```bash
cd weather
python qprior/qprior_climate_Z500.py --variable z500
```

Train QIML:

```bash
python train.py --qprior-on --eval-var z500 --experiment-tag z500_qprior
```

Train the matched Koopman baseline (same script, Q-Prior off):

```bash
python train.py --eval-var z500 --experiment-tag z500_koopman
```

Skill curves and world maps, after evaluation JSON / checkpoints exist:

```bash
python plot_acc_compare_koopman_qprior.py \
  --json-a results/baseline_results_koopman1x4ar_z500_no_qprior.json \
  --json-b results/baseline_results_koopman1x4ar_z500_with_qprior.json \
  --variable z500 --min-lead 24 --max-lead 240 --show-rmse

python plot_checkpoint_world_maps.py \
  --checkpoints checkpoints/koopman1x4ar_z500_baseline_with_qprior.pt \
                checkpoints/koopman1x4ar_z500_baseline_no_qprior.pt \
  --lead-hours 48 144 240 --variable z500 --sample-index 3
```

## Citation

```bibtex
@article{wang2026qiml,
  title   = {Quantum Statistical Memory Advantage Reveals Predictive Structure in Chaotic Invariant Measures},
  author  = {Wang, Maida and Xue, Xiao and Chung, Minh and Coveney, Peter V.},
  journal = {arXiv preprint arXiv:2606.13422},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.13422}
}
```

## License

MIT (see [`Weather_QIML_LICENSE`](Weather_QIML_LICENSE)).

## Contact

For questions about the code or paper, please contact Peter V. Coveney (`p.v.coveney@ucl.ac.uk`) or open an issue.
