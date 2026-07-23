# DeepBedMap

[![Paper](https://img.shields.io/badge/paper-The%20Cryosphere-2f6f9f)](https://doi.org/10.5194/tc-14-3687-2020)
[![Dataset](https://img.shields.io/badge/dataset-Zenodo-1682d4)](https://doi.org/10.5281/zenodo.4054246)
[![CI](https://github.com/weiji14/deepbedmap/actions/workflows/ci.yml/badge.svg)](https://github.com/weiji14/deepbedmap/actions/workflows/ci.yml)

DeepBedMap uses a super-resolution neural network to resolve Antarctic bed
topography beyond BEDMAP2. This repository now provides a project-local,
Windows-first PyTorch inference runtime while retaining the original research
notebooks, data catalogue, and published Chainer checkpoint for provenance.

![DeepBedMap DEM over Antarctica in EPSG:3031](https://user-images.githubusercontent.com/23487320/94385232-16dee180-01a1-11eb-83ce-8793709079ff.png)

> [!IMPORTANT]
> The modern runtime is for reproducible inference and checkpoint conversion.
> The historical notebooks remain available for reference, but the old
> Conda/Pipenv/Chainer/GMT workflow is not part of the supported runtime.

## Windows quick start

Requirements: x86-64 Windows, PowerShell, Git, and HTTPS access. A system Python,
Conda, CUDA Toolkit, GMT, and administrator rights are **not** required. An NVIDIA
GPU is optional.

```powershell
git clone https://github.com/weiji14/deepbedmap.git
cd deepbedmap
```

Then run three commands:

```powershell
# 1. Install uv, Python 3.12, and all runtime packages inside this repository.
.\setup.ps1

# 2. Download, verify, and convert the published v1.1 model weights.
.\run.ps1 fetch-weights

# 3. Run a real forward pass with the converted checkpoint.
.\run.ps1 smoke
```

`setup.ps1` selects CUDA when `nvidia-smi` reports a usable NVIDIA GPU;
otherwise it installs the CPU build. Everything is contained in `.tools/`,
`.venv/`, and `.cache/`, so it does not modify a global Python environment.

Check exactly what was installed:

```powershell
.\run.ps1 doctor
.\run.ps1 list-assets
```

If PowerShell blocks local scripts, use a process-scoped policy without changing
the machine policy:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## CPU and CUDA

Choose the runtime explicitly when needed:

```powershell
# Works on every supported Windows machine.
.\setup.ps1 -Accelerator cpu
.\run.ps1 smoke --device cpu

# NVIDIA GPU path; PyTorch carries its required CUDA runtime libraries.
.\setup.ps1 -Accelerator cuda
.\run.ps1 smoke --device cuda
```

`--device auto` is the default. It probes CUDA and falls back to CPU. Explicit
`--device cuda` fails clearly when the installed runtime or driver cannot use the
GPU. Use `doctor` to see the PyTorch build, CUDA availability, and selected device.

## Commands

```powershell
.\run.ps1 --help
.\run.ps1 doctor [--json]
.\run.ps1 fetch-weights [--offline] [--force]
.\run.ps1 fetch-dem [--offline] [--force]
.\run.ps1 list-assets [--json]
.\run.ps1 smoke [--device auto|cpu|cuda] [--random]
.\run.ps1 predict --help
```

Downloads are resumable, cached under `.cache/deepbedmap/`, and verified against
the release size and digest before use. `fetch-weights` also converts the public
Chainer `.npz` into a metadata-rich native PyTorch checkpoint; Chainer and CuPy
are not imported. The native v3 format validates its architecture metadata,
finite tensors, and a canonical state digest when loaded.

## Predict a local region

Prediction consumes five local, single-band rasters and one output extent. All
rasters must have a declared `EPSG:3031` CRS, square pixels, and coverage of the
requested region plus a 1 km model halo.

| Argument | Physical field | Model grid |
| --- | --- | ---: |
| `--bed` | BEDMAP2 bed elevation | 1,000 m |
| `--surface` | REMA surface elevation | 100 m |
| `--velocity-x` | Signed X ice velocity | 500 m |
| `--velocity-y` | Signed Y ice velocity | 500 m |
| `--accumulation` | Surface mass accumulation | 1,000 m |

Source rasters are reprojected/resampled onto those exact model grids. Velocity
components remain signed; do not supply speed magnitude in their place. Output
bounds are `LEFT BOTTOM RIGHT TOP` in metres in EPSG:3031, with width and height
both integer multiples of 1 km.

```powershell
.\run.ps1 predict `
  --bed D:\antarctica\bedmap2_bed.tif `
  --surface D:\antarctica\REMA_100m_dem.tif `
  --velocity-x D:\antarctica\velocity_x.tif `
  --velocity-y D:\antarctica\velocity_y.tif `
  --accumulation D:\antarctica\accumulation.tif `
  --bounds -500000 -1000000 -480000 -980000 `
  --output outputs\deepbedmap_tile.tif `
  --device auto
```

The output is a finite, float32 GeoTIFF in EPSG:3031 at 250 m resolution and does
not declare a numeric NoData sentinel. Its tags and JSON summary include the
actual checkpoint format, SHA256, and published/custom weight status. See
[data and prediction](docs/DATA_AND_PREDICTION.md) before running a scientific
region, especially when input rasters contain gaps.

## Scientific compatibility

The PyTorch generator preserves the published v1.1 architecture:

- 12 residual-in-residual dense blocks and residual scaling of `0.2`;
- 8,907,749 trainable parameters;
- lossless loading of all 384 arrays in the public Chainer checkpoint;
- an explicit Chainer-to-Torchvision deformable-convolution offset adapter; and
- fixed-input numerical regression against a Chainer 7 CPU reference.

The regression is tolerance-based because CPU convolution implementations are not
bit-for-bit identical across frameworks. A successful `smoke` proves that the
runtime, checkpoint, and device can execute; it does **not** by itself validate a
new scientific dataset, geographic extent, or full-continent product.

Details and the boundary between the new and historical stacks are documented in
[migration and troubleshooting](docs/MIGRATION_AND_TROUBLESHOOTING.md).

## Development and Linux

Add test tools on Windows and run the supported suite:

```powershell
.\setup.ps1 -Dev
.\run.ps1 test -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Linux is supported through standard `uv` commands (the convenience PowerShell
bootstrap is Windows-specific):

```bash
uv sync --frozen --extra cpu --dev
uv run --frozen --extra cpu python -m pytest
uv run --frozen --extra cpu deepbedmap smoke --device cpu --random
```

GitHub Actions cold-tests the PowerShell bootstrap on Windows and the locked CPU
environment on Linux with Python 3.11, 3.12, and 3.13. Every job converts the
published checkpoint and runs its numerical and GeoTIFF regressions. CI does not
install Conda, GMT, Chainer, CuPy, or a system CUDA toolkit.

## Repository map

- `src/deepbedmap/`: supported PyTorch model, raster I/O, assets, and CLI.
- `tests/`: cross-platform unit tests and optional checkpoint regression tests.
- `setup.ps1` / `run.ps1`: project-local Windows entry points.
- `data_list.yml`, `highres/`, `lowres/`, `misc/`: historical source catalogue.
- `*.ipynb`, paired root `*.py`, and legacy environment files: original research
  workflow, retained for auditability rather than used by the modern CLI.
- `paper/`: publication material and figures.

## Paper, data, and research record

- Paper: [Leong and Horgan (2020), *The Cryosphere*](https://doi.org/10.5194/tc-14-3687-2020)
- Published DeepBedMap DEM v1.1.0: [Zenodo](https://doi.org/10.5281/zenodo.4054246)
- Archived source release: [Zenodo](https://doi.org/10.5281/zenodo.3752613)
- Experiment record: [Comet](https://www.comet.ml/weiji14/deepbedmap/)
- [Poster](https://github.com/weiji14/deepbedmap/issues/133) and
  [ML4Polar presentation](https://hackmd.io/@weiji14/2022ML4Polar)

The complete historical input inventory, literature references, file URLs, and
digests remain in [`data_list.yml`](data_list.yml). The modern runtime downloads
only the published weights or final DEM on explicit request; it does not silently
download the multi-source training corpus.

## Citing

```bibtex
@Article{tc-14-3687-2020,
  AUTHOR = {Leong, W. J. and Horgan, H. J.},
  TITLE = {DeepBedMap: a deep neural network for resolving the bed topography of Antarctica},
  JOURNAL = {The Cryosphere},
  VOLUME = {14},
  YEAR = {2020},
  NUMBER = {11},
  PAGES = {3687--3705},
  URL = {https://tc.copernicus.org/articles/14/3687/2020/},
  DOI = {10.5194/tc-14-3687-2020}
}
```

## License

The code is distributed under the terms in [LICENSE.md](LICENSE.md). Input
datasets keep their own licences and citation requirements.
