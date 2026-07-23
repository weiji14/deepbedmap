# Migration and troubleshooting

## What changed

The supported inference path has moved from the original Python 3.7-era research
environment to Python 3.12, PyTorch, Rasterio, and `uv`.

| Historical path | Supported path |
| --- | --- |
| Conda plus Pipenv | Project-local `uv`, Python, and `.venv` |
| Chainer/CuPy model execution | PyTorch/Torchvision execution |
| GMT shell/data operations | Rasterio-based local tile preparation |
| Notebook side effects | Explicit `deepbedmap` CLI commands |
| Global CUDA assumptions | Probed CPU/CUDA choice with self-contained PyTorch wheels |

The public Chainer NPZ remains the source checkpoint. Conversion maps every tensor
into the PyTorch model, preserves the published residual scaling, remaps Chainer's
serialized repeated-block order, and adapts deformable-convolution offset layout
at runtime. A fixed-input Chainer CPU reference guards the numerical result.
The converted v3 checkpoint also carries a canonical state digest and a strict
architecture/CRS/resolution contract, so stale or modified native files fail
before inference rather than being mistaken for the published model.

The root notebooks and paired scripts document the 2020 research workflow. They
may still require obsolete dependencies, external datasets, Unix commands, API
credentials, or services. They are not imported by `src/deepbedmap/`, executed by
the Windows launchers, or tested in modern CI.

## Setup fails before creating `.venv`

Confirm HTTPS access to `github.com`, then retry. The bootstrap retains project
state and uses a pinned, checksum-verified `uv` archive. If a partial environment
is inconsistent, recreate only the repository-local virtual environment:

```powershell
.\setup.ps1 -Accelerator cpu -Recreate
```

`-Recreate` never targets a directory outside this repository. During an ordinary
CPU/CUDA switch, setup keeps the previous accelerator marker until the new
runtime passes a real model forward pass and attempts to restore the previous
runtime if installation fails. After a forced interruption, rerunning the same
setup command repairs the project-local environment from the lock file.

If script execution is blocked:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

## Weights are missing or a download was interrupted

Run:

```powershell
.\run.ps1 fetch-weights
.\run.ps1 list-assets
```

Downloads resume from `.part` files. Complete partials are verified before any
network request, and response bytes are capped at the declared asset size.
Existing assets are accepted only after size and digest verification; an invalid
completed destination is moved aside with a `.corrupt` suffix. To verify an
existing cache without network access:

```powershell
.\run.ps1 fetch-weights --offline
```

## CUDA was selected but cannot run

Start with evidence from:

```powershell
nvidia-smi
.\run.ps1 doctor
.\run.ps1 smoke --device cuda
```

An NVIDIA driver can report a GPU while still being too old or otherwise unable
to load the installed PyTorch runtime. Update the driver or switch cleanly to CPU:

```powershell
.\setup.ps1 -Accelerator cpu
.\run.ps1 smoke --device cpu
```

Installing the standalone CUDA Toolkit does not repair an incompatible PyTorch
wheel/driver combination; the wheel already supplies its runtime libraries.

## A raster is rejected

Common causes are a missing or non-EPSG:3031 CRS, rectangular source pixels,
insufficient halo coverage, or bounds that are not multiples of 1 km. Inspect the
source metadata in a GIS or with Rasterio, then correct the source rather than
renaming its CRS.

Surface gaps stop prediction by default. Other fields use explicit fill defaults;
see [data and prediction](DATA_AND_PREDICTION.md) before overriding them.

## A large region exhausts memory

Run the same command on adjacent bounds-aligned tiles. Keep every tile boundary on
the 1 km grid and retain the 1 km source halo. Mosaic the resulting 250 m GeoTIFFs
after checking overlaps and edge behaviour. CPU execution needs less specialized
hardware but does not remove host-memory limits.

## Get an actionable traceback

The CLI normally prints a concise error. Enable Python tracebacks for diagnosis:

```powershell
$env:DEEPBEDMAP_DEBUG = "1"
.\run.ps1 predict <arguments>
```

Also include `.\run.ps1 doctor --json`, the exact command, input raster metadata,
and the traceback when reporting a problem. Remove the variable afterward with:

```powershell
Remove-Item Env:DEEPBEDMAP_DEBUG
```
