# Data and prediction

This guide covers the supported local-raster inference path. It does not recreate
the historical data preparation or training pipeline.

## Published products versus model inputs

These are different workflows:

- `fetch-dem` downloads the already-published DeepBedMap v1.1.0 Antarctic DEM.
  Use it when you need the released product rather than a new prediction.
- `predict` runs the published generator weights over five local source rasters.
  Use it to reproduce or inspect a particular region.

```powershell
.\run.ps1 fetch-dem
.\run.ps1 list-assets
```

The final DEM download is about 473 MiB. The full source corpus is much larger and
is intentionally not fetched by setup.

## Input contract

Every source must be a readable raster whose first band contains the intended
physical field. The CLI checks for a declared EPSG:3031 CRS and square source
pixels, then uses bilinear reprojection onto exact model grids.

| Input | Channels | Target spacing | Typical historical source |
| --- | ---: | ---: | --- |
| Bed elevation | 1 | 1,000 m | BEDMAP2 bed |
| Surface elevation | 1 | 100 m | REMA 100 m DEM |
| Ice velocity | 2 | 500 m | MEaSUREs X and Y components |
| Accumulation | 1 | 1,000 m | Arthern/BEDMAP2 grid |

Keep velocity X and Y signed. The generator was conditioned on directional
components, not absolute speed. Units and vertical datums must match the source
products used by the study; the runtime cannot infer or repair inconsistent
metadata.

The source catalogue and citations are retained in
[`data_list.yml`](../data_list.yml), with concise generated tables in
[`lowres/README.md`](../lowres/README.md),
[`misc/README.md`](../misc/README.md), and
[`highres/README.md`](../highres/README.md).

## Bounds and halo

`--bounds LEFT BOTTOM RIGHT TOP` describes the output extent in EPSG:3031 metres.
Both dimensions must be positive integer multiples of 1,000 m. Internally the
reader expands that extent by 1 km on every side to provide the convolution halo.
All five inputs therefore need valid coverage beyond the output boundary.

For example, these bounds request a 20 km by 20 km output tile:

```text
-500000 -1000000 -480000 -980000
```

The result contains 80 by 80 pixels at 250 m.

## Missing data

Default fill behaviour is deliberately asymmetric:

| Field | Default outside-valid-data value |
| --- | ---: |
| Bed | -5,000 m |
| Surface | no default; prediction stops |
| Velocity X/Y | 0 |
| Accumulation | 0 |

Override a value only when it is scientifically justified:

```powershell
.\run.ps1 predict <required arguments> `
  --bed-fill -5000 `
  --surface-fill 0 `
  --velocity-fill 0 `
  --accumulation-fill 0
```

Using a convenient fill value can create edge artefacts or physically impossible
predictions. Prefer an extent with valid source coverage, particularly for REMA.

## Output contract

`predict` writes a single-band, float32, deflate-compressed GeoTIFF with:

- CRS `EPSG:3031`;
- 250 m square pixels;
- the exact requested bounds;
- finite samples only; and
- no numeric NoData sentinel;
- checkpoint format and SHA256 tags; and
- an explicit `v1.1.0` or `custom` weight-source tag.

The file is written beside its destination under a unique temporary name and
atomically replaced only after Rasterio closes it, so an interrupted prediction
does not leave a half-written file at the requested output path.

Inspect the JSON summary printed by the command and validate the result in a GIS
before treating it as a scientific product. Runtime smoke, file creation, and
visual plausibility are separate checks from regional or full-continent accuracy.
