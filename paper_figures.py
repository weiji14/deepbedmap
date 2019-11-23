# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.2.0
#   kernelspec:
#     display_name: deepbedmap
#     language: python
#     name: deepbedmap
# ---

# %% [markdown]
# # **DeepBedMap paper figures**
#
# Code used to produce each figure in the DeepBedMap paper.
# Uses [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) for architecture diagram,
# and [PyGMT](https://github.com/GenericMappingTools/pygmt/) for map plots.

# %%
import geopandas as gpd
import pandas as pd
import pygmt as gmt
import rasterio

# %% [markdown]
# # **Results**

# %% [markdown]
# ## **DeepBedMap_DEM Topography**

# %% [markdown]
# ### **Figure 2: 2D Digital Elevation Model over Antarctica**
# Also includes the following layers:
# - Grounding line
# - Pine Island Glacier extent in Figure 3
# - Thwaites Glacier extent in Figure 5
# - Bounding boxes of training tile regions

# %%
region_pineisle = rasterio.coords.BoundingBox(
    left=-1631500.0, bottom=-259000.0, right=-1536500.0, top=-95000.0
)
region_thwaites = rasterio.coords.BoundingBox(
    left=-1550000.0, bottom=-550000.0, right=-1250000.0, top=-300000.0
)
training_tiles = gpd.read_file("model/train/tiles_3031.geojson")

# %%
fig = gmt.Figure()
# Plot DeepBedMap Digital Elevation Model (DEM)
gmt.makecpt(cmap="oleron", series=[-2000, 4500])
fig.grdimage(
    grid="model/deepbedmap3_big_int16.tif",
    # grid="@BEDMAP_elevation.nc",
    region=[-2700000, 2800000, -2200000, 2300000],
    projection="x1:30000000",
    cmap=True,
    Q=True,
)
# Plot Antactic grounding line
fig.coast(
    region="model/deepbedmap3_big_int16.tif",
    projection="s0/-90/-71/1:30000000",
    area_thresh="+ag",
    resolution="i",
    shorelines="0.25p",
    # frame="ag",
)
# Plot Pine Island and Thwaites Glacier study regions, and Training Tile areas
fig.plot(
    data=pd.DataFrame([region_pineisle]).values,
    region=[-2700000, 2800000, -2200000, 2300000],
    projection="x1:30000000",
    style="r+s",
    pen="1.5p,purple2",
    label='"Pine Island Glacier"',
)
fig.plot(
    data=pd.DataFrame([region_thwaites]).values,
    style="r+s",
    pen="1.5p,yellow2",
    label='"Thwaites Glacier"',
)
fig.plot(
    data=training_tiles.bounds.values,
    style="r+s",
    pen="1p,orange2",
    label='"Training Regions"',
)
# Plot map elements (colorbar, legend, frame)
fig.colorbar(
    position="jBL+jBL+o2.0c/0.5c+w2.4c/0.3c+m",
    box="+gwhite+p0.5p",
    frame=["af", 'x+l"Elevation"', "y+lkm"],
    cmap="+Uk",  # kilo-units, i.e. divide by 1000
    S=True,  # no lines inside color scalebar
)
fig.legend(position="jBL+jBL+o2.7c/0.2c", box="+gwhite+p0.5p")
fig.basemap(
    region=[-2700, 2800, -2200, 2300],
    projection="x1:30000",
    Bx='af+l"Polar Stereographic X (km)"',
    By='af+l"Polar Stereographic Y (km)"',
    frame="WSne",
)
# Save and show the figure
fig.savefig(fname="paper/figures/fig2_deepbedmap_dem.png")
fig.show()
