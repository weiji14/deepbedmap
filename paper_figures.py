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

# %%
import os

import geopandas as gpd
import IPython.display
import pandas as pd
import pygmt as gmt
import rasterio

from paper.figures.PlotNeuralNet.pycore.tikzeng import (
    to_head,
    to_cor,
    to_begin,
    to_input,
    to_Conv,
    # to_ConvConvRelu,
    to_Pool,
    to_UnPool,
    to_ConvRes,
    # to_ConvSoftMax,
    # to_SoftMax,
    to_connection,
    to_skip,
    to_end,
    to_generate,
)

# %% [markdown]
# # **Methodology**
#
# Uses [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)
# for drawing convolutional neural network architecture diagram.

# %% [markdown]
# ### **Figure 1: DeepBedMap Model Architecture**

# %%
FIG_DIR = os.path.join("paper", "figures")
arch = [
    to_head(os.path.join(FIG_DIR, "PlotNeuralNet")),
    to_cor(),
    to_begin(),
    # input
    to_input(os.path.join(FIG_DIR, "ter_bedmap.png"), to="(-1.5,3,0)", width=3, height=3),
    to_input(os.path.join(FIG_DIR, "REMA-hillshade-rendering-800px-768x768.jpg"), to="(-1.5,0,0)", width=3, height=3),
    to_input(os.path.join(FIG_DIR, "glac_flowspeed.png"), to="(-1.5,-3,0)", width=3, height=3),
    to_input(os.path.join(FIG_DIR, "glac_albmap_snowacca.png"), to="(-1.5,-6,0)", width=3, height=3),

    # Input Convolutions
    to_Conv(name="x", s_filer=10, n_filer=32, offset="(0,3.5,0)", to="(0,0,0)", width=1.6, height=4, depth=4),
    to_Conv(name="w1", s_filer=100, n_filer=32, offset="(0,0,0)", to="(0,0,0)", width=1.6, height=16, depth=16),
    to_Conv(name="w2", s_filer=20, n_filer=32, offset="(0,-3.5,0)", to="(0,0,0)", width=1.6, height=8, depth=8),
    to_Conv(name="w3", s_filer=10, n_filer=32, offset="(0,-5.5,0)", to="(0,0,0)", width=1.6, height=4, depth=4),
    to_Conv(name="concat", s_filer=8, n_filer=128, offset="(1.2,0,0)", to="(w1-east)", width=4.8, height=4, depth=4, caption="Concat-enated Inputs"),
    to_connection(of="x", to="concat"),
    to_connection(of="w1", to="concat"),
    to_connection(of="w2", to="concat"),
    to_connection(of="w3", to="concat"),
    # RRDB Blocks
    to_Conv(name="pre-residual", s_filer=8, n_filer=64, offset="(1.2,0,0)", to="(concat-east)", width=3.2, height=4, depth=4, caption="Pre-residual"),
    to_connection(of="concat", to="pre-residual"),
    to_Pool(name="ghost-skip1", offset="(0.4,0,0)", to="(pre-residual-east)", width=0, height=0.1, depth=0.1, opacity=0.0),

    to_Conv(name="block1", s_filer=8, n_filer=64, offset="(0.8,0,0)", to="(pre-residual-east)", caption="Block 1", width=8, height=4, depth=4),
    to_connection(of="pre-residual", to="block1"),
    to_Pool(name="ghost-block", offset="(0.8,0,0)", to="(block1-east)", caption="\dots", width=0, height=0.1, depth=0.1, opacity=0.0),
    to_connection(of="block1", to="ghost-block"),
    to_Conv(name="blockB", s_filer=8, n_filer=64, offset="(0.8,0,0)", to="(ghost-block-east)", caption="Block B", width=8, height=4, depth=4),
    to_connection(of="ghost-block", to="blockB"),
    # Post-RRDB Layers
    to_Conv(name="post-residual", s_filer=8, n_filer=64, offset="(0.8,0,0)", to="(blockB-east)", width=3.2, height=4, depth=4, caption="Post-residual"),
    to_connection(of="blockB", to="post-residual"),
    to_Pool(name="ghost-skip2", offset="(0.4,0,0)", to="(post-residual-east)", width=0, height=0.1, depth=0.1, opacity=0.0),

    to_skip(of="ghost-skip1", to="ghost-skip2", pos=75),

    to_UnPool(name="upsample1", offset="(1.2,0,0)", to="(post-residual-east)", width=1.6, height=10, depth=10),
    to_connection(of="post-residual", to="upsample1"),
    to_UnPool(name="upsample2", offset="(0.3,0,0)", to="(upsample1-east)", width=1.6, height=16, depth=16, caption="Pixel-Shuffle Blocks 1\&2"),
    to_connection(of="upsample1", to="upsample2"),

    to_Conv(name="final-conv-block", s_filer=32, n_filer=32, offset="(1.2,0,0)", to="(upsample2-east)", width=1.6, height=16, depth=16, caption="Final Conv Block"),
    to_connection(of="upsample2", to="final-conv-block"),

    to_Conv(name="deepbedmap-dem", s_filer=32, n_filer=1, offset="(1.2,0,0)", to="(final-conv-block-east)", width=0.5, height=16, depth=16, caption="Deep-BedMap-DEM"),
    to_connection(of="final-conv-block", to="deepbedmap-dem"),

    to_end(),
]
to_generate(arch=arch, pathname=os.path.join(FIG_DIR, "deepbedmap_architecture.tex"))
!pdflatex -output-dir {FIG_DIR} {os.path.join(FIG_DIR, 'deepbedmap_architecture.tex')}

# %%
IPython.display.IFrame(
    src="paper/figures/deepbedmap_architecture.pdf", width=900, height=450
)


# %% [markdown]
# # **Results**
#
# Uses [PyGMT](https://github.com/GenericMappingTools/pygmt/) for drawing map plots.

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
