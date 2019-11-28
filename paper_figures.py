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
import subprocess

import geopandas as gpd
import IPython.display
import numpy as np
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
    # to_UnPool,
    # to_ConvRes,
    # to_ConvSoftMax,
    # to_SoftMax,
    to_connection,
    # to_skip,
    to_end,
    to_generate,
)
from paper.figures.PlotNeuralNet.pycore.tikzeng3 import (
    to_scalefont,
    to_flatimage,
    to_curvedskip,
    to_InOut,
    to_RRDB,
    to_ConvRelu,
    to_Upsample,
)

# %% [markdown]
# # **Methodology**
#
# Uses personal [fork](https://github.com/weiji14/PlotNeuralNet)
# of [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)
# for drawing convolutional neural network architecture diagram.

# %% [markdown]
# ### **Figure 1: DeepBedMap Model Architecture**

# %% [markdown]
# ### Thumbnail figures

# %%
fig = gmt.Figure()
fig.basemap(
    region=[-2700000, 2800000, -2200000, 2300000],
    projection="x1:100000000",
    frame="wsne",
)
gmt.makecpt(cmap="turbo", series=[-2800, 2800])
fig.grdimage(grid="lowres/bedmap2_bed.tif", cmap=True, I="+d", frame='+t"BEDMAP2"')
fig.savefig(fname="paper/figures/fig1a_bedmap2.png")
fig.show()

# %%
fig = gmt.Figure()
fig.grdimage(
    grid="https://www.the-cryosphere.net/13/665/2019/tc-13-665-2019-avatar-web.png",
    frame='+t"REMA"',
)
fig.savefig(fname="paper/figures/fig1b_rema.png")
fig.show()

# %%
fig = gmt.Figure()
fig.grdimage(
    grid="https://news.agu.org/files/2019/07/AntarcticaMap.jpg",
    frame='+t"Ice Velocity"',
)
fig.savefig(fname="paper/figures/fig1c_measures.png")
fig.show()

# %%
fig = gmt.Figure()
fig.grdimage(
    grid="https://wol-prod-cdn.literatumonline.com/cms/attachment/8bbfdd40-ea5b-409e-82a8-63a3b9ce4b7e/jgrd11996-fig-0005.png",
    frame='+t"Snow Accumulation"',
)
fig.savefig(fname="paper/figures/fig1d_accumulation.png")
fig.show()


# %% [markdown]
# ### Create [TikZ](https://en.wikipedia.org/wiki/PGF/TikZ) vector graphics of model architecture

# %%
def sizes(cl=(64, 9), scale=(30, 2.5), offset="(1.0,0,0)"):
    """
    Outputs pretty sized kwargs for PlotNeuralNet code,
    given [c]hannel(s) and [l]ength (we assume a square input, i.e. height==width).
    Optionally, adjust the channel and length scaling factor (default to (20, 2.5)).
    """
    channels, length = cl[0], cl[1]
    c_scale, l_scale = scale[0], scale[1]

    kwargs = {
        "n_filer": channels,
        "s_filer": length,
        "height": length / l_scale,
        "depth": length / l_scale,
        "offset": offset,
    }
    if isinstance(channels, tuple):
        kwargs["width"] = tuple([c / c_scale for c in channels])
    else:  # (int, float)
        kwargs["width"] = cl[0] / scale[0]

    return kwargs


# %%
input_layers = [
    # Actual raster images
    to_flatimage("paper/figures/fig1a_bedmap2.png", to="(-1.8,3,0)", width=2, height=2),
    to_flatimage("paper/figures/fig1b_rema.png", to="(-1.6,0,0)", width=2, height=2),
    to_flatimage(
        "paper/figures/fig1c_measures.png", to="(-1.8,-3.5,0)", width=2, height=2
    ),
    to_flatimage(
        "paper/figures/fig1d_accumulation.png", to="(-1.8,-5.5,0)", width=2, height=2
    ),
    # Input Raster Images
    to_InOut(name="x0_img", **sizes(cl=(1, 11)), to="(-1.2,3,0)"),
    to_InOut(name="w1_img", **sizes(cl=(1, 110), scale=(20, 7.5)), to="(-1.0,0,0)"),
    to_InOut(name="w2_img", **sizes(cl=(1, 22)), to="(-1.2,-3.5,0)"),
    to_InOut(name="w3_img", **sizes(cl=(1, 11)), to="(-1.2,-5.5,0)"),
    # First Convolution on input image
    to_Conv(name="x0", **sizes(cl=(32, 11)), to="(x0_img-east)"),
    to_Conv(name="w1", **sizes(cl=(32, 110), scale=(20, 7.5)), to="(w1_img-east)"),
    to_Conv(name="w2", **sizes(cl=(32, 22)), to="(w2_img-east)"),
    to_Conv(name="w3", **sizes(cl=(32, 11)), to="(w3_img-east)"),
    to_connection(of="x0_img", to="x0"),
    to_connection(of="w1_img", to="w1"),
    to_connection(of="w2_img", to="w2"),
    to_connection(of="w3_img", to="w3"),
    # Second Convolution
    to_Conv(name="x0_", **sizes(cl=(32, 9)), to="(x0-east)"),
    to_Conv(name="w1_", **sizes(cl=(32, 9)), to="(w1-east)"),
    to_Conv(name="w2_", **sizes(cl=(32, 9)), to="(w2-east)"),
    to_Conv(name="w3_", **sizes(cl=(32, 9)), to="(w3-east)"),
    to_connection(of="x0", to="x0_"),
    to_connection(of="w1", to="w1_"),
    to_connection(of="w2", to="w2_"),
    to_connection(of="w3", to="w3_"),
    # Concatenated Inputs
    to_Conv(
        name="concat",
        **sizes(cl=(128, 9)),
        to="(w1_-east)",
        caption=r"Concatenated\\Inputs",
    ),
    to_connection(of="x0_", to="concat"),
    to_connection(of="w1_", to="concat"),
    to_connection(of="w2_", to="concat"),
    to_connection(of="w3_", to="concat"),
    # Label for Input Module
    to_Pool(
        name="input-module-label",
        offset="(0,-1.5,0)",
        to="(w3_img-west)",
        caption=r"Input Module",
        width=24,
        height=0.1,
        depth=0.1,
        opacity=0.2,
    ),
]

# %%
rrdb_layers_simple = [
    # Residual in Residual Dense Block Layers
    #
    to_ConvRelu(
        name="pre-residual",
        **sizes(cl=((64,), 9)),
        to="(concat-east)",
        caption="Pre-residual",
    ),
    to_connection(of="concat", to="pre-residual"),
    # RRDB simplified box
    to_RRDB(
        name="rrdb-blocks",
        **sizes(cl=((32,) * 12, 9)),
        to="(pre-residual-east)",
        caption=r"Residual-in-Residual\\Dense Blocks\\x12",
    ),
    to_connection(of="pre-residual", to="rrdb-blocks"),
    #
    to_Conv(
        name="post-residual",
        **sizes(cl=(64, 9)),
        to="(rrdb-blocks-east)",
        caption="Post-residual",
    ),
    to_connection(of="rrdb-blocks", to="post-residual"),
    # Skip Connection
    to_curvedskip(of="pre-residual", to="post-residual", xoffset=0.6),
    # Label for Core RRDB module
    to_Pool(
        name="core-module-label",
        offset="(0.3,0,0)",
        to="(input-module-label-east)",
        caption=r"Core Module",
        width=32,
        height=0.1,
        depth=0.1,
        opacity=0.2,
    ),
]
# RRDB full expansion
rrdb_layers_complex = [
    # First Dense Block
    to_Conv(
        name="block1",
        **sizes(cl=(160, 9), offset="(-3.0,-4.5,0)"),
        to="(rrdb-blocks-west)",
        caption="Dense Block 1",
    ),
    #
    # Second Dense Block
    to_ConvRelu(name="block2a", **sizes(cl=((32,), 9)), to="(block1-east)"),
    to_ConvRelu(name="block2b", **sizes(cl=((32,), 9)), to="(block2a-east)"),
    to_ConvRelu(name="block2c", **sizes(cl=((32,), 9)), to="(block2b-east)"),
    to_ConvRelu(name="block2d", **sizes(cl=((32,), 9)), to="(block2c-east)"),
    to_Conv(name="block2e", **sizes(cl=(32, 9)), to="(block2d-east)"),
    # TODO write a loop...
    # inter-block connectors
    to_connection(of="block1", to="block2a"),
    to_connection(of="block2a", to="block2b"),
    to_connection(of="block2b", to="block2c"),
    to_connection(of="block2c", to="block2d"),
    to_connection(of="block2d", to="block2e"),
    # 1st order skips
    to_curvedskip(of="block1", to="block2a", xoffset=0.45),
    to_curvedskip(of="block2a", to="block2b", xoffset=0.45),
    to_curvedskip(of="block2b", to="block2c", xoffset=0.45),
    to_curvedskip(of="block2c", to="block2d", xoffset=0.45),
    # 2nd order skips
    to_curvedskip(of="block1", to="block2b", xoffset=0.45),
    to_curvedskip(of="block2a", to="block2c", xoffset=0.45),
    to_curvedskip(of="block2b", to="block2d", xoffset=0.45),
    # 3rd order skips
    to_curvedskip(of="block1", to="block2c", xoffset=0.45),
    to_curvedskip(of="block2a", to="block2d", xoffset=0.45),
    # 4th order skips
    to_curvedskip(of="block1", to="block2d", xoffset=0.45),
    #
    # Third Dense Block
    to_Conv(
        name="block3",
        **sizes(cl=(160, 9)),
        to="(block2e-east)",
        caption="Dense Block 3",
    ),
    to_connection(of="block2e", to="block3"),
    # ... connector
    to_Pool(
        name="ghost-block",
        offset="(1.0,0,0)",
        to="(block3-east)",
        caption=r"\dots",
        width=0,
        height=0.1,
        depth=0.1,
        opacity=0.0,
    ),
    to_connection(of="block3", to="ghost-block"),
]

# %%
upsampling_layers = [
    # Upsampling layers
    to_Upsample(name="upsample1", **sizes(cl=(64, 18)), to="(post-residual-east)"),
    to_ConvRelu(
        name="post-upsample1-conv",
        **sizes(cl=(64, 18), offset="(0,0,0)"),
        to="(upsample1-east)",
    ),
    to_connection(of="post-residual", to="upsample1"),
    #
    to_Upsample(
        name="upsample2",
        **sizes(cl=(64, 36), offset="(0.6,0,0)"),
        to="(post-upsample1-conv-east)",
        caption=r"Upsampling\\Blocks",
    ),
    to_ConvRelu(
        name="post-upsample2-conv",
        **sizes(cl=(64, 36), offset="(0,0,0)"),
        to="(upsample2-east)",
    ),
    to_connection(of="post-upsample1-conv", to="upsample2"),
    # Deformable Convolution layers
    to_ConvRelu(
        name="final-conv-block1",
        **sizes(cl=(64, 36), offset="(1.4,0,0)"),
        to="(post-upsample2-conv-east)",
        caption=r"Deformable\\Conv",
    ),
    to_connection(of="post-upsample2-conv", to="final-conv-block1"),
    to_Conv(
        name="final-conv-block2",
        **sizes(cl=(64, 36), offset="(0,0,0)"),
        to="(final-conv-block1-east)",
    ),
    # Output DeepBedMap DEM!
    to_InOut(
        name="deepbedmap-dem",
        **sizes(cl=(1, 36), offset="(1.2,0,0)"),
        to="(final-conv-block2-east)",
        caption=r"DeepBedMap\\DEM",
    ),
    to_connection(of="final-conv-block2", to="deepbedmap-dem"),
    #
    to_flatimage(
        "paper/figures/fig1e_deepbedmap.png", to="(21,0,0)", width=5, height=5
    ),
    # Label for Upsampling Module
    to_Pool(
        name="upsampling-module-label",
        offset="(0.3,0,0)",
        to="(core-module-label-east)",
        caption=r"Upsampling Module",
        width=32,
        height=0.1,
        depth=0.1,
        opacity=0.2,
    ),
]
legend_key = [
    to_Pool(
        name="key",
        offset="(3,-2.4,0)",
        to="(deepbedmap-dem-east)",
        caption=r"Key:",
        width=0,
        height=0.1,
        depth=0.1,
        opacity=0.0,
    ),
    to_Conv(
        name="conv",
        s_filer="\scriptsize Pixels",
        n_filer=("'Channels'", 0, 0),
        offset="(-1.2,-1.6,0)",
        to="(key-southwest)",
        width=64 / 30,
        height=3.6,
        depth=3.6,
        caption=r"Convolution\\Layer",
    ),
    to_ConvRelu(
        name="convrelu",
        **sizes(offset="(2,0,0)"),
        to="(conv-east)",
        caption=r"Convolution\\+LeakyReLU",
    ),
    # to_RRDB(name="rrdb-block", **sizes(offset="(2,0,0)"), to="(convrelu-east)", caption=r"RRDB"),
    to_Upsample(
        name="upsample",
        **sizes(offset="(0,-2,0)"),
        to="(conv-southwest)",
        caption=r"NN\\Upsample",
    ),
    to_InOut(
        name="in-out",
        **sizes(offset="(2,0,0)"),
        to="(upsample-east)",
        caption=r"Input/Output\\Images",
    ),
]

# %%
arch = [
    to_head("paper/figures/PlotNeuralNet"),
    to_cor(),
    to_begin(),
    to_scalefont(fontsize=r"\small"),
    *input_layers,
    *rrdb_layers_simple,
    # *rrdb_layers_complex,
    *upsampling_layers,
    *legend_key,
    to_end(),
]
to_generate(arch=arch, pathname="paper/figures/fig1_deepbedmap_architecture.tex")
# sudo apt install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra
_ = subprocess.run(
    [
        "pdflatex",
        "-output-directory=paper/figures",
        "paper/figures/fig1_deepbedmap_architecture.tex",
    ],
    stdout=subprocess.PIPE,
)

# %%
IPython.display.IFrame(
    src="paper/figures/fig1_deepbedmap_architecture.pdf", width=900, height=450
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
