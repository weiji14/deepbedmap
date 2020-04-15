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
import skimage
import xarray as xr

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
from features.environment import _load_ipynb_modules

data_prep = _load_ipynb_modules("data_prep.ipynb")
deepbedmap = _load_ipynb_modules("deepbedmap.ipynb")

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
gmt.makecpt(cmap="jet", series=[-2800, 2800, 200], D="o")
fig.grdimage(
    region=[-2700000, 2800000, -2200000, 2300000],
    projection="X8c/7c",
    grid="lowres/bedmap2_bed.tif",
    cmap=True,
    I="+d",
    Q=True,
    # frame='+t"BEDMAP2"'
)
fig.savefig(fname="paper/figures/fig1a_bedmap2.png")
fig.show()

# %%
fig = gmt.Figure()
fig.grdimage(
    grid="https://www.the-cryosphere.net/13/665/2019/tc-13-665-2019-avatar-web.png",
    # frame='+t"REMA"',
)
fig.savefig(fname="paper/figures/fig1b_rema.png")
fig.show()

# %%
fig = gmt.Figure()
fig.grdimage(
    grid="https://news.agu.org/files/2019/07/AntarcticaMap.jpg",
    # frame='+t"Ice Velocity"',
)
fig.savefig(fname="paper/figures/fig1c_measures.png")
fig.show()

# %%
fig = gmt.Figure()
fig.grdimage(
    grid="https://wol-prod-cdn.literatumonline.com/cms/attachment/8bbfdd40-ea5b-409e-82a8-63a3b9ce4b7e/jgrd11996-fig-0005.png",
    # frame='+t"Snow Accumulation"',
)
fig.savefig(fname="paper/figures/fig1d_accumulation.png")
fig.show()

# %%
fig = gmt.Figure()
gmt.makecpt(cmap="oleron", series=[-2000, 4500])
fig.grdimage(
    grid="model/deepbedmap3_big_int16.tif",
    region=[-2700000, 2800000, -2200000, 2300000],
    projection="x1:30000000",
    cmap=True,
    Q=True,
)
fig.savefig(fname="paper/figures/fig1e_deepbedmap.png")
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
    to_flatimage(
        "paper/figures/fig1a_bedmap2.png", to="(-1.8,4,0)", width=2, height=1.8
    ),
    to_flatimage("paper/figures/fig1b_rema.png", to="(-1.8,0,0)", width=2, height=1.8),
    to_flatimage(
        "paper/figures/fig1c_measures.png", to="(-1.8,-4.5,0)", width=2, height=1.6
    ),
    to_flatimage(
        "paper/figures/fig1d_accumulation.png", to="(-1.8,-8,0)", width=2, height=1.8
    ),
    # Input Raster Images
    to_InOut(name="x0_img", **sizes(cl=(1, 11)), to="(-1.2,4,0)", caption="BEDMAP2"),
    to_InOut(
        name="w1_img",
        **sizes(cl=(1, 110), scale=(20, 7.5)),
        to="(-1.0,0,0)",
        caption="REMA",
    ),
    to_InOut(
        name="w2_img", **sizes(cl=(2, 22)), to="(-1.2,-4.5,0)", caption="MEaSUREs"
    ),
    to_InOut(
        name="w3_img", **sizes(cl=(1, 11)), to="(-1.2,-8,0)", caption="Accumulation"
    ),
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
        offset="(0,-2.5,0)",
        to="(w3_img-east)",
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
    to_Pool(
        name="",
        offset="(1.7,2.5,0)",
        to="(rrdb-blocks-west)",
        caption=r"Skip",
        width=0,
        height=0,
        depth=0,
        opacity=0,
    ),
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
        "paper/figures/fig1e_deepbedmap.png", to="(21,0,0)", width=5, height=5 * 18 / 22
    ),
    # Label for Upsampling Module
    to_Pool(
        name="upsampling-module-label",
        offset="(0.3,0,0)",
        to="(core-module-label-east)",
        caption=r"Upsampling Module",
        width=33,
        height=0.1,
        depth=0.1,
        opacity=0.2,
    ),
]
legend_key = [
    to_Pool(
        name="key",
        offset="(2.7,-3,0)",
        to="(deepbedmap-dem-east)",
        caption=r"Key:",
        width=0,
        height=0,
        depth=0,
        opacity=0,
    ),
    to_Conv(
        name="conv",
        s_filer="\scriptsize Pixels",
        n_filer=("'Channels'", 0, 0),
        offset="(-1.2,-2,0)",
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
        **sizes(offset="(0,-2.5,0)"),
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
# Compress pdf following https://tex.stackexchange.com/a/19047
pdfcompress = lambda pdfpath: subprocess.run(
    [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        "-dPrinted=false",
        f"-sOutputFile={pdfpath}_compressed.pdf",
        f"{pdfpath}.pdf",
    ],
    stdout=subprocess.PIPE,
)
_ = pdfcompress(pdfpath="paper/figures/fig1_deepbedmap_architecture")
# %%
IPython.display.IFrame(
    to_InOut(
        name="deepbedmap-dem",
        **sizes(cl=(1, 36), offset="(1.2,0,0)"),
        to="(final-conv-block2-east)",
        caption=r"DeepBedMap\\DEM",
    ),
    src="paper/figures/fig1_deepbedmap_architecture_compressed.pdf",
    width=900,
    height=450,
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
# - Thwaites Glacier extent in Figure 4
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
fig.savefig(fname="paper/figures/fig2_deepbedmap_dem.pdf", dpi=300)
_ = pdfcompress(pdfpath="paper/figures/fig2_deepbedmap_dem")
fig.show()


# %%

# %% [markdown]
# ### **Figure 3: 3D plots over Pine Island Glacier**

# %%
# Process BedMachine Antarctica grid
# Get from NSIDC at https://doi.org/10.5067/C2GFER6PTOS4
window_bound = region_pineisle
M_tile = data_prep.selective_tile(
    filepath="netcdf:model/BedMachineAntarctica_2019-11-05_v01.nc:bed",
    window_bounds=[[*window_bound]],
    interpolate=True,
)
print(M_tile.shape)
bedmachinea = skimage.transform.rescale(
    image=M_tile[0, 0, :, :].astype(np.int32),
    scale=2,  # 2x upscaling
    order=3,  # cubic interpolation
    mode="reflect",
    anti_aliasing=True,
    multichannel=False,
    preserve_range=True,
)
bedmachinea = np.expand_dims(np.expand_dims(bedmachinea, axis=0), axis=0)
print(bedmachinea.shape)

# Save Bicubic Resampled BedMachine Antarctica to GeoTiff and NetCDF format
bedmachinea_grid = data_prep.save_array_to_grid(
    outfilepath="model/bedmachinea",
    window_bound=window_bound,
    array=bedmachinea[0, :, :, :],
    save_netcdf=True,
)

# %%
fig = gmt.Figure()
deepbedmap.subplot(
    directive="begin", row=2, col=2, A="+jCT+o-4c/-5c", Fs="9c/9c", M="2c/3c"
)
deepbedmap.plot_3d_view(
    img="model/deepbedmap3.nc",  # DeepBedMap
    ax=(0, 0),
    zmin=-1400,
    title="a) DeepBedMap",  # ours
    zlabel="Bed elevation (metres)",
)
deepbedmap.plot_3d_view(
    img="model/cubicbedmap.nc",  # BEDMAP2
    ax=(0, 1),
    zmin=-1400,
    title="b) BEDMAP2",
    zlabel="Bed elevation (metres)",
)
deepbedmap.plot_3d_view(
    img="model/elevdiffmap.nc",  # DeepBedMap - BEDMAP2
    ax=(1, 0),
    zmin=-400,
    cmap="vik",
    title="c) DeepBedMap - BEDMAP2",
    zlabel="Difference (metres)",
)
deepbedmap.plot_3d_view(
    img="model/bedmachinea.nc",  # BedMachine Antarctica
    ax=(1, 1),
    zmin=-1400,
    title="d) BedMachine",
    zlabel="Bed elevation (metres)",
)
deepbedmap.subplot(directive="end")
fig.savefig(
    fname="paper/figures/fig3_qualitative_bed_comparison.png", dpi=300, crop=False
)
fig.show()

# %%

# %% [markdown]
# ### **Figure 4: Closeup images of DeepBedMap_DEM**

# %%
def closeup_fig(
    letter: str,
    name: str,
    midx: int,
    midy: int,
    annot_xyt: list,
    size: int = 100_000,
    fig: gmt.Figure = None,
):
    """
    Produces a closeup figure of a DeepBedMap_DEM area, with text annotations
    """
    region = [midx - size, midx + size, midy - size, midy + size]

    if fig is None:  # initialize figure if no Figure is given
        fig = gmt.Figure()

    # Plot DeepBedMap Digital Elevation Model (DEM)
    gmt.makecpt(cmap="oleron", series=[-2000, 4500])
    fig.grdimage(
        grid="model/deepbedmap3_big_int16.tif",
        # grid="@BEDMAP_elevation.nc",
        region=region,
        projection="x1:1500000",
        cmap=True,
        shading="+d",  # default illumination from azimuth -45deg, intensity of +1
        Q=True,
    )
    # Plot text annotation, black text against white background
    for x, y, text in annot_xyt:
        fig.text(
            x=x,
            y=y,
            text=text,
            font="12p,Helvetica-Bold,black",
            G="white",
            region=region,
            projection="x1:1500000",
        )
    # Plot map elements (frame, colorbar)
    fig.basemap(
        region=[r / 1000 for r in region],
        projection="x1:1500",
        Bx='af+l"Polar Stereographic X (km)"',
        By='af+l"Polar Stereographic Y (km)"',
        frame=f'WSne+t"{letter}) {name}"',
    )
    fig.colorbar(
        position="jBL+jBL+o2.0c/0.5c+w2.0c/0.2c+m",
        box="+gwhite+p0.5p",
        frame=["af", 'x+l"Elevation"', "y+lkm"],
        cmap="+Uk",  # kilo-units, i.e. divide by 1000
        S=True,  # no lines inside color scalebar
    )
    # Save and show the figure
    # fig.savefig(fname=f"paper/figures/fig4{letter}_deepbedmap_closeup.png")

    return fig


# %%
fig = gmt.Figure()
deepbedmap.subplot(directive="begin", row=3, col=3, B="wsne", Fs="15c/15c", M="1c/1c")
# Transantarctic Mountains - Scott Glacier
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="a",
    name="Scott Glacier",
    midx=-200_000,
    midy=-400_000,
    annot_xyt=[(-230000, -390000, "S")],
    fig=fig,
)
# Siple Coast - Whillans Ice Stream
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="b",
    name="Whillans Ice Stream",
    midx=-400_000,
    midy=-550_000,
    annot_xyt=[(-350000, -540000, "R")],
    fig=fig,
)
# Siple Coast - Bindschadler Ice Stream
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="c",
    name="Bindschadler Ice Stream",
    midx=-550_000,
    midy=-800_000,
    annot_xyt=[(-610000, -740000, "R")],
    fig=fig,
)
# Weddell Sea Region - Evans Ice Stream
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="d",
    name="Evans Ice Stream",
    midx=-1500_000,
    midy=350_000,
    annot_xyt=[(-1450000, 320000, "R")],
    fig=fig,
)
# Weddell Sea Region - Rutford Ice Stream
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="e",
    name="Rutford Ice Stream",
    midx=-1300_000,
    midy=150_000,
    annot_xyt=[(-1220000, 150000, "R")],
    fig=fig,
)
# Weddell Sea Region - Foundation Ice Stream
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="f",
    name="Foundation Ice Stream",
    midx=-600_000,
    midy=350_000,
    annot_xyt=[(-630000, 360000, "R")],
    fig=fig,
)
# East Antarctica - Totten Glacier
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="g",
    name="Totten Glacier",
    midx=2250_000,
    midy=-1050_000,
    annot_xyt=[(2270000, -1070000, "R"), (2180000, -970000, "W")],
    fig=fig,
)
# East Antarctica - Byrd Glacier
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="h",
    name="Byrd Glacier",
    midx=400_000,
    midy=-950_000,
    annot_xyt=[(380000, -980000, "R"), (400000, -1040000, "S")],
    fig=fig,
)
# East Antarctica - Gamburtsev Subglacial Mountains
deepbedmap.subplot(directive="set")
fig = closeup_fig(
    letter="i",
    name="Gamburtsev Subglacial Mountains",
    midx=800_000,
    midy=200_000,
    annot_xyt=[(710000, 240000, "T")],
    fig=fig,
)
deepbedmap.subplot(directive="end")
fig.savefig(fname=f"paper/figures/fig4_deepbedmap_closeups.eps", dpi=300)
fig.show()

# %%

# %% [markdown]
# ## **Surface Roughness**

# %%
region = [
    region_thwaites.left,
    region_thwaites.right,
    region_thwaites.bottom,
    region_thwaites.top,
]
kmregion = [r / 1000 for r in region]  # coordinates in km instead of m


# %%
def standard_deviation_2d(grid: xr.DataArray, window_length: int):
    """
    Get standard deviation of each pixel in a grid over a rolling 2d window.
    A fairly basic proxy for the 'roughness' of a terrain.

    >>> grid = xr.DataArray(data=np.arange(0, 15, 1).reshape(3, 5), dims=("x", "y"))
    >>> standard_deviation_2d(grid=grid, window_length=3)
    <xarray.DataArray (x: 3, y: 5)>
    array([[2.54951 , 2.629956, 2.629956, 2.629956, 2.54951 ],
           [4.112988, 4.163332, 4.163332, 4.163332, 4.112988],
           [2.54951 , 2.629956, 2.629956, 2.629956, 2.54951 ]])
    Dimensions without coordinates: x, y
    """
    yrol = grid.rolling(y=window_length, center=True).construct("y_window")
    xrol = yrol.rolling(x=window_length, center=True).construct("x_window")

    # xr.apply_ufunc(np.std, xrol, input_core_dims=[["y_window", "x_window"]])

    roughness = xrol.std(dim=["y_window", "x_window"])

    return roughness


# %%
def prepare_grid(file: str, region: list):
    """
    Prepares a raster grid for further plotting in PyGMT.
    Reads in the grid from file using xarray.open_rasterio,
    selects the first band and slices it using a bounding box region.
    Also changes data type to float32 if required.
    """
    grid = xr.open_rasterio(file).sel(
        band=1, x=slice(region[0], region[1]), y=slice(region[3], region[2])
    )
    if grid.dtype != np.float32:
        grid = grid.astype(np.float32)

    return grid


# %%
#!gmt grdcut model/deepbedmap3_big_int16.tif -Gmodel/deepbedmap3_thwaites.nc -R-1550000/-1250000/-550000/-300000
# deepbedmap3grid = prepare_grid(file="model/deepbedmap3_thwaites.nc", region=region)
deepbedmap3grid = prepare_grid(file="model/deepbedmap3_big_int16.tif", region=region)
groundtruthgrid = prepare_grid(file="highres/20xx_Antarctica_DC8.nc", region=region)
bedmap2grid = prepare_grid(file="lowres/bedmap2_bed.tif", region=region)
cubicbedmap2 = skimage.transform.rescale(
    image=bedmap2grid.astype(np.int32),
    scale=4,  # 4x upscaling
    order=3,  # cubic interpolation
    mode="reflect",
    anti_aliasing=True,
    multichannel=False,
    preserve_range=True,
)
cubicbedmap2grid = xr.DataArray(
    data=cubicbedmap2[2:-2, 2:-2], coords=deepbedmap3grid.coords
)

bedmachinegrid = prepare_grid(
    file="netcdf:model/BedMachineAntarctica_2019-11-05_v01.nc:bed", region=region
)
cubicbedmachine = skimage.transform.rescale(
    image=bedmachinegrid.astype(np.int32),
    scale=2,  # 2x upscaling
    order=3,  # cubic interpolation
    mode="reflect",
    anti_aliasing=True,
    multichannel=False,
    preserve_range=True,
)
cubicbedmachinegrid = xr.DataArray(
    data=cubicbedmachine[1:-1, 1:-1], coords=deepbedmap3grid.coords
)

gridDict = {
    "DeepBedMap": deepbedmap3grid,
    "Groundtruth": groundtruthgrid,
    # "BEDMAP2": cubicbedmap2grid,
    "BedMachine": cubicbedmachinegrid,
}
roughDict = {}
for name, grid in gridDict.items():
    roughness = standard_deviation_2d(grid=grid, window_length=5)
    roughDict[name] = roughness

# %%
# Subset and Plot Operation IceBridge (OIB) groundtruth points over a transect
# oibpoints = data_prep.ascii_to_xyz(pipeline_file="highres/20xx_Antarctica_DC8.json")
_ = data_prep.download_to_path(
    path="highres/Data_20141121_05.csv",
    url="https://data.cresis.ku.edu/data/rds/2014_Antarctica_DC8/csv_good/Data_20141121_05.csv",
)
oibpoints = data_prep.ascii_to_xyz(pipeline_file="highres/Data_20141121_05.json")
oibpoints = oibpoints.where(
    cond=(
        (oibpoints.x > region[0])
        & (oibpoints.x < -1300_000)  # (oibpoints.x < region[1])
        & (oibpoints.y > region[2])
        & (oibpoints.y < -425_000)  # (oibpoints.y < region[3])
    )
)
oibpoints.dropna(inplace=True)
oibpoints.reset_index(drop=True, inplace=True)

len(oibpoints)

# %% [raw]
# # Start and Stop Longitude/Latitude coordinates manually picked from
# # Operation IceBridge dataset, specifically Data_20141121_05.json
# import pyproj
#
# start_lonlat = (-106.415404, -76.343903)
# stop_lonlat = (-104.371744, -77.692208)
#
#
# reprj_func = pyproj.Transformer.from_crs(
#     crs_from=pyproj.CRS.from_epsg(4326),
#     crs_to=pyproj.CRS.from_epsg(3031),
#     always_xy=True,
# )
#
# start_xy = reprj_func.transform(xx=start_lonlat[0], yy=start_lonlat[1])
# stop_xy = reprj_func.transform(xx=stop_lonlat[0], yy=stop_lonlat[1])
# start_stop_inc = f"{start_xy[0]}/{start_xy[1]}/{stop_xy[0]}/{stop_xy[1]}/+i125"
# print(start_stop_inc)


# %%
elevpoints = {}
# Get elevation (z) values
for name, grid in gridDict.items():
    elevpoints[name] = gmt.grdtrack(
        points=oibpoints[["x", "y"]],
        grid=grid,
        newcolname="elevation",
        R="/".join(str(r) for r in region),
        # E="-1573985/-470866/-993375/-464996+i250",
    )
    elevpoints[name] = elevpoints[name].dropna()
    elevpoints[name].x = elevpoints[name].x.astype(float)
    print(len(elevpoints[name]), name)
# Get roughness values
for name, grid in roughDict.items():
    roughness = gmt.grdtrack(
        points=oibpoints[["x", "y"]],
        grid=grid,
        newcolname="roughness",
        R="/".join(str(r) for r in region),
        # E=start_stop_inc,
    ).roughness
    elevpoints[name]["roughness"] = roughness

# %% [raw]
# # deepbedmap3_error = elevpoints["model/deepbedmap3_thwaites.nc"].z - oibpoints.z
# # cubicbedmap_error = elevpoints["lowres/bedmap2_bed.tif"].z - oibpoints.z
# deepbedmap3_error = (
#     elevpoints["DeepBedMap"].roughness - elevpoints["Groundtruth"].roughness
# )
# cubicbedmap_error = (
#     elevpoints["BedMap2"].roughness - elevpoints["Groundtruth"].roughness
# )
#
# deepbedmap3_error.describe()
# cubicbedmap_error.describe()
#
# rmse_deepbedmap3 = (deepbedmap3_error ** 2).mean() ** 0.5
# rmse_cubicbedmap = (cubicbedmap_error ** 2).mean() ** 0.5
# print(rmse_deepbedmap3, rmse_cubicbedmap)


# %% [markdown]
# ### **Figure 5: 2D view of roughness grids over Thwaites Glacier, West Antarctica**

# %%
## Copied cpt from GenericMappingTools/gmt @ 3fb8efa8bf2c3016d6b22d8e9f0e84dbcc1965ae
fig = gmt.Figure()
deepbedmap.subplot(directive="begin", row=2, col=2, Fs="17c/17c", C="1.5c", M="0c/1c")
deepbedmap.subplot(directive="set")

fig.basemap(
    region=kmregion,
    projection="X14c",
    Bx='af+l"Polar Stereographic X (km)"',
    By='af+l"Polar Stereographic Y (km)"',
    frame=['WSne+t"a) DeepBedMap DEM"'],
)
# Plot Figure 5a DeepBedMap DEM
gmt.makecpt(cmap="oleron", series=[-2000, 2500])
fig.grdimage(grid=gridDict["DeepBedMap"], region=region)
fig.colorbar(position="JBC", frame=["af", 'x+l"Elevation"', "y+lm"])
# Plot transect line points
fig.plot(
    x=oibpoints.x,
    y=oibpoints.y,
    color="orange",
    style="c0.1c",
    label='"Transect points"',
)
fig.legend(position="JBL+jBL+o0.2c", box="+gwhite+p1p")
# Save and show the figure
# fig.savefig(fname="paper/figures/fig5a_elevation_deepbedmap.png")
# fig.show()

# %%
# Plot Figures 5b, c, d 2D roughness grids
for letter, (name, grid) in zip(["b", "c", "d"], roughDict.items()):
    if name == "BEDMAP2":
        maxstddev = 100  # lower scale as few pixels with high standard dev
    elif name == "BedMachine":
        maxstddev = 200
    else:
        maxstddev = 400
    deepbedmap.subplot(directive="set")
    # fig = gmt.Figure()
    fig.basemap(
        region=kmregion,
        projection="X14c",
        Bx='af+l"Polar Stereographic X (km)"',
        By='af+l"Polar Stereographic Y (km)"',
        frame=[f'WSne+t"{letter}) {name} roughness"'],
    )
    print(float(grid.min()), float(grid.max()))
    gmt.makecpt(cmap="davos", series=[0, maxstddev, maxstddev / 8], M="d")
    fig.grdimage(grid=grid, region=region, cmap=True)
    fig.colorbar(position="JBC+ef", frame=["af", 'x+l"Standard Deviation"', "y+lm"])
    # fig.savefig(fname=f"paper/figures/fig5{letter}_roughness_{name.lower()}.png")

deepbedmap.subplot(directive="end")
fig.savefig(fname="paper/figures/fig5_elevation_roughness_grids.eps", dpi=300)
fig.show()

# %%

# %% [markdown]
# ### **Figure 6: 1D Elevation and Roughness over transect**

# %%
fig = gmt.Figure()
deepbedmap.subplot(
    directive="begin",
    row=2,
    col=1,
    A="+jLT",
    Fs="12c/6c",
    B="WSne",
    SC='b+l"Polar Stereographic X (km)"',
)
for zvalue, yrange in (("elevation", [-1600, -400]), ("roughness", [0, 100])):
    deepbedmap.subplot(directive="set")
    fig.basemap(region=[-1550, -1300, *yrange], frame=f'yaf+l"{zvalue.title()} (m)"')
    for grid, color in zip(roughDict, ("purple", "orange", "green")):
        fig.plot(
            x=elevpoints[grid].x,
            y=elevpoints[grid][zvalue],
            region=[-1550000, -1300000, *yrange],
            style="c0.02c",
            color=color,
            label=grid,
        )
    fig.legend(S=10)  # position="jTR+o0/0", box=True,
deepbedmap.subplot(directive="end")
fig.savefig(fname="paper/figures/fig6_elevation_roughness_transect.eps", dpi=300)
fig.show()

# %%
