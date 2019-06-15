# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.1.4-rc1
#   kernelspec:
#     display_name: deepbedmap
#     language: python
#     name: deepbedmap
# ---

# %% [markdown]
# # **DeepBedMap**
#
# Predicting the bed elevation of Antarctica using a Super Resolution Deep Neural Network.

# %%
import math
import os
import typing

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import xarray as xr
import salem

import comet_ml
import cupy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import pygmt as gmt
import quilt
import rasterio
import skimage

import chainer

from features.environment import _load_ipynb_modules

data_prep = _load_ipynb_modules("data_prep.ipynb")

# %% [markdown]
# ## Get bounding box of area we want to predict on

# %%
def get_image_with_bounds(filepaths: list, indexers: dict = None) -> xr.DataArray:
    """
    Retrieve raster image in xarray.DataArray format patched
    with projected coordinate bounds as (xmin, ymin, xmax, ymax)

    Note that if more than one filepath is passed in,
    the output groundtruth image array will not be valid
    (see https://github.com/pydata/xarray/issues/2159),
    but the window_bound extents will be correct
    """

    with xr.open_mfdataset(paths=filepaths, concat_dim=None) as dataset:
        # Retrieve dataarray from NetCDF datasets
        dataarray = dataset.z.isel(indexers=indexers)

    # Patch projection information into xarray grid
    dataarray.attrs["pyproj_srs"] = "epsg:3031"
    sgrid = dataarray.salem.grid.corner_grid
    assert sgrid.origin == "lower-left"  # should be "lower-left", not "upper-left"

    # Patch bounding box extent into xarray grid
    if len(filepaths) == 1:
        left, right, bottom, top = sgrid.extent
    elif len(filepaths) > 1:
        print("WARN: using multiple inputs, output groundtruth image may look funny")
        x_offset, y_offset = sgrid.dx / 2, sgrid.dy / 2
        left, right = (
            float(dataarray.x[0] - x_offset),
            float(dataarray.x[-1] + x_offset),
        )
        assert sgrid.x0 == left
        bottom, top = (
            float(dataarray.y[0] - y_offset),
            float(dataarray.y[-1] + y_offset),
        )
        assert sgrid.y0 == bottom  # dataarray.y.min()-y_offset

    # check that y-axis and x-axis lengths are divisible by 4
    try:
        shape = int((top - bottom) / sgrid.dy), int((right - left) / sgrid.dx)
        assert all(i % 4 == 0 for i in shape)
    except AssertionError:
        print(f"WARN: Image shape {shape} should be divisible by 4 for DeepBedMap")
    finally:
        dataarray.attrs["bounds"] = [left, bottom, right, top]

    return dataarray


# %%
test_filepaths = ["highres/2007tx"]  # , "highres/2010tr", "highres/istarxx"]
groundtruth = get_image_with_bounds(
    filepaths=[f"{t}.nc" for t in test_filepaths],
    # indexers={"y": slice(0, -1), "x": slice(0, -1)},  # for 2007tx, 2010tr and istarxx
    indexers={"y": slice(1, -1), "x": slice(1, -1)},  # for 2007tx only
)

# %% [markdown]
# ## Get neural network input datasets for our area of interest

# %%
def get_deepbedmap_model_inputs(
    window_bound: rasterio.coords.BoundingBox, padding=1000
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Outputs one large tile for each of
    BEDMAP2, REMA and MEASURES Ice Flow Velocity
    according to a given window_bound in the form of
    (xmin, ymin, xmax, ymax).
    """
    data_prep = _load_ipynb_modules("data_prep.ipynb")

    X_tile = data_prep.selective_tile(
        filepath="lowres/bedmap2_bed.tif",
        window_bounds=[[*window_bound]],
        # out_shape=None,  # 1000m spatial resolution
        padding=padding,
    )
    W3_tile = data_prep.selective_tile(
        filepath="misc/Arthern_accumulation_bedmap2_grid1.tif",
        window_bounds=[[*window_bound]],
        # out_shape=None,  # 1000m spatial resolution
        padding=padding,
    )
    W2_tile = data_prep.selective_tile(
        filepath="misc/MEaSUREs_IceFlowSpeed_450m.tif",
        window_bounds=[[*window_bound]],
        out_shape=(2 * X_tile.shape[2], 2 * X_tile.shape[3]),  # 500m spatial resolution
        padding=padding,
        gapfill_raster_filepath="misc/lisa750_2013182_2017120_0000_0400_vv_v1_myr.tif",
    )
    W1_tile = data_prep.selective_tile(
        filepath="misc/REMA_100m_dem.tif",
        window_bounds=[[*window_bound]],
        # out_shape=(5 * W2_tile.shape[2], 5 * W2_tile.shape[3]),  # 100m spatial resolution
        padding=padding,
        gapfill_raster_filepath="misc/REMA_200m_dem_filled.tif",
    )

    return X_tile, W1_tile, W2_tile, W3_tile


# %%
def plot_3d_view(
    img: np.ndarray,
    ax: matplotlib.axes._subplots.Axes,
    elev: int = 60,
    azim: int = 330,
    cm_norm: matplotlib.colors.Normalize = None,
    title: str = None,
):
    # Get x, y, z data, assuming image in NCHW format
    image = img[0, :, :, :]
    xx, yy = np.mgrid[0 : image.shape[1], 0 : image.shape[2]]
    zz = image[0, :, :]

    # Make the 3D plot
    ax.view_init(elev=elev, azim=azim)
    ax.plot_surface(xx, yy, zz, cmap="BrBG", norm=cm_norm)
    ax.set_title(label=f"{title}\n", fontsize=22)

    return ax


# %%
X_tile, W1_tile, W2_tile, W3_tile = get_deepbedmap_model_inputs(
    window_bound=groundtruth.bounds
)
print(X_tile.shape, W1_tile.shape, W2_tile.shape, W3_tile.shape)

# Build quilt package for datasets covering our test region
reupload = True
if reupload == True:
    quilt.build(package="weiji14/deepbedmap/model/test/W1_tile", path=W1_tile)
    quilt.build(package="weiji14/deepbedmap/model/test/W2_tile", path=W2_tile)
    quilt.build(package="weiji14/deepbedmap/model/test/W3_tile", path=W3_tile)
    quilt.build(package="weiji14/deepbedmap/model/test/X_tile", path=X_tile)
    quilt.push(package="weiji14/deepbedmap/model/test", is_public=True)

# %%
fig, axarr = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(16, 12))
axarr[0, 0].imshow(X_tile[0, 0, :, :], cmap="BrBG")
axarr[0, 0].set_title("BEDMAP2\n(1000m resolution)")
axarr[0, 1].imshow(W1_tile[0, 0, :, :], cmap="BrBG")
axarr[0, 1].set_title("Reference Elevation Model of Antarctica\n(100m resolution)")
axarr[0, 2].imshow(W2_tile[0, 0, :, :], cmap="BrBG")
axarr[0, 2].set_title("MEaSUREs Ice Velocity\n(450m, resampled to 500m)")
axarr[0, 3].imshow(W3_tile[0, 0, :, :], cmap="BrBG")
axarr[0, 3].set_title("Antarctic Snow Accumulation\n(1000m resolution)")
plt.show()

# %%
fig = plt.figure(figsize=plt.figaspect(1 / 4) * 2.5)

ax = fig.add_subplot(1, 4, 1, projection="3d")
ax = plot_3d_view(img=X_tile, ax=ax, title="BEDMAP2\n(1000m resolution)")
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

ax = fig.add_subplot(1, 4, 2, projection="3d")
ax = plot_3d_view(
    img=W1_tile,
    ax=ax,
    title="Reference Elevation Model of Antarctica\n(100m resolution)",
)
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

ax = fig.add_subplot(1, 4, 3, projection="3d")
ax = plot_3d_view(
    img=W2_tile, ax=ax, title="MEaSUREs Surface Ice Velocity\n(450m, resampled to 500m)"
)
ax.set_zlabel("\n\nSurface Ice Velocity (metres/year)", fontsize=16)

ax = fig.add_subplot(1, 4, 4, projection="3d")
ax = plot_3d_view(
    img=W3_tile, ax=ax, title="Antarctic Snow Accumulation\n(1000m resolution)"
)
ax.set_zlabel("\n\nSnow Accumulation (kg/m2/a)", fontsize=16)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Create custom neural network for our area of interest
#
# Fully convolutional networks rock!!
# Since we have a fully convolutional model architecture,
# we can change the shape of the inputs/outputs,
# but use the same trained weights!
# That way we can predict directly on an arbitrarily sized window.

# %%
def load_trained_model(
    model=None,
    model_weights_path: str = "model/weights/srgan_generator_model_weights.npz",
):
    """
    Builds the Generator component of the DeepBedMap neural network.
    Also loads trained parameter weights into the model from a .npz file.
    """
    srgan_train = _load_ipynb_modules("srgan_train.ipynb")

    if model is None:
        model = srgan_train.GeneratorModel(
            # num_residual_blocks=12,
            # residual_scaling=0.4,
        )

    # Load trained neural network weights into model
    chainer.serializers.load_npz(file=model_weights_path, obj=model)

    return model


# %% [markdown]
# ## Make prediction

# %%
model = load_trained_model()

# %%
# Prediction on small area
Y_hat = model.forward(x=X_tile, w1=W1_tile, w2=W2_tile, w3=W3_tile).array

# %% [markdown]
# ## Load other interpolated grids for comparison
#
# - Bicubic interpolated BEDMAP2
# - Synthetic High Resolution Grid from [Graham et al.](https://doi.org/10.5194/essd-9-267-2017)

# %%
cubicbedmap2 = skimage.transform.rescale(
    image=X_tile[0, 0, 1:-1, 1:-1].astype(np.int32),
    scale=4,  # 4x upscaling
    order=3,  # cubic interpolation
    mode="reflect",
    anti_aliasing=True,
    multichannel=False,
    preserve_range=True,
)
cubicbedmap2 = np.expand_dims(np.expand_dims(cubicbedmap2, axis=0), axis=0)
print(cubicbedmap2.shape, Y_hat.shape)
assert cubicbedmap2.shape == Y_hat.shape

# %%
S_tile = data_prep.selective_tile(
    filepath="model/hres.tif", window_bounds=[[*groundtruth.bounds]]
)

# %% [markdown]
# ## Plot results

# %%
fig, axarr = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(22, 12))
axarr[0, 0].imshow(X_tile[0, 0, :, :], cmap="BrBG")
axarr[0, 0].set_title("BEDMAP2")
axarr[0, 1].imshow(Y_hat[0, 0, :, :], cmap="BrBG")
axarr[0, 1].set_title("Super Resolution Generative Adversarial Network prediction")
axarr[0, 2].imshow(S_tile[0, 0, :, :], cmap="BrBG")
axarr[0, 2].set_title("Synthetic High Resolution Grid")
groundtruth.plot(ax=axarr[0, 3], cmap="BrBG")
axarr[0, 3].set_title("Groundtruth grids")
plt.show()

# %%
fig = plt.figure(figsize=plt.figaspect(12 / 9) * 4.5)  # (height/width)*scaling

zmin, zmax = (Y_hat.min(), Y_hat.max())
norm_Z = matplotlib.cm.colors.Normalize(vmin=zmin, vmax=zmax)

# DeepBedMap
ax = fig.add_subplot(2, 2, 1, projection="3d")
ax = plot_3d_view(
    img=Y_hat, ax=ax, cm_norm=norm_Z, title="DeepBedMap (ours)\n 250m resolution"
)
ax.set_zlim(bottom=zmin, top=zmax)
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

# BEDMAP2
ax = fig.add_subplot(2, 2, 2, projection="3d")
ax = plot_3d_view(img=X_tile, ax=ax, cm_norm=norm_Z, title="BEDMAP2\n1000m resolution")
ax.set_zlim(bottom=zmin, top=zmax)
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

# DeepBedMap - BEDMAP2
ax = fig.add_subplot(2, 2, 3, projection="3d")
ax = plot_3d_view(
    img=Y_hat - cubicbedmap2,
    ax=ax,
    cm_norm=norm_Z,
    title="DeepBedMap minus\nBEDMAP2 (cubic interpolated)",
)
ax.set_zlim(bottom=-500, top=500)
ax.set_zlabel("\n\nDifference (metres)", fontsize=16)

# Synthetic
ax = fig.add_subplot(2, 2, 4, projection="3d")
ax = plot_3d_view(img=S_tile, ax=ax, cm_norm=norm_Z, title="Synthetic HRES ")
ax.set_zlim(bottom=zmin, top=zmax)
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

plt.subplots_adjust(wspace=0.0001, hspace=0.0001, left=0.0, right=0.9, top=1.2)
# plt.savefig(fname="esrgan_prediction.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %% [markdown]
# # Save Bicubic BEDMAP2 and ESRGAN DeepBedMap to a grid file

# %%
def save_array_to_grid(
    window_bound: tuple, array: np.ndarray, outfilepath: str, dtype: str = None
):
    """
    Saves a numpy array to geotiff and netcdf format.
    Appends ".tif" and ".nc" file extension to the outfilepath
    for geotiff and netcdf outputs respectively.
    """

    assert array.ndim == 4
    assert array.shape[1] == 1  # check that there is only one channel

    transform = rasterio.transform.from_bounds(
        *window_bound, height=array.shape[2], width=array.shape[3]
    )

    # Save array as a GeoTiff first
    with rasterio.open(
        f"{outfilepath}.tif",
        mode="w",
        driver="GTiff",
        height=array.shape[2],
        width=array.shape[3],
        count=1,
        crs="EPSG:3031",
        transform=transform,
        dtype=array.dtype if dtype is None else dtype,
        nodata=-2000,
    ) as new_geotiff:
        new_geotiff.write(array[0, 0, :, :], 1)

    # Convert deepbedmap3 and cubicbedmap2 from geotiff to netcdf format
    xr.open_rasterio(f"{outfilepath}.tif").to_netcdf(f"{outfilepath}.nc")


# %%
# Save BEDMAP3 to GeoTiff and NetCDF format
save_array_to_grid(
    window_bound=groundtruth.bounds, array=Y_hat, outfilepath="model/deepbedmap3"
)

# %%
# Save Bicubic Resampled BEDMAP2 to GeoTiff and NetCDF format
save_array_to_grid(
    window_bound=groundtruth.bounds, array=cubicbedmap2, outfilepath="model/cubicbedmap"
)

# %%
synthetic = skimage.transform.rescale(
    image=S_tile[0, 0, :, :].astype(np.int32),
    scale=1 / 2.5,  # 2.5 downscaling
    order=1,  # billinear interpolation
    mode="reflect",
    anti_aliasing=True,
    multichannel=False,
    preserve_range=True,
)
save_array_to_grid(
    window_bound=groundtruth.bounds,
    array=np.expand_dims(np.expand_dims(synthetic, axis=0), axis=0),
    outfilepath="model/synthetichr",
)

# %% [markdown]
# # Crossover analysis
#
# We use [grdtrack](https://gmt.soest.hawaii.edu/doc/latest/grdtrack) to sample our grid along the survey tracks.
#
# The survey tracks are basically geographic xy points flown by a plane.
# The three grids are all 250m spatial resolution, and they are:
#
# - Groundtruth grid (interpolated from our groundtruth points using [surface](https://gmt.soest.hawaii.edu/doc/latest/surface.html))
# - DeepBedMap3 grid (predicted from our [Super Resolution Generative Adversarial Network model](/srgan_train.ipynb))
# - CubicBedMap grid (interpolated from BEDMAP2 using a [bicubic spline algorithm](http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rescale))
# - Synthetic High Res grid (created by [Graham et al.](https://doi.org/10.5194/essd-9-267-2017))
#
# Reference:
#
# Wessel, P. (2010). Tools for analyzing intersecting tracks: The x2sys package. Computers & Geosciences, 36(3), 348–354. https://doi.org/10.1016/j.cageo.2009.05.009

# %%
test_filepath = "highres/2007tx"  # only one NetCDF grid can be tested
xyz_track = data_prep.ascii_to_xyz(pipeline_file=f"{test_filepath}.json")

# %%
## TODO make this multi-grid method work...
# track_test = pd.concat(
#     objs=[data_prep.ascii_to_xyz(pipeline_file=f"{pf}.json") for pf in test_filepaths]
# )
# grids = r"\n".join(f"{grid}.nc" for grid in test_filepaths)
# print(grids)
# !echo "{grids}" > tmp.txt
# !gmt grdtrack track_test.xyz -G+ltmp.txt -h1 -i0,1,2 > track_groundtruth.xyzi

# %%
df_groundtruth = gmt.grdtrack(
    points=xyz_track, grid=f"{test_filepath}.nc", newcolname="z_interpolated"
)
df_deepbedmap3 = gmt.grdtrack(
    points=xyz_track, grid="model/deepbedmap3.nc", newcolname="z_interpolated"
)
df_cubicbedmap = gmt.grdtrack(
    points=xyz_track, grid="model/cubicbedmap.nc", newcolname="z_interpolated"
)
df_synthetichr = gmt.grdtrack(
    points=xyz_track, grid="model/synthetichr.nc", newcolname="z_interpolated"
)

# %% [markdown]
# ### Get table statistics

# %%
df_groundtruth["error"] = df_groundtruth.z_interpolated - df_groundtruth.z
df_groundtruth.describe()

# %%
df_deepbedmap3["error"] = df_deepbedmap3.z_interpolated - df_deepbedmap3.z
df_deepbedmap3.describe()

# %%
df_cubicbedmap["error"] = df_cubicbedmap.z_interpolated - df_cubicbedmap.z
df_cubicbedmap.describe()

# %%
df_synthetichr["error"] = df_synthetichr.z_interpolated - df_synthetichr.z
df_synthetichr.describe()

# %%
# https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
rmse_groundtruth = (df_groundtruth.error ** 2).mean() ** 0.5
rmse_deepbedmap3 = (df_deepbedmap3.error ** 2).mean() ** 0.5
rmse_cubicbedmap = (df_cubicbedmap.error ** 2).mean() ** 0.5
rmse_synthetichr = (df_synthetichr.error ** 2).mean() ** 0.5
print(f"Difference      : {rmse_deepbedmap3 - rmse_cubicbedmap:.2f}")

bins = 50

fig, ax = plt.subplots(figsize=(16, 9))
# ax.set_yscale(value="symlog")
ax.set_xlim(left=-150, right=100)
df_groundtruth.hist(
    column="error",
    bins=25,
    ax=ax,
    histtype="step",
    label=f"Groundtruth RMSE: {rmse_groundtruth:.2f}",
)
df_deepbedmap3.hist(
    column="error",
    bins=25,
    ax=ax,
    histtype="step",
    label=f"DeepBedMap RMSE: {rmse_deepbedmap3:.2f}",
)
df_cubicbedmap.hist(
    column="error",
    bins=25,
    ax=ax,
    histtype="step",
    label=f"CubicBedMap RMSE: {rmse_cubicbedmap:.2f}",
)
"""
df_synthetichr.hist(
    column="error",
    bins=bins,
    ax=ax,
    histtype="step",
    label=f"SyntheticHR RMSE: {rmse_synthetichr:.2f}",
)
"""

ax.set_title("Elevation error between interpolated bed and groundtruth", fontsize=32)
ax.set_xlabel("Error in metres", fontsize=32)
ax.set_ylabel("Number of data points", fontsize=32)
ax.legend(loc="upper left", fontsize=32)
plt.tick_params(axis="both", labelsize=20)
plt.axvline(x=0)

# plt.savefig(fname="elevation_error_histogram.pdf", format="pdf", bbox_inches="tight")
plt.show()

# %%
# https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
rmse_groundtruth = (df_groundtruth.error ** 2).mean() ** 0.5
rmse_deepbedmap3 = (df_deepbedmap3.error ** 2).mean() ** 0.5
rmse_synthetichr = (df_synthetichr.error ** 2).mean() ** 0.5
rmse_cubicbedmap = (df_cubicbedmap.error ** 2).mean() ** 0.5
print(f"Groundtruth RMSE: {rmse_groundtruth}")
print(f"DeepBedMap3 RMSE: {rmse_deepbedmap3}")
print(f"SyntheticHR RMSE: {rmse_synthetichr}")
print(f"CubicBedMap RMSE: {rmse_cubicbedmap}")
print(f"Difference      : {rmse_deepbedmap3 - rmse_cubicbedmap}")

# %% [markdown]
# # **DeepBedMap** for the whole of Antarctica

# %%
# Bounding Box region in EPSG:3031 covering Antarctica
window_bound_big = rasterio.coords.BoundingBox(
    left=-2_700_000.0, bottom=-2_200_000.0, right=2_800_000.0, top=2_300_000.0
)
print(window_bound_big)

# %%
X_tile, W1_tile, W2_tile, W3_tile = get_deepbedmap_model_inputs(
    window_bound=window_bound_big
)

# %%
# Oh we will definitely need a GPU for this
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model.to_gpu()

# %%
# special zero padding for REMA, 5 pixels on top and bottom, 10 pixels on left and right
W1_tile = np.pad(
    array=W1_tile, pad_width=[(0, 0), (0, 0), (5, 5), (10, 10)], mode="constant"
)

# %%
print(X_tile.shape, W1_tile.shape, W2_tile.shape, W3_tile.shape)

# %% [markdown]
# ## The whole of Antarctica tiler and predictor!!
#
# Antarctica won't fit into our 16GB of GPU memory, so we have to:
#
# 1. Cut a 1000x1000 tile and load the data within this one small tile into GPU memory
# 2. Use our GPU-enabled model to make a prediction for this tile area
# 3. Repeat (1) and (2) for every tile we have covering Antarctica

# %%
# The whole of Antarctica tile and predictor
if 1 == 1:
    # Size are in kilometres
    final_shape = (18000, 22000)  # 4x that of BEDMAP2
    ary_height, ary_width = (1000, 1000)
    stride_height, stride_width = (1000, 1000)

    Y_hat = np.full(
        shape=(1, final_shape[0] + 20, final_shape[1] + 20),
        fill_value=np.nan,
        dtype=np.float32,
    )

    for y_step in range(0, final_shape[0], stride_height):
        for x_step in range(0, final_shape[1], stride_width):
            # plus 1 pixel on left and right
            x0, x1 = ((x_step // 4), ((x_step + ary_width) // 4) + 2)
            # plus 1 pixel on bottom and top
            y0, y1 = ((y_step // 4), ((y_step + ary_height) // 4) + 2)
            # x0, y0, x1, y1 = (3000,3000,3250,3250)

            X_tile_crop = cupy.asarray(a=X_tile[:, :, y0:y1, x0:x1], dtype="float32")
            W1_tile_crop = cupy.asarray(
                a=W1_tile[:, :, y0 * 10 : y1 * 10, x0 * 10 : x1 * 10], dtype="float32"
            )
            W2_tile_crop = cupy.asarray(
                a=W2_tile[:, :, y0 * 2 : y1 * 2, x0 * 2 : x1 * 2], dtype="float32"
            )
            W3_tile_crop = cupy.asarray(a=W3_tile[:, :, y0:y1, x0:x1], dtype="float32")

            Y_pred = model.forward(
                x=X_tile_crop, w1=W1_tile_crop, w2=W2_tile_crop, w3=W3_tile_crop
            )
            try:
                Y_hat[
                    :, (y0 * 4) + 4 : (y1 * 4) - 4, (x0 * 4) + 4 : (x1 * 4) - 4
                ] = cupy.asnumpy(Y_pred.array[0, :, :, :])
                # print(x0, y0, x1, y1)
            except ValueError:
                raise
            finally:
                X_tile_crop = W1_tile_crop = W2_tile_crop = W3_tile_crop = None

    Y_hat = Y_hat[:, 10:-10, 10:-10]

# %% [markdown]
# ## Save full map to file

# %%
# Save BEDMAP3 to GeoTiff and NetCDF format
# Using int16 instead of float32 to keep things smaller
save_array_to_grid(
    window_bound=window_bound_big,
    array=np.expand_dims(Y_hat.astype(dtype=np.int16), axis=0),
    outfilepath="model/deepbedmap3_big_int16",
    dtype=np.int16,
)

# %% [markdown]
# ## Show *the* DeepBedMap

# %%
with rasterio.open("model/deepbedmap3_big_int16.tif") as raster_tiff:
    deepbedmap_dem = raster_tiff.read(indexes=1, masked=True)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True, figsize=(12, 12))
rasterio.plot.show(
    source=np.ma.masked_less(x=deepbedmap_dem, value=-10000, copy=False),
    transform=raster_tiff.transform,
    cmap="BrBG_r",
    ax=ax,
)
