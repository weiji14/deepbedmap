# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: deepbedmap
#     language: python
#     name: deepbedmap
# ---

# %% [markdown]
# # DeepBedMap
#
# Predicting the bed elevation of Antarctica using a Super Resolution Deep Neural Network.

# %%
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import comet_ml
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import quilt
import rasterio
import skimage
import xarray as xr

import keras

from features.environment import _load_ipynb_modules

# %% [markdown]
# ## Get bounding box of area we want to predict on

# %%
def get_image_and_bounds(filepath: str):
    """
    Retrieve raster image in numpy array format and
    geographic bounds as (xmin, ymin, xmax, ymax)
    """
    with xr.open_dataset(filepath) as data:
        groundtruth = data.z.to_masked_array()
        groundtruth = np.flipud(groundtruth)  # flip on y-axis...
        groundtruth = np.expand_dims(
            np.expand_dims(groundtruth, axis=-1), axis=0
        )  # add extra dimensions (batch and channel)

        xmin, xmax = float(data.x.min()), float(data.x.max())
        ymin, ymax = float(data.y.min()), float(data.y.max())

        window_bound = rasterio.coords.BoundingBox(
            left=xmin, bottom=ymin, right=xmax, top=ymax
        )
    return groundtruth, window_bound


# %%
test_file = "2007tx"  # "istarxx"
test_filepath = f"highres/{test_file}"
groundtruth, window_bound = get_image_and_bounds(filepath=f"{test_filepath}.nc")
print(window_bound)

# %% [markdown]
# ## Get neural network input datasets for our area of interest

# %%
def get_deepbedmap_model_inputs(
    window_bound: rasterio.coords.BoundingBox, padding=1000
):
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
        padding=padding,
    )
    W1_tile = data_prep.selective_tile(
        filepath="misc/REMA_100m_dem.tif",
        window_bounds=[[*window_bound]],
        gapfill_raster_filepath="misc/REMA_200m_dem_filled.tif",
        padding=padding,
    )
    W2_tile = data_prep.selective_tile(
        filepath="misc/MEaSUREs_IceFlowSpeed_450m.tif",
        window_bounds=[[*window_bound]],
        out_shape=(2 * X_tile.shape[1], 2 * X_tile.shape[2]),
        padding=padding,
    )

    return X_tile, W1_tile, W2_tile


# %%
def plot_3d_view(
    img: np.ndarray,
    ax: matplotlib.axes._subplots.Axes,
    elev: int = 60,
    azim: int = 330,
    cm_norm: matplotlib.colors.Normalize = None,
    title: str = None,
):
    # Get x, y, z data
    image = img[0, :, :, :]
    xx, yy = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    zz = image[:, :, 0]

    # Make the 3D plot
    ax.view_init(elev=elev, azim=azim)
    ax.plot_surface(xx, yy, zz, cmap="BrBG", norm=cm_norm)
    ax.set_title(label=f"{title}\n", fontsize=22)

    return ax


# %%
X_tile, W1_tile, W2_tile = get_deepbedmap_model_inputs(window_bound=window_bound)

# Build quilt package for datasets covering our test region
reupload = False
if reupload == True:
    quilt.build(package="weiji14/deepbedmap/model/test/W1_tile", path=W1_tile)
    quilt.build(package="weiji14/deepbedmap/model/test/W2_tile", path=W2_tile)
    quilt.build(package="weiji14/deepbedmap/model/test/X_tile", path=X_tile)
    quilt.push(package="weiji14/deepbedmap/model/test", is_public=True)

# %%
fig, axarr = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(16, 12))
axarr[0, 0].imshow(X_tile[0, :, :, 0], cmap="BrBG")
axarr[0, 0].set_title("BEDMAP2\n(1000m resolution)")
axarr[0, 1].imshow(W1_tile[0, :, :, 0], cmap="BrBG")
axarr[0, 1].set_title("Reference Elevation Model of Antarctica\n(100m resolution)")
axarr[0, 2].imshow(W2_tile[0, :, :, 0], cmap="BrBG")
axarr[0, 2].set_title("MEaSUREs Ice Velocity\n(450m, resampled to 500m)")
plt.show()

# %%
fig = plt.figure(figsize=plt.figaspect(1 / 3) * 2.5)

ax = fig.add_subplot(1, 3, 1, projection="3d")
ax = plot_3d_view(img=X_tile, ax=ax, title="BEDMAP2\n(1000m resolution)")
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

ax = fig.add_subplot(1, 3, 2, projection="3d")
ax = plot_3d_view(
    img=W1_tile,
    ax=ax,
    title="Reference Elevation Model of Antarctica\n(100m resolution)",
)
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

ax = fig.add_subplot(1, 3, 3, projection="3d")
ax = plot_3d_view(
    img=W2_tile, ax=ax, title="MEaSUREs Surface Ice Velocity\n(450m, resampled to 500m)"
)
ax.set_zlabel("\n\nSurface Ice Velocity (metres/year)", fontsize=16)

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
def load_trained_model(model_inputs: tuple):
    """
    Creates a custom DeepBedMap neural network model
    according to the shapes of the raster image inputs.

    Also loads trained parameter weights into the model.
    """
    srgan_train = _load_ipynb_modules("srgan_train.ipynb")

    X_tile, W1_tile, W2_tile = model_inputs

    network = srgan_train.generator_network(
        input1_shape=X_tile.shape[1:],
        input2_shape=W1_tile.shape[1:],
        input3_shape=W2_tile.shape[1:],
    )

    model = keras.models.Model(
        inputs=network.inputs, outputs=network.outputs, name="generator_model"
    )

    # Load trained neural network weights into model
    model.load_weights(filepath="model/weights/srgan_generator_model_weights.hdf5")

    return model


# %% [markdown]
# ## Make prediction

# %%
model = load_trained_model(model_inputs=(X_tile, W1_tile, W2_tile))
Y_hat = model.predict(x=[X_tile, W1_tile, W2_tile], verbose=1)
Y_hat.shape

# %% [markdown]
# ## Plot results

# %%
fig, axarr = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(16, 12))
axarr[0, 0].imshow(X_tile[0, :, :, 0], cmap="BrBG")
axarr[0, 0].set_title("BEDMAP2")
axarr[0, 1].imshow(Y_hat[0, :, :, 0], cmap="BrBG")
axarr[0, 1].set_title("Super Resolution Generative Adversarial Network prediction")
axarr[0, 2].imshow(groundtruth[0, :, :, 0], cmap="BrBG")
axarr[0, 2].set_title("Groundtruth grids")
plt.show()

# %%
fig = plt.figure(figsize=plt.figaspect(1 / 2) * 2.5)

zmin, zmax = (X_tile.min(), X_tile.max())
norm_Z = matplotlib.cm.colors.Normalize(vmin=zmin, vmax=zmax)

ax = fig.add_subplot(1, 2, 1, projection="3d")
ax = plot_3d_view(
    img=X_tile, ax=ax, cm_norm=norm_Z, title="BEDMAP2\n(1000m resolution)"
)
ax.set_zlim(bottom=zmin, top=zmax)
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

ax = fig.add_subplot(1, 2, 2, projection="3d")
ax = plot_3d_view(
    img=Y_hat,
    ax=ax,
    cm_norm=norm_Z,
    title="Super Resolution Generative Adversarial Network prediction\n(250m resolution)",
)
ax.set_zlim(bottom=zmin, top=zmax)
ax.set_zlabel("\n\nElevation (metres)", fontsize=16)

plt.show()

# %% [markdown]
# # Save Bicubic BEDMAP2 and SRGAN DEEPBEDMAP to a grid file

# %%
def save_array_to_grid(window_bound: tuple, array: np.ndarray, outfilepath: str):
    """
    Saves a numpy array to geotiff and netcdf format
    """

    assert array.ndim == 4
    assert array.shape[3] == 1  # check that there is only one channel

    transform = rasterio.transform.from_bounds(
        *window_bound, height=array.shape[1], width=array.shape[2]
    )

    # Save array as a GeoTiff first
    with rasterio.open(
        f"{outfilepath}.tif",
        mode="w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=1,
        crs="EPSG:3031",
        transform=transform,
        dtype=array.dtype,
    ) as new_geotiff:
        new_geotiff.write(array[0, :, :, 0], 1)

    # Convert deepbedmap3 and cubicbedmap2 from geotiff to netcdf format
    xr.open_rasterio(f"{outfilepath}.tif").to_netcdf(f"{outfilepath}.nc")


# %%
# Save BEDMAP3 to GeoTiff and NetCDF format
save_array_to_grid(
    window_bound=window_bound, array=Y_hat, outfilepath="model/deepbedmap3"
)

# %%
# Save Bicubic Resampled BEDMAP2 to GeoTiff and NetCDF format
cubicbedmap2 = skimage.transform.rescale(
    image=X_tile[0].astype(np.int32),
    scale=4,
    order=3,
    mode="reflect",
    anti_aliasing=True,
    multichannel=True,
    preserve_range=True,
)
save_array_to_grid(
    window_bound=window_bound,
    array=np.expand_dims(cubicbedmap2, axis=0),
    outfilepath="model/cubicbedmap",
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
#
# Reference:
#
# Wessel, P. (2010). Tools for analyzing intersecting tracks: The x2sys package. Computers & Geosciences, 36(3), 348–354. https://doi.org/10.1016/j.cageo.2009.05.009

# %%
data_prep = _load_ipynb_modules("data_prep.ipynb")
track_test = data_prep.ascii_to_xyz(pipeline_file=f"{test_filepath}.json")
track_test.to_csv("track_test.xyz", sep="\t", index=False)

# %%
!gmt grdtrack track_test.xyz -G{test_filepath}.nc -h1 -i0,1,2 > track_groundtruth.xyzi
!gmt grdtrack track_test.xyz -Gmodel/deepbedmap3.nc -h1 -i0,1,2 > track_deepbedmap3.xyzi
!gmt grdtrack track_test.xyz -Gmodel/cubicbedmap.nc -h1 -i0,1,2 > track_cubicbedmap.xyzi
!head track_*.xyzi -n5

# %% [markdown]
# ### Get table statistics

# %%
df_groundtruth = pd.read_table(
    "track_groundtruth.xyzi", header=1, names=["x", "y", "z", "z_interpolated"]
)
df_deepbedmap3 = pd.read_table(
    "track_deepbedmap3.xyzi", header=1, names=["x", "y", "z", "z_interpolated"]
)
df_cubicbedmap = pd.read_table(
    "track_cubicbedmap.xyzi", header=1, names=["x", "y", "z", "z_interpolated"]
)

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
# https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
rmse_groundtruth = (df_groundtruth.error ** 2).mean() ** 0.5
rmse_deepbedmap3 = (df_deepbedmap3.error ** 2).mean() ** 0.5
rmse_cubicbedmap = (df_cubicbedmap.error ** 2).mean() ** 0.5
print(f"Difference      : {rmse_deepbedmap3 - rmse_cubicbedmap:.2f}")

fig, ax = plt.subplots(figsize=(16, 9))
df_groundtruth.hist(
    column="error",
    bins=50,
    ax=ax,
    histtype="step",
    label=f"Groundtruth RMSE: {rmse_groundtruth:.2f}",
)
df_deepbedmap3.hist(
    column="error",
    bins=50,
    ax=ax,
    histtype="step",
    label=f"DeepBedMap3 RMSE: {rmse_deepbedmap3:.2f}",
)
df_cubicbedmap.hist(
    column="error",
    bins=50,
    ax=ax,
    histtype="step",
    label=f"CubicBedMap RMSE: {rmse_cubicbedmap:.2f}",
)
ax.set_title(
    "Elevation error between interpolated bed (Z_interpolated) and groundtruth bed (Z)",
    fontsize=22,
)
ax.set_xlabel("Error in metres", fontsize=16)
ax.set_ylabel("Number of data points", fontsize=16)
ax.legend(loc="upper left", fontsize=16)
plt.show()

# %%
# https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
rmse_groundtruth = (df_groundtruth.error ** 2).mean() ** 0.5
rmse_deepbedmap3 = (df_deepbedmap3.error ** 2).mean() ** 0.5
rmse_cubicbedmap = (df_cubicbedmap.error ** 2).mean() ** 0.5
print(f"Groundtruth RMSE: {rmse_groundtruth}")
print(f"DeepBedMap3 RMSE: {rmse_deepbedmap3}")
print(f"CubicBedMap RMSE: {rmse_cubicbedmap}")
print(f"Difference      : {rmse_deepbedmap3 - rmse_cubicbedmap}")
