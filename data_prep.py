# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.0.3
#   kernelspec:
#     display_name: deepbedmap
#     language: python
#     name: deepbedmap
# ---

# %% [markdown]
# # Data Preparation
#
# Here in this data preparation jupyter notebook, we will prepare our data that will go into a Convolutional Neural Network model later.

# %% [markdown]
# ## 0. Setup parameters and load libraries

# %%
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import urllib

import tqdm
import yaml

import geopandas as gpd
import pygmt as gmt
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import quilt
import rasterio
import rasterio.mask
import rasterio.plot
import shapely.geometry
import skimage.util.shape
import xarray as xr

print("Python       :", sys.version.split("\n")[0])
print("Geopandas    :", gpd.__version__)
print("GMT          :", gmt.__version__)
print("Numpy        :", np.__version__)
print("Rasterio     :", rasterio.__version__)
print("Scikit-image :", skimage.__version__)
print("Xarray       :", xr.__version__)

# %% [markdown]
# ## 1. Get Data!

# %%
def download_to_path(path: str, url: str):
    r"""
    Download from a HTTP or FTP url to a filepath.

    >>> d = download_to_path(
    ...    path="highres/Data_20171204_02.csv",
    ...    url="ftp://data.cresis.ku.edu/data/rds/2017_Antarctica_Basler/csv_good/Data_20171204_02.csv",
    ... )
    >>> open("highres/Data_20171204_02.csv").readlines()
    ['LAT,LON,UTCTIMESOD,THICK,ELEVATION,FRAME,SURFACE,BOTTOM,QUALITY\n']
    >>> os.remove(path="highres/Data_20171204_02.csv")
    """

    folder, filename = os.path.split(p=path)
    downloaded_filename = os.path.basename(urllib.parse.urlparse(url=url).path)

    # Download file using URL first
    if not os.path.exists(os.path.join(folder, downloaded_filename)):
        r = urllib.request.urlretrieve(
            url=url, filename=os.path.join(folder, downloaded_filename)
        )

    # If downloaded file is not the final file (e.g. file is in an archive),
    # then extract the file from the archive!
    if filename != downloaded_filename:
        # Extract tar.gz archive file
        if downloaded_filename.endswith(("tgz", "tar.gz")):
            try:
                archive = tarfile.open(name=f"{folder}/{downloaded_filename}")
                archive.extract(member=filename, path=folder)
            except:
                raise
        else:
            raise ValueError(
                f"Unsupported archive format for downloaded file: {downloaded_filename}"
            )

    return os.path.exists(path=path)


# %%
def check_sha256(path: str):
    """
    Returns SHA256 checksum of a file

    >>> d = download_to_path(
    ...    path="highres/Data_20171204_02.csv",
    ...    url="https://data.cresis.ku.edu/data/rds/2017_Antarctica_Basler/csv_good/Data_20171204_02.csv",
    ... )
    >>> check_sha256("highres/Data_20171204_02.csv")
    '53cef7a0d28ff92b30367514f27e888efbc32b1bda929981b371d2e00d4c671b'
    >>> os.remove(path="highres/Data_20171204_02.csv")
    """
    with open(file=path, mode="rb") as afile:
        sha = hashlib.sha256(afile.read())

    return sha.hexdigest()


# %% [markdown]
# ## Parse [data_list.yml](/data_list.yml)

# %%
def parse_datalist(
    yaml_file: str = "data_list.yml",
    record_path: str = "files",
    schema: list = [
        "citekey",
        "folder",
        "location",
        "resolution",
        ["doi", "dataset"],
        ["doi", "literature"],
    ],
) -> pd.DataFrame:

    assert yaml_file.endswith((".yml", ".yaml"))

    with open(file=yaml_file, mode="r") as yml:
        y = yaml.safe_load(stream=yml)

    datalist = pd.io.json.json_normalize(
        data=y, record_path=record_path, meta=schema, sep="_"
    )

    return datalist


# %%
# Pretty print table with nice column order and clickable url links
pprint_table = lambda df, folder: IPython.display.HTML(
    df.query(expr="folder == @folder")
    .reindex(columns=["folder", "filename", "url", "sha256"])
    .style.format({"url": lambda url: f'<a target="_blank" href="{url}">{url}</a>'})
    .render(uuid=f"{folder}")
)
dataframe = parse_datalist()

# %%
# Code to autogenerate README.md files in highres/lowres/misc folders from data_list.yml
columns = ["Filename", "Location", "Resolution", "Literature Citation", "Data Citation"]
for folder, md_header in [
    ("lowres", "Low Resolution"),
    ("highres", "High Resolution"),
    ("misc", "Miscellaneous"),
]:
    assert folder in pd.unique(dataframe["folder"])
    md_name = f"{folder}/README.md"

    with open(file=md_name, mode="w") as md_file:
        md_file.write(f"# {md_header} Antarctic datasets\n\n")
        md_file.write("Note: This file was automatically generated from ")
        md_file.write("[data_list.yml](/data_list.yml) using ")
        md_file.write("[data_prep.ipynb](/data_prep.ipynb)\n\n")

    md_table = pd.DataFrame(columns=columns)
    md_table.loc[0] = ["---", "---", "---", "---", "---"]

    keydf = dataframe.groupby("citekey").aggregate(lambda x: set(x).pop())
    for row in keydf.query(expr="folder == @folder").itertuples():
        filecount = len(dataframe[dataframe["citekey"] == row.Index])
        extension = os.path.splitext(row.filename)[-1]
        row_dict = {
            "Filename": row.filename
            if filecount == 1
            else f"{filecount} *{extension} files",
            "Location": row.location,
            "Resolution": row.resolution,
            "Literature Citation": f"[{row.Index}]({row.doi_literature})",
            "Data Citation": f"[DOI]({row.doi_dataset})"
            if row.doi_dataset != "nan"
            else None,
        }
        md_table = md_table.append(other=row_dict, ignore_index=True)

    md_table.to_csv(path_or_buf=md_name, mode="a", sep="|", index=False)

# %% [markdown]
# ### Download Low Resolution bed elevation data (e.g. [BEDMAP2](https://doi.org/10.5194/tc-7-375-2013))

# %%
for dataset in dataframe.query(expr="folder == 'lowres'").itertuples():
    path = f"{dataset.folder}/{dataset.filename}"  # path to download the file to
    if not os.path.exists(path=path):
        download_to_path(path=path, url=dataset.url)
    assert check_sha256(path=path) == dataset.sha256
pprint_table(dataframe, "lowres")

# %%
with rasterio.open("lowres/bedmap2_bed.tif") as raster_source:
    rasterio.plot.show(source=raster_source, cmap="BrBG_r")

# %% [markdown]
# ### Download miscellaneous data (e.g. [REMA](https://doi.org/10.7910/DVN/SAIK8B), [MEaSUREs Ice Flow](https://doi.org/10.5067/D7GK8F5J8M8R), [LISA](https://doi.org/10.7265/nxpc-e997))

# %%
for dataset in dataframe.query(expr="folder == 'misc'").itertuples():
    path = f"{dataset.folder}/{dataset.filename}"  # path to download the file to
    if not os.path.exists(path=path):
        download_to_path(path=path, url=dataset.url)
    assert check_sha256(path=path) == dataset.sha256
pprint_table(dataframe, "misc")

# %% [markdown]
# ### Download High Resolution bed elevation data (e.g. some-DEM-name)

# %%
for dataset in dataframe.query(expr="folder == 'highres'").itertuples():
    path = f"{dataset.folder}/{dataset.filename}"  # path to download the file to
    if not os.path.exists(path=path):
        download_to_path(path=path, url=dataset.url)
    assert check_sha256(path=path) == dataset.sha256
pprint_table(dataframe, "highres")

# %% [markdown]
# ## 2. Process high resolution data into grid format
#
# Our processing step involves two stages:
#
# 1) Cleaning up the raw **vector** data, performing necessary calculations and reprojections to EPSG:3031.
#
# 2) Convert the cleaned vector data table via an interpolation function to a **raster** grid.

# %% [markdown]
# ### 2.1 [Raw ASCII Text](https://pdal.io/stages/readers.text.html) to [Clean XYZ table](https://gmt.soest.hawaii.edu/doc/latest/GMT_Docs.html#table-data)
#
# ![Raw ASCII to Clean Table via pipeline file](https://yuml.me/diagram/scruffy;dir:LR/class/[Raw-ASCII-Text|*.csv/*.txt]->[Pipeline-File|*.json],[Pipeline-File]->[Clean-XYZ-Table|*.xyz])

# %%
def ascii_to_xyz(pipeline_file: str) -> pd.DataFrame:
    """
    Converts ascii txt/csv files to xyz pandas.DataFrame via
    a JSON Pipeline file similar to the one used by PDAL.

    >>> os.makedirs(name="/tmp/highres", exist_ok=True)
    >>> d = download_to_path(
    ...    path="/tmp/highres/2011_Antarctica_TO.csv",
    ...    url="https://data.cresis.ku.edu/data/rds/2011_Antarctica_TO/csv_good/2011_Antarctica_TO.csv",
    ... )
    >>> _ = shutil.copy(src="highres/20xx_Antarctica_TO.json", dst="/tmp/highres")
    >>> df = ascii_to_xyz(pipeline_file="/tmp/highres/20xx_Antarctica_TO.json")
    >>> df.head(2)
                   x             y         z
    0  345580.826265 -1.156471e+06 -377.2340
    1  345593.322948 -1.156460e+06 -376.6332
    >>> shutil.rmtree(path="/tmp/highres")
    """
    assert os.path.exists(pipeline_file)
    assert pipeline_file.endswith((".json"))

    # Read json file first
    j = json.loads(open(pipeline_file).read())
    jdf = pd.io.json.json_normalize(j, record_path="pipeline")
    jdf = jdf.set_index(keys="type")
    reader = jdf.loc["readers.text"]  # check how to read the file(s)

    ## Basic table read
    skip = int(reader.skip)  # number of header rows to skip
    sep = reader.separator  # delimiter to use
    names = reader.header.split(sep=sep)  # header/column names as list
    usecols = reader.usecols.split(sep=sep)  # column names to use
    na_values = None if not hasattr(reader, "na_values") else reader.na_values

    path_pattern = os.path.join(os.path.dirname(pipeline_file), reader.filename)
    files = [file for file in glob.glob(path_pattern)]
    assert len(files) > 0  # check that there are actually files being matched!

    df = pd.concat(
        pd.read_csv(
            f, sep=sep, header=skip, names=names, usecols=usecols, na_values=na_values
        )
        for f in files
    )
    df.reset_index(drop=True, inplace=True)  # reset index after concatenation

    ## Advanced table read with conversions
    try:
        # Perform math operations
        newcol, expr = reader.converters.popitem()
        df[newcol] = df.eval(expr=expr)
        # Drop unneeded columns
        dropcols = reader.dropcols.split(sep=sep)
        df.drop(columns=dropcols, inplace=True)
    except AttributeError:
        pass

    assert len(df.columns) == 3  # check that we have 3 columns i.e. x, y, z
    df.sort_index(axis="columns", inplace=True)  # sort cols alphabetically
    df.set_axis(labels=["x", "y", "z"], axis="columns", inplace=True)  # lower case

    ## Reproject x and y coordinates if necessary
    try:
        reproject = jdf.loc["filters.reprojection"]
        p1 = pyproj.Proj(init=reproject.in_srs)
        p2 = pyproj.Proj(init=reproject.out_srs)
        reproj_func = lambda x, y: pyproj.transform(p1=p1, p2=p2, x=x, y=y)

        x2, y2 = reproj_func(np.array(df["x"]), np.array(df["y"]))
        df["x"] = pd.Series(x2)
        df["y"] = pd.Series(y2)

    except KeyError:
        pass

    return df


# %%
xyz_dict = {}
for pf in sorted(glob.glob("highres/*.json")):
    print(f"Processing {pf} pipeline", end=" ... ")
    name = os.path.splitext(os.path.basename(pf))[0]
    xyz_dict[name] = ascii_to_xyz(pipeline_file=pf)
    print(f"{len(xyz_dict[name])} datapoints")

# %% [markdown]
# ### 2.2 [Clean XYZ table](https://gmt.soest.hawaii.edu/doc/latest/GMT_Docs.html#table-data) to [Raster Grid](https://gmt.soest.hawaii.edu/doc/latest/GMT_Docs.html#grid-files)
#
# ![Clean XYZ Table to Raster Grid via interpolation function](https://yuml.me/diagram/scruffy;dir:LR/class/[Clean-XYZ-Table|*.xyz]->[Interpolation-Function],[Interpolation-Function]->[Raster-Grid|*.tif/*.nc])

# %%
def get_region(xyz_data: pd.DataFrame) -> str:
    """
    Gets the bounding box region of an xyz pandas.DataFrame in string
    format xmin/xmax/ymin/ymax rounded to 5 decimal places.
    Used for the -R 'region of interest' parameter in GMT.

    >>> xyz_data = pd.DataFrame(np.random.RandomState(seed=42).rand(30).reshape(10, 3))
    >>> get_region(xyz_data=xyz_data)
    '0.05808/0.83244/0.02058/0.95071'
    """
    xmin, ymin, _ = xyz_data.min(axis="rows")
    xmax, ymax, _ = xyz_data.max(axis="rows")
    return f"{xmin:.5f}/{xmax:.5f}/{ymin:.5f}/{ymax:.5f}"


# %%
def xyz_to_grid(
    xyz_data: pd.DataFrame,
    region: str,
    spacing: int = 250,
    tension: float = 0.35,
    outfile: str = None,
    mask_cell_radius: int = 3,
):
    """
    Performs interpolation of x, y, z point data to a raster grid.

    >>> xyz_data = 1000*pd.DataFrame(np.random.RandomState(seed=42).rand(60).reshape(20, 3))
    >>> region = get_region(xyz_data=xyz_data)
    >>> grid = xyz_to_grid(xyz_data=xyz_data, region=region, spacing=250)
    >>> grid.to_array().shape
    (1, 5, 5)
    >>> grid.to_array().values
    array([[[403.17618 , 544.92535 , 670.7824  , 980.75055 , 961.47723 ],
            [379.0757  , 459.26407 , 314.38297 , 377.78555 , 546.0469  ],
            [450.67664 , 343.26    ,  88.391594, 260.10492 , 452.3337  ],
            [586.09906 , 469.74008 , 216.8168  , 486.9802  , 642.2116  ],
            [451.4794  , 652.7244  , 325.77896 , 879.8973  , 916.7921  ]]],
          dtype=float32)
    """
    ## Preprocessing with blockmedian
    with gmt.helpers.GMTTempFile(suffix=".txt") as tmpfile:
        with gmt.clib.Session() as lib:
            file_context = lib.virtualfile_from_matrix(matrix=xyz_data.values)
            with file_context as infile:
                kwargs = {"V": "", "R": region, "I": f"{spacing}+e"}
                arg_str = " ".join(
                    [infile, gmt.helpers.build_arg_string(kwargs), "->" + tmpfile.name]
                )
                lib.call_module(module="blockmedian", args=arg_str)
            x, y, z = np.loadtxt(fname=tmpfile.name, unpack=True)

    ## XYZ point data to NetCDF grid via GMT surface
    grid = gmt.surface(
        x=x,
        y=y,
        z=z,
        region=region,
        spacing=f"{spacing}+e",
        T=tension,
        V="",
        M=f"{mask_cell_radius}c",
    )

    ## Save grid to NetCDF with projection information
    if outfile is not None:
        grid.to_netcdf(path=outfile)  ##TODO add CRS!!

    return grid


# %%
grid_dict = {}
for name in xyz_dict.keys():
    print(f"Gridding {name}", end=" ... ")
    xyz_data = xyz_dict[name]
    region = get_region(xyz_data)
    grid_dict[name] = xyz_to_grid(
        xyz_data=xyz_data, region=region, outfile=f"highres/{name}.nc"
    )
    print(f"done! {grid_dict[name].to_array().shape}")

# %% [markdown]
# ### 2.3 Plot raster grids

# %%
grids = sorted(glob.glob("highres/*.nc"))
fig, axarr = plt.subplots(
    nrows=1 + ((len(grids) - 1) // 3), ncols=3, squeeze=False, figsize=(15, 15)
)

for i, grid in enumerate(grids):
    with rasterio.open(grid) as raster_source:
        rasterio.plot.show(
            source=raster_source, cmap="BrBG_r", ax=axarr[i // 3, i % 3], title=grid
        )

# %% [markdown]
# ## 3. Tile data

# %% [markdown]
# ### Big raster to many small square tiles

# %%
def get_window_bounds(
    filepath: str, height: int = 32, width: int = 32, step: int = 4
) -> list:
    """
    Reads in a raster and finds tiles for them according to a stepped moving window.
    Returns a list of bounding box coordinates corresponding to a tile that looks like
    [(minx, miny, maxx, maxy), (minx, miny, maxx, maxy), ...]

    >>> xr.DataArray(
    ...     data=np.zeros(shape=(36, 32)),
    ...     coords={"x": np.arange(1, 37), "y": np.arange(1, 33)},
    ...     dims=["x", "y"],
    ... ).to_netcdf(path="/tmp/tmp_wb.nc")
    >>> get_window_bounds(filepath="/tmp/tmp_wb.nc")
    Tiling: /tmp/tmp_wb.nc ... 2
    [(0.5, 4.5, 32.5, 36.5), (0.5, 0.5, 32.5, 32.5)]
    >>> os.remove("/tmp/tmp_wb.nc")
    """
    assert height == width  # make sure it's a square!
    assert height % 2 == 0  # make sure we are passing in an even number

    with xr.open_rasterio(filepath) as dataset:
        print(f"Tiling: {filepath} ... ", end="")
        # Vectorized 'loop' along the raster image from top to bottom, and left to right

        # Get boolean true/false mask of where the data/nodata pixels lie
        mask = dataset.to_masked_array(copy=False).mask
        mask = mask[0, :, :]  # change to shape (height, width)

        # Sliding window view of the input geographical raster image
        window_views = skimage.util.shape.view_as_windows(
            arr_in=mask, window_shape=(height, width), step=step
        )
        filled_tiles = ~window_views.any(
            axis=(-2, -1)
        )  # find tiles which are fully filled, i.e. no blank/NODATA pixels
        tile_indexes = np.argwhere(filled_tiles)  # get x and y index of filled tiles

        # Convert x,y tile indexes to bounding box coordinates
        windows = [
            rasterio.windows.Window(
                col_off=ulx * step, row_off=uly * step, width=width, height=height
            )
            for uly, ulx in tile_indexes
        ]
        window_bounds = [
            rasterio.windows.bounds(
                window=window,
                transform=rasterio.Affine(*dataset.transform),
                width=width,
                height=height,
            )
            for window in windows
        ]
        print(len(window_bounds))

    return window_bounds


# %%
filepaths = sorted([g for g in glob.glob("highres/*.nc") if g != "highres/2007tx.nc"])
window_bounds = [get_window_bounds(filepath=grid) for grid in filepaths]
window_bounds_concat = np.concatenate([w for w in window_bounds]).tolist()
print(f"Total number of tiles: {len(window_bounds_concat)}")

# %% [markdown]
# ### Show and save tiles

# %%
gdf = pd.concat(
    objs=[
        gpd.GeoDataFrame(
            pd.Series(
                data=len(window_bound) * [os.path.basename(filepath)], name="grid_name"
            ),
            crs={"init": "epsg:3031"},
            geometry=[shapely.geometry.box(*bound) for bound in window_bound],
        )
        for filepath, window_bound in zip(filepaths, window_bounds)
    ]
).reset_index(drop=True)
gdf.plot()

# %%
gdf.to_file(filename="model/train/tiles_3031.geojson", driver="GeoJSON")
gdf.to_crs(crs={"init": "epsg:4326"}).to_file(
    filename="model/train/tiles_4326.geojson", driver="GeoJSON"
)

# %% [markdown]
# ### Do the actual tiling

# %%
def selective_tile(
    filepath: str,
    window_bounds: list,
    padding: int = 0,
    out_shape: tuple = None,
    gapfill_raster_filepath: str = None,
) -> np.ndarray:
    """
    Reads in raster and tiles them selectively.
    Tiles will go according to list of window_bounds.
    Output shape can be set to e.g. (16,16) to resample input raster to
    some desired shape/resolution.

    >>> xr.DataArray(
    ...     data=np.random.RandomState(seed=42).rand(64).reshape(8, 8),
    ...     coords={"x": np.arange(8), "y": np.arange(8)},
    ...     dims=["x", "y"],
    ... ).to_netcdf(path="/tmp/tmp_st.nc", mode="w")
    >>> selective_tile(
    ...    filepath="/tmp/tmp_st.nc",
    ...    window_bounds=[(1.0, 4.0, 3.0, 6.0), (2.0, 5.0, 4.0, 7.0)],
    ... )
    Tiling: /tmp/tmp_st.nc ... done!
    array([[[[0.18485446, 0.96958464],
             [0.4951769 , 0.03438852]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[0.04522729, 0.32533032],
             [0.96958464, 0.77513283]]]], dtype=float32)
    >>> os.remove("/tmp/tmp_st.nc")
    """
    array_list = []

    with rasterio.open(filepath) as dataset:
        print(f"Tiling: {filepath} ... ", end="")
        for window_bound in window_bounds:

            if padding > 0:
                window_bound = (
                    window_bound[0] - padding,  # minx
                    window_bound[1] - padding,  # miny
                    window_bound[2] + padding,  # maxx
                    window_bound[3] + padding,  # maxy
                )

            window = rasterio.windows.from_bounds(
                *window_bound, transform=dataset.transform, precision=None
            ).round_offsets()

            # Read the raster according to the crop window
            array = dataset.read(
                indexes=list(range(1, dataset.count + 1)),
                masked=True,
                window=window,
                out_shape=out_shape,
            )
            assert array.ndim == 3  # check that we have shape like (1, height, width)
            assert array.shape[0] == 1  # channel-first (assuming only 1 channel)

            try:
                assert not array.mask.any()  # check that there are no NAN values
            except AssertionError:
                # Replace pixels from another raster if available, else raise error
                if gapfill_raster_filepath is not None:
                    with rasterio.open(gapfill_raster_filepath) as dataset2:
                        window2 = rasterio.windows.from_bounds(
                            *window_bound, transform=dataset2.transform, precision=None
                        ).round_offsets()

                        array2 = dataset2.read(
                            indexes=list(range(1, dataset2.count + 1)),
                            masked=True,
                            window=window2,
                            out_shape=array.shape[1:],
                        )

                    np.copyto(
                        dst=array, src=array2, where=array.mask
                    )  # fill in gaps where mask is True

                    assert (
                        not array.mask.any()
                    )  # check that there are no NAN values after gapfill
                else:
                    plt.imshow(array.data[0, :, :])
                    plt.show()
                    raise ValueError(
                        f"Tile has missing data, try passing in gapfill_raster_filepath"
                    )

            # assert array.shape[0] == array.shape[1]  # check that height==width
            array_list.append(array.data.astype(dtype=np.float32))
        print("done!")

    return np.stack(arrays=array_list)


# %%
geodataframe = gpd.read_file("model/train/tiles_3031.geojson")
filepaths = geodataframe.grid_name.unique()
window_bounds = [
    [geom.bounds for geom in geodataframe.query("grid_name == @filepath").geometry]
    for filepath in filepaths
]
window_bounds_concat = np.concatenate([w for w in window_bounds]).tolist()

# %% [markdown]
# ### Tile High Resolution data

# %%
hireses = [
    selective_tile(filepath=f"highres/{f}", window_bounds=w)
    for f, w in zip(filepaths, window_bounds)
]
hires = np.concatenate(hireses)
print(hires.shape, hires.dtype)

# %% [markdown]
# ### Tile low resolution data

# %%
lores = selective_tile(
    filepath="lowres/bedmap2_bed.tif", window_bounds=window_bounds_concat, padding=1000
)
print(lores.shape, lores.dtype)

# %% [markdown]
# ### Tile miscellaneous data

# %%
rema = selective_tile(
    filepath="misc/REMA_100m_dem.tif",
    window_bounds=window_bounds_concat,
    padding=1000,
    gapfill_raster_filepath="misc/REMA_200m_dem_filled.tif",
)
print(rema.shape, rema.dtype)

# %%
## Custom processing for LISA to standardize units with MEASURES Ice Velocity
# Convert units from metres/day to metres/year by multiplying 1st band by 365.25
!rio calc "(* 365.25 (read 1))" misc/lisa750_2013182_2017120_0000_0400_vv_v1.tif misc/lisa750_2013182_2017120_0000_0400_vv_v1_myr.tif
# Set NODATA mask where pixels are 36159.75 = 99 * 365.25
!rio edit-info misc/lisa750_2013182_2017120_0000_0400_vv_v1_myr.tif --nodata 36159.75
!rio info misc/lisa750_2013182_2017120_0000_0400_vv_v1_myr.tif

# %%
measuresiceflow = selective_tile(
    filepath="misc/MEaSUREs_IceFlowSpeed_450m.tif",
    window_bounds=window_bounds_concat,
    padding=1000,
    out_shape=(20, 20),
    # gapfill_raster_filepath="misc/lisa750_2013182_2017120_0000_0400_vv_v1_myr.tif",
)
print(measuresiceflow.shape, measuresiceflow.dtype)

# %% [markdown]
# ## 4. Save the arrays
#
# We'll save the numpy arrays to the filesystem first.
# We label inputs as X (low resolution bed DEMs) and W (miscellaneous).
# Groundtruth high resolution bed DEMs are labelled as Y.
#
# Also, we'll serve the data up on the web using:
# - [Quilt](https://quiltdata.com/) - Python data versioning
# - [Dat](https://datproject.org/) - Distributed data sharing (TODO)

# %%
os.makedirs(name="model/train", exist_ok=True)
np.save(file="model/train/W1_data.npy", arr=rema)
np.save(file="model/train/W2_data.npy", arr=measuresiceflow)
np.save(file="model/train/X_data.npy", arr=lores)
np.save(file="model/train/Y_data.npy", arr=hires)

# %% [markdown]
# ### Quilt
#
# Login -> Build -> Push

# %%
quilt.login()

# %%
# Tiled datasets for training neural network
quilt.build(package="weiji14/deepbedmap/model/train/W1_data", path=rema)
quilt.build(package="weiji14/deepbedmap/model/train/W2_data", path=measuresiceflow)
quilt.build(package="weiji14/deepbedmap/model/train/X_data", path=lores)
quilt.build(package="weiji14/deepbedmap/model/train/Y_data", path=hires)

# Original datasets for neural network predictions on bigger area
quilt.build(
    package="weiji14/deepbedmap/lowres/bedmap2_bed", path="lowres/bedmap2_bed.tif"
)
quilt.build(
    package="weiji14/deepbedmap/misc/REMA_100m_dem", path="misc/REMA_100m_dem.tif"
)
quilt.build(
    package="weiji14/deepbedmap/misc/REMA_200m_dem_filled",
    path="misc/REMA_200m_dem_filled.tif",
)
quilt.build(
    package="weiji14/deepbedmap/misc/MEaSUREs_IceFlowSpeed_450m",
    path="misc/MEaSUREs_IceFlowSpeed_450m.tif",
)

# %%
quilt.push(package="weiji14/deepbedmap", is_public=True)
