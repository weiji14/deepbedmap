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
import tempfile
import urllib
import yaml
import zipfile

# need to import before rasterio
import xarray as xr
import salem

import dask
import dask.diagnostics
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
import tqdm

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
        # Extract from .zip archive file
        elif downloaded_filename.endswith((".zip")):
            try:
                archive = zipfile.ZipFile(file=f"{folder}/{downloaded_filename}")
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
# ### Download miscellaneous data (e.g. [REMA](https://doi.org/10.7910/DVN/SAIK8B), [MEaSUREs phase-based Ice Velocity](https://doi.org/10.5067/PZ3NJ5RXRH10), [Arthern Accumulation](https://doi.org/10.1029/2004JD005667))

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
    jdf = pd.io.json.json_normalize(data=j, record_path="pipeline", max_level=0)
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
    df.dropna(axis="index", inplace=True)  # drop rows with NaN values
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
        p1 = pyproj.CRS.from_string(in_crs_string=reproject.in_srs)
        p2 = pyproj.CRS.from_string(in_crs_string=reproject.out_srs)
        reprj_func = pyproj.Transformer.from_crs(crs_from=p1, crs_to=p2, always_xy=True)

        x2, y2 = reprj_func.transform(xx=np.array(df["x"]), yy=np.array(df["y"]))
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
def get_region(xyz_data: pd.DataFrame, round_increment: int = 250) -> str:
    """
    Gets an extended bounding box region for points in an xyz pandas.DataFrame with
    columns x, y, and z. The coordinates will be rounded to values specified by the
    round_increment parameter. Implementation uses gmt.info with the -I (increment)
    setting, see also https://gmt.soest.hawaii.edu/doc/latest/gmtinfo.html#i

    The output region is returned in a string format 'xmin/xmax/ymin/ymax' directly
    usable as the -R 'region of interest' parameter in GMT. Indeed, the rounding is
    specifically optimized to give grid dimensions for fastest results in programs like
    GMT surface.

    >>> xyz_data = pd.DataFrame(
    ...     10000 * np.random.RandomState(seed=42).rand(30).reshape(10, 3),
    ...     columns=["x", "y", "z"],
    ... )
    >>> get_region(xyz_data=xyz_data)
    '-250/9500/0/9750'
    """
    assert (xyz_data.columns == pd.Index(data=["x", "y", "z"], dtype="object")).all()

    with tempfile.NamedTemporaryFile(suffix=".csv") as tmpfile:
        xyz_data.to_csv(tmpfile.name, header=False, index=False)
        region = gmt.info(fname=tmpfile.name, I=f"s{round_increment}").strip()[2:]

    return region


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

    >>> xyz_data = pd.DataFrame(
    ...     600 * np.random.RandomState(seed=42).rand(60).reshape(20, 3),
    ...     columns=["x", "y", "z"],
    ... )
    >>> region = get_region(xyz_data=xyz_data)
    >>> grid = xyz_to_grid(xyz_data=xyz_data, region=region, spacing=250)
    >>> grid.to_array().shape
    (1, 3, 3)
    >>> grid.to_array().values
    array([[[208.90086, 324.8038 , 515.93726],
            [180.06642, 234.68915, 452.8586 ],
            [170.60728, 298.23764, 537.49774]]], dtype=float32)
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
        V="n",  # normal verbosity: produce only fatal error messages
        M=f"{mask_cell_radius}c",
    )

    ## Save grid to NetCDF with projection information
    if outfile is not None:
        # TODO add CRS!! See https://github.com/pydata/xarray/issues/2288
        grid.to_netcdf(path=outfile)

    ## Resample grid from gridline to pixel registration
    with gmt.helpers.GMTTempFile(suffix=".nc") as tmpfile:
        with gmt.clib.Session() as lib:
            if outfile is not None:  # kind == "file"
                file_context = gmt.helpers.dummy_context(outfile)
            else:  # kind == "grid"
                file_context = lib.virtualfile_from_grid(grid.z)
                outfile = tmpfile.name
            with file_context as infile:
                kwargs = {"T": "", "G": f"{outfile}"}
                arg_str = " ".join([infile, gmt.helpers.build_arg_string(kwargs)])
                lib.call_module(module="grdsample", args=arg_str)
            with xr.open_dataset(outfile) as dataset:
                grid = dataset.load()

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
    print(f"done! {grid_dict[name].z.shape}")

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
    filepath: str,
    pyproj_srs: str = "epsg:3031",
    height: int = 36,  # 36 pixels * 250m resolution = 9000m height
    width: int = 36,  # 36 pixels * 250m resolution = 9000m width
    step: int = 3,  # 3 pixels * 250m resolution = 750m step size
) -> list:
    """
    Reads in a raster and finds tiles for them according to a stepped moving window.
    Returns a list of bounding box coordinates corresponding to a tile that looks like
    [(minx, miny, maxx, maxy), (minx, miny, maxx, maxy), ...]

    >>> xr.DataArray(
    ...     data=np.zeros(shape=(40, 36)),
    ...     coords={"y": np.arange(0.5, 40.5), "x": np.arange(0.5, 36.5)},
    ...     dims=["y", "x"],
    ... ).to_netcdf(path="/tmp/tmp_wb.nc")
    >>> get_window_bounds(filepath="/tmp/tmp_wb.nc")
    Tiling: /tmp/tmp_wb.nc ... 2
    [(0.0, 4.0, 36.0, 40.0), (0.0, 1.0, 36.0, 37.0)]
    >>> os.remove("/tmp/tmp_wb.nc")
    """
    assert height == width  # make sure it's a square!
    assert height % 2 == 0  # make sure we are passing in an even number

    with xr.open_dataarray(filepath) as dataset:
        print(f"Tiling: {filepath} ... ", end="")

        # Use salem to patch projection information into xarray.DataArray
        # See also https://salem.readthedocs.io/en/latest/xarray_acc.html
        dataset.attrs["pyproj_srs"] = pyproj_srs
        sgrid = dataset.salem.grid.corner_grid
        assert sgrid.origin == "lower-left"  # should be "lower-left", not "upper-left"

        ## Vectorized 'loop' along raster image from top to bottom, and left to right

        # Get boolean true/false mask of where the data/nodata pixels lie
        mask = dataset.to_masked_array(copy=False).mask
        mask = np.ascontiguousarray(a=np.flipud(m=mask))  # flip on y-axis

        # Sliding window view of the input geographical raster image
        window_views = skimage.util.shape.view_as_windows(
            arr_in=mask, window_shape=(height, width), step=step
        )
        filled_tiles = ~window_views.any(
            axis=(-2, -1)
        )  # find tiles which are fully filled, i.e. no blank/NODATA pixels
        tile_indexes = np.argwhere(a=filled_tiles)  # get x and y index of filled tiles

        # Convert x,y tile indexes to bounding box coordinates
        # Complicated as xarray uses centre-based coordinates,
        # while rasterio uses corner-based coordinates
        windows = [
            rasterio.windows.Window(
                col_off=ulx * step, row_off=uly * step, width=width, height=height
            )
            for uly, ulx in tile_indexes
        ]
        window_bounds = [
            rasterio.windows.bounds(
                window=window,
                transform=rasterio.Affine(
                    sgrid.dx, 0, sgrid.x0, 0, -sgrid.dy, sgrid.y_coord[-1] + sgrid.dy
                ),
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
# ### Subset tiles to those within grounding line, plot to show, and save

# %%
tile_gdf = pd.concat(
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

# %%
# Load grounding line polygon and buffer by 10km
gline = gpd.read_file("misc/GroundingLine_Antarctica_v2.shp")
gline.crs = {"init": "epsg:3031"}
gline.geometry = gline.geometry.buffer(distance=10000)

# %%
# Select tiles within the buffered grounding line
gdf = gpd.sjoin(left_df=tile_gdf, op="within", right_df=gline, how="inner")
gdf = gdf.reset_index()[["grid_name", "geometry"]]
gdf.plot()

# %%
# Save subsetted tiles to file in both EPSG 3031 and 4326
print(f"Saving only {len(gdf)} tiles out of {len(tile_gdf)}")
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
    padding: int = 0,  # in projected coordinate system units, e.g. 1000 for 1km
    resolution: float = None,  # spatial resolution, e.g. 500 for 500m
    gapfiller: float = None,  # number to fill in NaNs, e.g. 0
    interpolate: bool = True,  # resample grid correct bounds, else just use slicing
) -> np.ndarray:
    """
    Reads in a raster and tiles them selectively according to list of window_bounds in
    the form of (xmin, ymin, xmax, ymax), optionally extended by some padding length.
    The images will be bilinearly interpolated/geo-registered to match the exact bounds
    by default, else set interpolate=False to simply use direct slicing, recommended
    only if you are confident that the bounding boxes will cut the grid directly.
    A resolution can be optionally set to e.g. 500 to resample the raster tiles to some
    desired spatial resolution/shape.
    Gaps in the main raster can be filled by passing in a number to gapfiller.

    >>> xr.DataArray(
    ...     data=np.flipud(m=np.diag(v=np.arange(8))).astype(dtype=np.float32),
    ...     coords={"y": np.linspace(7, 0, 8), "x": np.linspace(0, 7, 8)},
    ...     dims=["y", "x"],
    ... ).to_netcdf(path="/tmp/tmp_st.nc", mode="w")
    >>> selective_tile(
    ...    filepath="/tmp/tmp_st.nc",
    ...    window_bounds=[(0.5, 0.5, 2.5, 2.5), (2.5, 1.5, 4.5, 3.5)],
    ... )
    Tiling: /tmp/tmp_st.nc ... done!
    array([[[[0., 2.],
             [1., 0.]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[3., 0.],
             [0., 0.]]]], dtype=float32)
    >>> os.remove("/tmp/tmp_st.nc")
    """

    # Convert list of bounding box tuples to nice rasterio.coords.BoundingBox class
    window_bounds = [
        rasterio.coords.BoundingBox(
            left=x0 - padding, bottom=y0 - padding, right=x1 + padding, top=y1 + padding
        )
        for x0, y0, x1, y1 in window_bounds  # xmin, ymin, xmax, ymax
    ]

    # Retrieve tiles from the main raster
    with xr.open_rasterio(filepath, chunks={}) as dataset:
        print(f"Tiling: {filepath} ... ", end="")

        assert dataset.res[0] == dataset.res[1]  # ensure our pixels are square

        if resolution is None:
            resolution = dataset.res[0]
        halfpix = resolution / 2  # half pixel width used as offset during interpolation

        # Use first window_bound to determine size of output tile
        assert dataset.y[0] > dataset.y[-1]  # check that y runs from top to bottom
        y_length = int((window_bounds[0].top - window_bounds[0].bottom) / resolution)
        x_length = int((window_bounds[0].right - window_bounds[0].left) / resolution)

        # Subset dataset according to window bound (wb)
        if interpolate:
            subset_func = dask.delayed(
                lambda y, x: dataset.interp(y=y, x=x, method="linear")
            )
        else:
            subset_func = dask.delayed(
                lambda y, x: dataset.sel(y=y, x=x, method="nearest", tolerance=0)
            )

        daarray_list = []
        for wb in window_bounds:
            # Interpolation (billinear)
            new_y = np.linspace(wb.top - halfpix, wb.bottom + halfpix, num=y_length)
            new_x = np.linspace(wb.left + halfpix, wb.right - halfpix, num=x_length)
            da_subset = subset_func(y=new_y, x=new_x)

            # Mask NaN values, if the NaN value is 'nan' itself, we convert to small num
            da_masked = dask.delayed(dask.array.ma.masked_values)(
                x=da_subset,
                value=np.nan_to_num(dataset.nodatavals[0], nan=np.nan_to_num(-np.inf)),
            )
            daarray_list.append(da_masked)

        # Run actual heavy computation
        daarray_stack = dask.delayed(dask.array.stack)(daarray_list)
        # Stack interpolated grids together, and retrieve the data and mask
        daarray_stack = daarray_stack.compute()

        assert daarray_stack.ndim == 4  # check that shape is like (m, 1, height, width)
        assert daarray_stack.shape[1] == 1  # channel-first (assuming only 1 channel)
        assert not 0 in daarray_stack.shape  # ensure no empty dimensions (bad window)

        out_tiles = np.asarray(a=dask.array.ma.getdata(daarray_stack), dtype=np.float32)
        mask = np.asanyarray(a=dask.array.ma.getmaskarray(daarray_stack))

    # Gapfill main raster if there are blank spaces
    if mask.any():  # check that there are no NAN values
        nan_grid_indexes = np.argwhere(mask.any(axis=(-3, -2, -1))).ravel()

        # Replace NaN values with some number, else raise error
        if gapfiller is not None:
            print(f"gapfilling ... ", end="")

            np.copyto(
                dst=out_tiles,
                src=np.full_like(a=out_tiles, fill_value=gapfiller),
                where=mask,
            )

        else:
            for i in nan_grid_indexes:
                daarray_list[i].plot()
                plt.show()
            print(
                f"WARN: Tiles have missing data, try passing in a number to 'gapfiller'"
            )

    print("done!")
    return out_tiles


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
    selective_tile(filepath=f"highres/{f}", window_bounds=w, interpolate=False)
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
#
# - REMA (100m) is gapfilled with a 200m_filled version (bilinear interpolated to 100m)

# %%
def save_array_to_grid(
    outfilepath: str,  # without any extension! Will append .tif and .nc to this name
    window_bound: tuple,  # bounding box in format (minx, miny, maxx, maxy)
    array: np.ndarray,  # must be in CHW format (channel, height, width)
    save_netcdf: bool = False,  # whether to also save a NetCDF file
    crs: str = "EPSG:3031",  # projected coordinate system to use
    dtype: str = None,  # data type to use e.g. np.float32, default inferred from array
    nodataval: float = -2000,  # what to use as NaN, hardcoded default to -2000m
    tiled: bool = False,  # store data arranged in square tiles, default is False
    compression: rasterio.enums.Compression = rasterio.enums.Compression.none.value,
) -> xr.DataArray:
    """
    Saves a numpy array to geotiff and netcdf format according to
    some bounding box window given as (minx, miny, maxx, maxy).
    Script will append ".tif" and ".nc" file extension to the outfilepath
    for geotiff and netcdf outputs respectively. If save_netcdf=True,
    will also return an xarray.DataArray version of the resulting grid.

    Optionally set a compression algorithm to compact the geotiff
    into a smaller filesize e.g. rasterio.enums.Compression.lzw or zstd.
    """

    assert array.ndim == 3
    assert array.shape[0] == 1  # check that there is only one band/channel

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
        crs=crs,
        transform=transform,
        dtype=array.dtype if dtype is None else dtype,
        nodata=nodataval,
        compress=compression,
        tiled=tiled,
        bigtiff="YES",
    ) as new_geotiff:
        new_geotiff.write(array[0, :, :], 1)

    # Convert deepbedmap3 and cubicbedmap2 from geotiff to netcdf format
    if save_netcdf is True:
        with xr.open_rasterio(f"{outfilepath}.tif") as dataset:
            dataset.to_netcdf(f"{outfilepath}.nc")
    else:
        dataset = None

    return dataset


# %%
# Gapfill REMA_100m_dem.tif with REMA_200m_dem_filled.tif
if not os.path.exists("misc/REMA_100m_dem_filled.tif"):
    window_bound_big = rasterio.coords.BoundingBox(
        left=-2_700_000.0, bottom=-2_200_000.0, right=2_800_000.0, top=2_300_000.0
    )

    # Read in 100m spatial resolution REMA which has data gaps in some areas
    with rasterio.open("misc/REMA_100m_dem.tif") as REMA100:
        window = rasterio.windows.from_bounds(
            *window_bound_big, transform=REMA100.transform, precision=None
        ).round_offsets()
        array100 = REMA100.read(indexes=1, masked=True, window=window)

        # Read in 200m spatial resolution REMA which has complete coverage
        # Resample it using bilinear interpolation to match the REMA 100m dataset
        with rasterio.open("misc/REMA_200m_dem_filled.tif") as REMA200_filled:
            window2 = rasterio.windows.from_bounds(
                *window_bound_big, transform=REMA200_filled.transform, precision=None
            ).round_offsets()
            array200 = REMA200_filled.read(
                indexes=1,
                masked=True,
                window=window2,
                out_shape=array100.shape,
                resampling=rasterio.enums.Resampling.bilinear,
            )

    assert array100.shape == array200.shape == (45000, 55000)
    # fill in data gaps, i.e. where mask is True
    np.copyto(dst=array100, src=array200, where=array100.mask)

    save_array_to_grid(
        outfilepath="misc/REMA_100m_dem_filled",
        window_bound=window_bound_big,
        array=np.expand_dims(array100, axis=0),
        dtype=array100.dtype,
        nodataval=REMA200_filled.nodata,
        tiled=True,  # store data arranged as tiles instead of strips for easier tiling
        compression=rasterio.enums.Compression.lzw.value,  # Lempel-Ziv-Welch, lossless
    )

# %%
with dask.diagnostics.ProgressBar():
    rema = selective_tile(
        filepath="misc/REMA_100m_dem_filled.tif",
        window_bounds=window_bounds_concat,
        padding=1000,
    )
print(rema.shape, rema.dtype)

# %%
measures_velocity_x = selective_tile(
    filepath="netcdf:misc/antarctic_ice_vel_phase_map_v01.nc:VX",
    window_bounds=window_bounds_concat,
    padding=1000,
    resolution=500,
)
measures_velocity_y = selective_tile(
    filepath="netcdf:misc/antarctic_ice_vel_phase_map_v01.nc:VY",
    window_bounds=window_bounds_concat,
    padding=1000,
    resolution=500,
)
assert measures_velocity_x.shape == measures_velocity_y.shape
measuresvelocity = np.concatenate([measures_velocity_x, measures_velocity_y], axis=1)
print(measuresvelocity.shape, measuresvelocity.dtype)

# %%
accumulation = selective_tile(
    filepath="misc/Arthern_accumulation_bedmap2_grid1.tif",
    window_bounds=window_bounds_concat,
    padding=1000,
)
print(accumulation.shape, accumulation.dtype)

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
np.save(file="model/train/W2_data.npy", arr=measuresvelocity)
np.save(file="model/train/W3_data.npy", arr=accumulation)
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
quilt.build(package="weiji14/deepbedmap/model/train/W2_data", path=measuresvelocity)
quilt.build(package="weiji14/deepbedmap/model/train/W3_data", path=accumulation)
quilt.build(package="weiji14/deepbedmap/model/train/X_data", path=lores)
quilt.build(package="weiji14/deepbedmap/model/train/Y_data", path=hires)

# %%
# Original datasets for neural network predictions on bigger area
quilt.build(
    package="weiji14/deepbedmap/lowres/bedmap2_bed", path="lowres/bedmap2_bed.tif"
)
quilt.build(
    package="weiji14/deepbedmap/misc/REMA_100m_dem_filled",
    path="misc/REMA_100m_dem_filled.tif",
)
with xr.open_dataset("misc/antarctic_ice_vel_phase_map_v01.nc") as ds:
    with tempfile.NamedTemporaryFile(suffix=".nc") as tmpfile:
        ds[["VX", "VY"]].to_netcdf(path=tmpfile.name)  # save only VX, VY variables
        quilt.build(
            package="weiji14/deepbedmap/misc/antarctic_ice_vel_phase_map_v01_VX_VY",
            path=tmpfile.name,
        )
quilt.build(
    package="weiji14/deepbedmap/misc/Arthern_accumulation_bedmap2_grid1",
    path="misc/Arthern_accumulation_bedmap2_grid1.tif",
)

# %%
quilt.push(package="weiji14/deepbedmap", is_public=True)
