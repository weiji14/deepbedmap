import os

from behave import given, when, then
import geopandas as gpd
import rasterio


@given("this {url} link to a file hosted on the web")
def set_url(context, url):
    context.url = url


@when("we download it to {filepath}")
def download_from_url_to_path(context, filepath):
    context.filepath = filepath
    context.data_prep.download_to_path(path=filepath, url=context.url)


@then("the local file should have this {sha256} checksum")
def check_sha256_of_file(context, sha256):
    assert context.data_prep.check_sha256(path=context.filepath) == sha256
    os.remove(path=context.filepath)  # remove downloaded file


@given("a collection of raw high resolution datasets {input_pattern}")
def collection_of_high_resolution_datasets(context, input_pattern):
    df = context.data_prep.parse_datalist()  # retrieve from data_list.yml
    subset_df = df[df.filename.str.match(input_pattern)]  # pattern match filename

    context.input_files = []  # setup empty list to store path to downloaded input files
    for file in subset_df.itertuples():
        filepath = os.path.join(file.folder, file.filename)  # join folder and filename
        context.data_prep.download_to_path(path=filepath, url=file.url)  # download
        assert context.data_prep.check_sha256(path=filepath) == file.sha256
        context.input_files.append(filepath)  # append filepath to the input list


@when("we process the data through {pipeline_file}")
def process_data_through_pipeline_and_get_output(context, pipeline_file):
    pf = os.path.join("highres", pipeline_file)  # join folder and filename
    context.xyz_data = context.data_prep.ascii_to_xyz(pipeline_file=pf)
    assert list(context.xyz_data.columns) == ["x", "y", "z"]


@when("interpolate the xyz data table to {output_file}")
def interpolate_xyz_data_to_grid(context, output_file):
    region = context.data_prep.get_region(context.xyz_data)
    context.outfile = os.path.join("/tmp/highres", output_file)
    os.makedirs(os.path.dirname(context.outfile), exist_ok=True)
    context.data_prep.xyz_to_grid(
        xyz_data=context.xyz_data, region=region, outfile=context.outfile
    )


@then("a high resolution raster grid is returned")
def open_raster_grid_to_check(context):
    with rasterio.open(context.outfile) as raster_source:
        assert raster_source.closed == False  # check that it can be opened


@given("a big {dataset_type} raster grid {raster_grid}")
def get_a_raster_grid(context, dataset_type, raster_grid):
    context.raster_grid = raster_grid
    context.filepath = os.path.join("/tmp", dataset_type, raster_grid)
    os.makedirs(os.path.dirname(context.filepath), exist_ok=True)
    url = (
        f"https://github.com/weiji14/deepbedmap/releases/download/v0.7.0/{raster_grid}"
    )
    context.data_prep.download_to_path(path=context.filepath, url=url)


@given('a collection of square bounding boxes "{geojson_file}"')
def get_window_bounds(context, geojson_file):
    geodataframe = gpd.read_file(geojson_file)
    context.window_bounds = [
        geom.bounds
        for geom in geodataframe.query("grid_name == @context.raster_grid").geometry
    ]
    context.number_of_square_bounds = len(context.window_bounds)


@when("we crop the big raster grid using those bounding boxes")
def big_raster_grid_to_small_square_tiles(context):
    context.tiles = context.data_prep.selective_tile(
        filepath=context.filepath, window_bounds=context.window_bounds
    )


@then("a stack of small raster tiles is returned")
def step_impl(context):
    assert context.tiles.ndim == 4  # check that shape is like (m,height,width,channels)
    assert len(context.tiles) == context.number_of_square_bounds  # check array length
