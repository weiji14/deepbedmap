from behave import given, when, then
import os
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


@given("a high resolution dataset {input_file}")
def collection_of_high_resolution_datasets(context, input_file):
    #!cd highres && ls *.json
    dataframe = context.data_prep.parse_datalist()  # retrieve from data_list.yml
    subset_df = dataframe[dataframe.filename == input_file]  # select based on filename

    context.input_files = []  # setup empty list to store input files
    for file in subset_df.itertuples():
        context.folder = file.folder
        filepath = os.path.join(file.folder, file.filename)  # join folder and filename
        context.data_prep.download_to_path(path=filepath, url=file.url)  # download
        context.input_files.append(filepath)  # append filepath to the input list


@when("we process it through {pipeline_file}")
def process_data_through_pipeline_and_get_output(context, pipeline_file):
    context.output_files = []  # setup empty list to store output files
    for filepath in context.input_files:
        pipeline_file = os.path.join(context.folder, pipeline_file)  # prepend dirname
        outfile = context.data_prep.processing_pipeline(pipeline_file=pipeline_file)
        context.output_files.append(outfile)  # append filepath to the output list


@then("a raster grid is returned {output_file}")
def open_raster_grid_to_check(context, output_file):
    assert len(context.output_files) == 1  # check that only one raster is produced
    for outfile in context.output_files:
        assert outfile == os.path.join(context.folder, output_file)
        with rasterio.open(outfile) as raster_source:
            assert raster_source.closed == False  # check that it can be opened
