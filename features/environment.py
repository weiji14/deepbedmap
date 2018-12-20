import ast
import base64
import os
import sys
import types

from behave import fixture, use_fixture
import comet_ml
import nbconvert
import nbformat
import pandas as pd
import quilt
import requests


def _load_ipynb_modules(ipynb_path: str):
    """
    First converts data_prep.ipynb to a temp .py file
    Then it loads the functions from the .py file
    Returns the loaded modules
    """
    assert ipynb_path.endswith(".ipynb")
    basename, _ = os.path.splitext(ipynb_path)

    # read ipynb file and get the text
    with open(ipynb_path) as ipynb_file:
        nb = nbformat.reads(s=ipynb_file.read(), as_version=nbformat.NO_CONVERT)
    assert isinstance(nb, nbformat.notebooknode.NotebookNode)

    # convert the .ipynb text to a string .py format
    pyexporter = nbconvert.PythonExporter()
    source, meta = pyexporter.from_notebook_node(nb=nb)
    assert isinstance(source, str)

    # parse the .py string to pick out only 'import' and 'def function's
    parsed_code = ast.parse(source=source)
    for node in parsed_code.body[:]:
        if node.__class__ not in [ast.FunctionDef, ast.Import, ast.ImportFrom]:
            parsed_code.body.remove(node)
    assert len(parsed_code.body) > 0

    # import modules from the parsed .py string
    module = types.ModuleType(basename)
    code = compile(source=parsed_code, filename=f"{basename}.py", mode="exec")
    exec(code, module.__dict__)

    return module


def _quick_download_lowres_misc_datasets():
    """
    Retrieves low resolution and miscellaneous datasets quickly using Quilt
    instead of downloading from the original source.
    """
    with open(os.devnull, "w") as null:
        print("Downloading neural network model input datasets ...", end=" ")

        _stdout = sys.stdout
        _stderr = sys.stderr
        sys.stdout = sys.stderr = null

        for geotiff in [
            "lowres/bedmap2_bed",
            "misc/REMA_100m_dem",
            "misc/REMA_200m_dem_filled",
            "misc/MEaSUREs_IceFlowSpeed_450m",
        ]:

            if not os.path.exists(path=f"{geotiff}.tif"):
                # Download packages first
                quilt.install(package=f"weiji14/deepbedmap/{geotiff}", force=True)
                # Export the files to the right pathname
                quilt.export(package=f"weiji14/deepbedmap/{geotiff}", force=True)
                # Add .tif extension to filename
                os.rename(src=geotiff, dst=f"{geotiff}.tif")

        sys.stderr = _stderr
        sys.stdout = _stdout
        print("done!")


def _download_deepbedmap_model_weights_from_comet():
    """
    Download latest neural network model weights from Comet.ML
    Uses their REST API endpoint https://www.comet.ml/docs/rest-api/endpoints/
    Requires the COMET_REST_API_KEY environment variable to be set in the .env file
    """
    authHeader = {"Authorization": base64.b64decode(s=os.environ["COMET_REST_API_KEY"])}

    # Get list of DeepBedMap experiments (projectId a7e4f47215b94cd98d6db8a092d78232)
    r = requests.get(
        url="https://www.comet.ml/api/rest/v1/experiments",
        params={"projectId": "a7e4f47215b94cd98d6db8a092d78232"},
        headers=authHeader,
    )
    df = pd.io.json.json_normalize(r.json()["experiments"])

    # Get the key to the latest DeepBedMap experiment on Comet ML
    experiment_key = df.loc[df["start_server_timestamp"].idxmax()].experiment_key

    # Use key to access url to the experiment's asset which is the hdf5 weight file
    r = requests.get(
        url="https://www.comet.ml/api/rest/v1/asset/get-asset-list",
        params={"experimentKey": experiment_key},
        headers=authHeader,
    )
    asset_url = r.json()[0]["link"]

    # Download the neural network weight file (hdf5 format) to the right place!
    r = requests.get(url=asset_url, headers=authHeader)
    open(file="model/weights/srgan_generator_model_weights.hdf5", mode="wb").write(
        r.content
    )


@fixture
def fixture_data_prep(context):
    # set context.data_prep to have all the module functions
    context.data_prep = _load_ipynb_modules(ipynb_path="data_prep.ipynb")
    return context.data_prep


@fixture
def fixture_deepbedmap(context):
    # Quickly download all the neural network input datasets
    # _quick_download_lowres_misc_datasets()
    # Download trained neural network weight file
    _download_deepbedmap_model_weights_from_comet()
    # set context.deepbedmap to have all the module functions
    context.deepbedmap = _load_ipynb_modules(ipynb_path="deepbedmap.ipynb")
    return context.deepbedmap


def before_tag(context, tag):
    if tag == "fixture.data_prep":
        use_fixture(fixture_func=fixture_data_prep, context=context)
    elif tag == "fixture.deepbedmap":
        use_fixture(fixture_func=fixture_deepbedmap, context=context)
