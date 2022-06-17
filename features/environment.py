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

    # parse the .py string to pick out only 'class', 'import' and 'def function's
    parsed_code = ast.parse(source=source)
    for node in parsed_code.body[:]:
        if node.__class__ not in [
            ast.ClassDef,
            ast.FunctionDef,
            ast.Import,
            ast.ImportFrom,
        ]:
            parsed_code.body.remove(node)
    assert parsed_code.body  # make sure there is a non-empty list

    # import modules from the parsed .py string
    module = types.ModuleType(basename)
    code = compile(source=parsed_code, filename=f"{basename}_ipynb.py", mode="exec")
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
            "misc/Arthern_accumulation_bedmap2_grid1",
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


def _download_model_weights_from_comet(
    experiment_key: str = "latest",
    download_path: str = "model/weights/srgan_generator_model_weights.npz",
) -> (int, float):
    """
    Download DeepBedMap's Generator neural network model weights from Comet.ML
    By default, the model weights from the latest experimental run are downloaded
    Passing in an alternative experiment_key hash will download that one instead.
    Also returns the model's num_residual_blocks and residual_scaling hyperparameters.

    Uses Comet.ML's Python REST API class at https://www.comet.ml/docs/python-sdk/API/
    Requires the COMET_REST_API_KEY environment variable to be set in the .env file
    """
    comet_api = comet_ml.API(
        # rest_api_key=base64.b64decode(s=os.environ["COMET_REST_API_KEY"])
    )

    # Get pointer to a DeepBedMap experiment on Comet ML
    if experiment_key == "latest":
        # Get list of DeepBedMap experiments
        project = comet_api.get(workspace="weiji14", project="deepbedmap")
        df = pd.io.json.json_normalize(data=project.data["experiments"].values())
        # Get the key to the latest DeepBedMap experiment on Comet ML
        experiment_key = df.loc[df["start_server_timestamp"].idxmax()].experiment_key

    experiment = comet_api.get(
        workspace="weiji14", project="deepbedmap", experiment=experiment_key
    )

    # Use key to access url to the experiment's asset which is the npz weight file
    assets = experiment.asset_list
    for asset in experiment.asset_list:
        # make sure we pick the correct .npz file
        if asset["fileName"] == os.path.basename(download_path):
            asset_id = asset["assetId"]
            break

    # Download the neural network weight file (npz format) to the right place!
    os.makedirs(name=os.path.dirname(download_path), exist_ok=True)
    with open(download_path, mode="wb") as model_weight_file:
        model_weight_file.write(experiment.get_asset(asset_id=asset_id))

    # Get hyperparameters needed to recreate DeepBedMap model architecture properly
    hyperparameters: dict = (
        pd.io.json.json_normalize(data=experiment.parameters)
        .set_index(keys="name")
        .valueCurrent.to_dict()
    )
    return hyperparameters


@fixture
def fixture_data_prep(context):
    # set context.data_prep to have all the module functions
    context.data_prep = _load_ipynb_modules(ipynb_path="data_prep.ipynb")
    return context.data_prep


@fixture
def fixture_srgan_train(context):
    # set context.srgan_train to have all the module functions
    context.srgan_train = _load_ipynb_modules(ipynb_path="srgan_train.ipynb")
    return context.srgan_train


@fixture
def fixture_deepbedmap(context):
    # Quickly download all the neural network input datasets
    # _quick_download_lowres_misc_datasets()
    # set context.deepbedmap to have all the module functions
    context.deepbedmap = _load_ipynb_modules(ipynb_path="deepbedmap.ipynb")
    return context.deepbedmap


def before_tag(context, tag):
    if tag == "fixture.data_prep":
        use_fixture(fixture_func=fixture_data_prep, context=context)
    elif tag == "fixture.srgan_train":
        use_fixture(fixture_func=fixture_srgan_train, context=context)
    elif tag == "fixture.deepbedmap":
        use_fixture(fixture_func=fixture_deepbedmap, context=context)
