import ast
from behave import fixture, use_fixture
import nbconvert
import nbformat
import os
import types


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


@fixture
def fixture_data_prep(context):
    # set context.data_prep to have all the module functions
    context.data_prep = _load_ipynb_modules(ipynb_path="data_prep.ipynb")
    return context.data_prep


def before_tag(context, tag):
    if tag == "fixture.data_prep":
        use_fixture(fixture_func=fixture_data_prep, context=context)
