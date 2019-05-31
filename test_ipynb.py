# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.4-rc1
#   kernelspec:
#     display_name: deepbedmap
#     language: python
#     name: deepbedmap
# ---

# %% [markdown]
# # Behavioural Driven Development Testing for Jupyter Notebooks
#
# Handy way to process the run unit tests (via doctest) and integration tests (via behave) in jupyter notebooks (.ipynb) containing Python functions.
# The script will convert an .ipynb to a string format (basically a .py file), loads them as modules, and runs the tests on them.
# To run it in the console, do:
#
#     python -m pytest --verbose --disable-warnings --nbval test_ipynb.ipynb
#
# The script should tell you which ipynb file's doctests has failed (e.g. srgan_train.ipynb).
# You can then open up this very jupyter notebook to debug and inspect the situation further.

# %%
from features.environment import _load_ipynb_modules
import behave.__main__

import doctest
import os
import sys


def _unit_test_ipynb(path: str):
    """
    Unit tests on loaded modules from a .ipynb file.
    Uses doctest.
    """
    assert path.endswith(".ipynb")

    module = _load_ipynb_modules(ipynb_path=path)
    num_failures, num_attempted = doctest.testmod(m=module, verbose=True)
    if num_failures > 0:
        sys.exit(num_failures)


def _integration_test_ipynb(path: str, summary: bool = False):
    """
    Integration tests on various feature behaviours inside a .feature file.
    Uses behave.
    """
    assert os.path.exists(path=path)
    assert path.endswith(".feature")

    if summary == False:
        args = f"--tags ~@skip --no-summary {path}"
    elif summary == True:
        args = f"--tags ~@skip {path}"

    num_failures = behave.__main__.main(args=args)
    if num_failures > 0:
        sys.exit(num_failures)


# %% [markdown]
# ## Unit tests
# Uses [doctest](https://en.wikipedia.org/wiki/Doctest).
# Small tests for each individual function.

# %%
_unit_test_ipynb(path="data_prep.ipynb")

# %%
_unit_test_ipynb(path="srgan_train.ipynb")

# %% [markdown]
# ## Integration tests
#
# Uses [behave](https://github.com/behave/behave).
# Medium sized tests which checks that components work together properly.
# Ensures that the behaviour of features (made up of units) is sound.

# %%
_integration_test_ipynb(path="features/data_prep.feature")

# %%
_integration_test_ipynb(path="features/srgan_train.feature")

# %%
_integration_test_ipynb(path="features/deepbedmap.feature")
