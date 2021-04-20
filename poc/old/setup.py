from setuptools import find_packages, setup
from pathlib import Path


# Package meta-data.
NAME = 'auto-machine-learning'
DESCRIPTION = "Package for automatic transformers and models selection"
URL = "https://github.com/jslomkowski/aml"
EMAIL = "jerzy.slomkowski@gmail.com"
AUTHOR = "jerzy slomkowski"
REQUIRES_PYTHON = ">=3.7.9"


# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()


# Load the package's VERSION file as a dictionary.
about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'aml'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"aml": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
)
