# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools
import os

DESCRIPTION = "Python library for machine learning on graphs"
URL = "https://github.com/stellargraph/stellargraph"

# Required packages
# full tensorflow is too big for readthedocs's builder
tensorflow = "tensorflow-cpu" if "READTHEDOCS" in os.environ else "tensorflow"
REQUIRES = [
    f"{tensorflow}>=2.1.0",
    "numpy>=1.14",
    "scipy>=1.1.0",
    "networkx>=2.2",
    "scikit_learn>=0.20",
    "matplotlib>=2.2",
    "pandas>=0.24",
]

# The demos requirements are as follows:
#
# * demos/community_detection: mplleaflet, python-igraph (separate
#   'extra', because it's only available on some platforms)
#
# * demos/ensembles/ensemble-node-classification-example.ipynb: seaborn
#
# * demos/link-prediction/hinsage/utils.py: numba
#
# Other demos do not have specific requirements
EXTRAS_REQUIRES = {
    "demos": [
        "numba",
        "jupyter",
        "seaborn",
        "rdflib",
        "mplleaflet==0.0.5",
        "gensim>=4.0.0",
    ],
    "igraph": ["python-igraph"],
    "neo4j": ["py2neo"],
    "test": [
        "pytest==5.3.1",
        "pytest-benchmark>=3.1",
        "pytest-cov>=2.6.0",
        "coverage>=4.4,<5.0",
        "black>=19.3b0",
        "nbconvert>=5.5.0",
        "treon>=0.1.2",
        "papermill>=2.0.0",
        "rdflib",
        "commonmark==0.9.1",
    ],
}

# Long description
try:
    with open("README.md", "r", encoding="utf8") as fh:
        LONG_DESCRIPTION = fh.read()
except FileNotFoundError:
    # can't find the README (e.g. building the docker image), so skip it
    LONG_DESCRIPTION = ""

# Get global version
# see: https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open("stellargraph/version.py", "r") as fh:
    exec(fh.read(), version)
VERSION = version["__version__"]

setuptools.setup(
    name="stellargraph",
    version=VERSION,
    description=DESCRIPTION,
    author="Data61, CSIRO",
    author_email="stellargraph.io@gmail.com",
    url=URL,
    license="Apache 2.0",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires=">=3.6.0, <3.9.0",
    install_requires=REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
