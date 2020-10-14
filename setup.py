#!/usr/bin/env python
import setuptools
import os
from setuptools import setup


def get_version(path):
    with open(path, "r") as f:
        _, version = f.read().strip().split("=")
        version = version.strip().strip('"')
    return version


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mgt2001",
    version=get_version(os.path.join(
        ".",  # os.path.dirname(os.path.realpath(__file__)),
        "mgt2001",
        "_version.py",
    )),
    author="Brian L. Chen",
    author_email="brian.lxchen@gmail.com",
    description="A small package for MGT 2001 use only",
    long_description_content_type="text/markdown",
    url="https://github.com/icheft/MGT2001_package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'pandas', 'matplotlib', 'adjustText', 'scipy'],
    include_package_data=True,
)
