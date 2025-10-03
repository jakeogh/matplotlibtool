# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

import fastentrypoints

dependencies = ["click"]

config = {
    "version": "0.1",
    "name": "matplotlibtool",
    "url": "https://github.com/jakeogh/matplotlibtool",
    "license": "ISC",
    "author": "Justin Keogh",
    "author_email": "github.com@v6y.net",
    "description": "2d and 3d matplotlib plotting lib for structured arrays",
    "long_description": __doc__,
    "packages": find_packages(exclude=["tests"]),
    "package_data": {"matplotlibtool": ["py.typed"]},
    "include_package_data": True,
    "zip_safe": False,
    "platforms": "any",
    "install_requires": dependencies,
    "entry_points": {
        "console_scripts": [
            "matplotlibtool=matplotlibtool.cli:cli",
        ],
    },
}

setup(**config)
