#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019 November 05 20:13:17 (EST) 

@author: KanExtension
"""
from setuptools import setup, find_packages

setup(
    name="topological_clustering",
    version="0.1.0",
    description="Implementation of the ToMATo clustering algorithm, "
                "with clique complex and KNN nearest neighbors graph.",
    long_description_content_type="markdown",
    author="KanExtension",
    author_email="infinite.omega.category@gmail.com",
    url="https://github.com/KvitnucaZahradka/TOMATO_ALGORITHM",
    packages=find_packages(),
    install_requires=['scikit-learn>=0.20.0',
                      'numpy>=1.17.0',
                      'matplotlib>=2.1.0',
                      'networkx>=2.0',
                      'scipy>=1.3.0'],
    include_package_data=True,
    zip_safe=False,
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ),
)
