#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019 October 27 21:56:29 (EST) 

@author: KanExtension
"""

import warnings


def user_is_using_default_distance_function():
    warnings.warn('User is using the default distance function: `scipy.spatial.distance.euclidean`!')
