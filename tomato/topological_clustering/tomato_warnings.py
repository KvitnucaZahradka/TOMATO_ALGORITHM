#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019 October 27 21:56:29 (EST) 

@author: KanExtension
"""

import warnings


def user_is_using_default_distance_function():
    warnings.warn('User is using the default distance function: `scipy.spatial.distance.euclidean`!')

def operation_might_take_long_to_finish(operation: str):
    warnings.warn(f'This operation `{operation}` might take significant time and resources to finish!')

def warn_user_about_empty_densities_store():
    warnings.warn('The user\'s instance of `DensityEstimators` has an empty `density_store`.'
                  'Either use a `generate_full_density_store` function or generate store by hand.')