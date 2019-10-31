#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019 October 26 20:37:09 (EST) 

@author: KanExtension
"""

import os
import numpy as np
from sklearn.neighbors import BallTree, NearestNeighbors

import tomato_utils as tu
import tomato_warnings as tw

from itertools import repeat, tee
from typing import Iterator, Union, Dict, Callable, Tuple

from functools import reduce


'''

C L A S S E S

'''


class DensityEstimators:
    """This class implements one particular density estimator

    Attributes
    ----------
    density_estimators: Union[Dict[str, dict], None]
        if None, then we do not want to prepare any `bulk store` (expalined later),
        then the aim of this class is to use particular density estimators on will

        if it is dictionary of restricted strings (keys) with values being the dictionaries of fitting hyperparameters.
        The restricted string might have the following values:
        - `kde_gauss_grid_search`
        - `kde_gauss_logarithmic_gauss_grid_search`
        - `kde_tophat`
        - `kde_logarithmic_tophat`
        - `kde_logarithmic_knn`
        For further explanations and descriptions of above estimator, see `Notes` in this docstring.

    Methods
    -------

    Notes
    -----
    - `kde_gauss_grid_search`:
    - `kde_gauss_logarithmic_gauss_grid_search`:
    - `kde_tophat`:
    - `kde_logarithmic_tophat`:
    - `kde_logarithmic_knn`:


    """

    # -- defaults --
    _allowed_density_estimators = {'kde_gauss_grid_search', 'kde_logarithmic_gauss_grid_search',
                                   'kde_tophat', 'kde_logarithmic_tophat', 'kde_logarithmic_knn'}

    # -- constructor --
    def __init__(self, density_estimators: Union[Dict[str, dict], None] = None):

        self._density_estimators = None
        self._densities_store = None

        # @property handling
        self.__density_estimators = density_estimators


    '''
    HELPER FUNCTIONS
    '''
    @property
    def densities_store(self) -> Union[Dict[str, Iterator[np.ndarray]], None]:
        """

        Returns
        -------

        """
        if self._densities_store is None:
            tw.warn_user_about_empty_densities_store()

        return self._densities_store

    @densities_store.setter
    def densities_store(self, densities_store: Union[Dict[str, Iterator[np.ndarray]], None]):
        """

        Parameters
        ----------
        densities_store

        Returns
        -------

        """
        assert isinstance(densities_store, (dict, None)), f'The value of `densities_store` is neither `dict` nor `None`!'

        if densities_store is None:
            tw.warn_user_about_empty_densities_store()

        self._densities_store = densities_store

    @property
    def __density_estimator(self) -> Union[Dict[str, dict], None]:
        """

        Returns
        -------

        """
        return self._density_estimators

    @__density_estimator.setter
    def __density_estimator(self, density_estimators: Union[Dict[str, dict], None]):
        """

        Parameters
        ----------
        density_estimators

        Returns
        -------

        """
        assert isinstance(density_estimators, (dict, None))

        if density_estimators is None:
            self._density_estimators = None

        else:
            _check_set = set(density_estimators).difference(DensityEstimators._allowed_density_estimators)
            assert not _check_set, f'The following names in `density_estimators` ' \
                f'are not allowed: {_check_set}!'

            self._density_estimators = density_estimators

    '''
    M A I N  F U N C T I O N S
    '''
    def kde_gauss_grid_search(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """This defines one particular Morse function.

        Parameters
        ----------
        X: np.ndarray

        **cross_validation: int
            -- default -- is 2

        **bandwidth: int
            -- default -- np.linspace(0.1, 1.0, 30)

        Returns
        -------
        np.ndarray

        """
        # -- defaults --
        _cross_validation = kwargs.get('cross_validation', 2)
        _bandwidth = kwargs.get('bandwidth', np.linspace(0.1, 1.0, 30))

        return tu.evaluate_kde_estimator(tu.fit_gaussian_kde_estimator(X, cross_validation=_cross_validation,
                                                                       bandwidth=_bandwidth), X)

    def kde_logarithmic_gauss_grid_search(self,  X: np.ndarray, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        X: np.ndarray

        **cross_validation: int
            -- default -- is 2

        **bandwidth: int
            -- default -- np.linspace(0.1, 1.0, 30)

        Returns
        -------
        np.ndarray


        """
        # -- defaults --
        _cross_validation = kwargs.get('cross_validation', 2)
        _bandwidth = kwargs.get('bandwidth', np.linspace(0.1, 1.0, 30))

        _densities = tu.evaluate_kde_estimator(tu.fit_gaussian_kde_estimator(X, cross_validation=_cross_validation,
                                                                             bandwidth=_bandwidth), X, False)
        # you must shift quasi - densities to be nonzero (for ToMATo to work correctly.)
        return _densities + np.abs(np.min((0.0, np.min(_densities))))

    def kde_tophat(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        X
        kwargs

        Returns
        -------

        """
        return tu.evaluate_kde_estimator(tu.fit_single_kde_estimator(X, kernel='tophat', **kwargs), X)

    def kde_logarithmic_tophat(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        X
        kwargs

        Returns
        -------

        """

        _densities = tu.evaluate_kde_estimator(tu.fit_single_kde_estimator(X, kernel='tophat', **kwargs), X, False)

        # quasi - densities shift
        return _densities + np.abs(np.min((0.0, np.min(_densities))))

    def kde_logarithmic_knn(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """

        Parameters
        ----------
        X
        kwargs

        Returns
        -------

        """
        # -- constants --
        _allowed_algorithms = {'BallTree', 'NearestNeighbors'}

        # -- defaults --
        _algorithm = kwargs.get('knn_algorithm', 'BallTree')

        assert _algorithm in _allowed_algorithms, f'The algorithm name you provided: {_algorithm} is not among ' \
            f'allowed knn algorithms: {_allowed_algorithms}'

        _log_regularizer = kwargs.get('log_regularizer', 0.0)

        _leaf_size = kwargs.get('leaf_size', 6)

        _n_neighbors = kwargs.get('knn_n_neighbors', 6)

        # -- code --
        if _algorithm == 'BallTree':
            # create the local graph
            _graph = BallTree(X, leaf_size=_leaf_size)

        elif _algorithm == 'NearestNeighbors':
            # create the local graph
            _graph = NearestNeighbors(n_neighbors=_n_neighbors, leaf_size=_leaf_size,
                                      algorithm='ball_tree', p=2, n_jobs=os.cpu_count() - 1)
            # fit graph
            _graph.fit(X)

            # define the query function, so to be consistent with other functions
            _graph.query = lambda X, k: _graph.kneighbors(X=X, n_neighbors=k)

        else:
            raise ValueError('Something horrible has happened, contact the project owner!')

        # -- functions --
        def _log_density_estimator(in_vector: np.ndarray, k: int = _n_neighbors) -> float:
            return np.log(k) - np.log(X.shape[0]) - \
                      X.shape[1] * np.log(_log_regularizer + _graph.query(in_vector, k=k)[0][0][-1])

        # -- code --

        # it might happen that the `_log_density_estimator` is negative, so you must shift it
        _densities = np.array(list(map(lambda ind: _log_density_estimator(X[ind:ind+1]), range(X.shape[0]))))

        return _densities + np.abs(np.min((0.0, np.min(_densities))))

    def generate_full_density_store(self, X: np.ndarray):
        """This creates an infinite `bulk store` of densities. Note, all the densities are
        at already evaluated once.

        Parameters
        ----------
        X: np.ndarray

        Returns
        -------
        dictionary
            it has structure: {'density_alg_name': repeat(density_algorithm(**kwargs)), ...}

        Notes
        -----
        - every Iterator under key, is part of itertools.repeat, so its infinite iterator over
        the np.ndarrays

        """
        self.densities_store = reduce(lambda d, key_val:\
                          {**d, **{key_val[0]: repeat(DensityEstimators.__dict__[key_val[0]](self, X, **key_val[1]))}},
                      self._density_estimators.items(), {})

    def get_single_density(self, density_name: str) -> np.ndarray:
        """

        Parameters
        ----------
        density_name

        Returns
        -------
        np.ndarray

        """
        assert density_name in self.densities_store, f'The density name you provided: {density_name} is not in the ' \
            f'`densities_store` you created!'

        return next(self.densities_store[density_name])


class Tomato:
    """
    <ADD DOCSTRING>
    """
    def __init__(self, X: np.ndarray, density_estimator: np.ndarray, X_metadata: Union[np.ndarray, None] = None,
                 **kwargs):

        _bool_filter = kwargs.get('bool_density_filter', lambda x: True)

        self._ordered_data = None
        self._density_estimator = None
        self._X_metadata = None
        self._tilde_f = None

        # this attribute will be eventually used during the `fit` procedure
        self._union_find: Union[tu.UnionFind, None] = None

        # here we keep the persistence homology data, the dictionary keeping the `deaths` and persistence clusters
        self._persistence_data: Union[dict, None] = None

        # here we will keep the attribute for the graph type; also eventually filled
        self._graph_type = None

        # this empty attribute will hold a graph that will later present the neighborhood structure
        self._graph = None

        # @property handling of the correctness of `density_estimator`
        self.__X_metadata = X_metadata
        self.__density_estimator = density_estimator

        # this sets up the correct `_ordered_data`, `_density_estimator` and `_X_metadata`
        self.__ordered_data = X, _bool_filter

    '''
    HELPER FUNCTIONS
    '''
    @property
    def __ordered_data(self) -> np.ndarray:
        """

        Returns
        -------

        """
        return self._ordered_data

    @__ordered_data.setter
    def __ordered_data(self, data_flt: Tuple[np.ndarray, Callable]):
        assert isinstance(data_flt[0], np.ndarray), f'The data held in `X` variable must have type: `np.ndarray`.'
        assert isinstance(data_flt[1], Callable), f'The **kwargs argument `bool_density_filter` must be of type' \
            f' `Callable`!'
        assert self._density_estimator.shape[0] == data_flt[0].shape[0], f'The length of `density_estimator` array ' \
            f'and the data array held in `X` must be the same!'

        _X, _bool_filter = data_flt

        # -- create the data store --
        if self._X_metadata is None:

            _data_store = zip(_X, self._density_estimator)

            # apply filter with respect to density
            _data_store = filter(lambda x: _bool_filter(x[-1]), _data_store)

            _data_store = sorted(_data_store, key=lambda x: -x[-1])

        else:
            _data_store = zip(_X, self._X_metadata, self._density_estimator)

            # apply filter with respect to density
            _data_store = filter(lambda x: _bool_filter(x[-1]), _data_store)

            _data_store = sorted(_data_store, key=lambda x: -x[-1])

            # set up the _X_metadata
            self._X_metadata = np.array(list(map(lambda x: x[1], _data_store)))

            _data_store = tee(map(lambda x: (x[0], x[-1]), _data_store))

        # set the `self._tile_f`
        self._tilde_f = np.array(list(map(lambda x: x[-1], _data_store[0])))

        # set the ordered data
        self._ordered_data = np.array(list(map(lambda x: x[0], _data_store[1])))

    @property
    def __X_metadata(self) -> Union[np.ndarray, None]:
        """

        Returns
        -------

        """
        return self._X_metadata

    @__X_metadata.setter
    def __X_metadata(self, X_metadata: Union[np.ndarray, None]):
        """

        Parameters
        ----------
        X_metadata

        Returns
        -------

        """
        assert isinstance(X_metadata, (np.ndarray, None)), f'The value of `X_metadata` must be either `np.ndarray`' \
            f' or `None`'

        self._X_metadata = X_metadata

    @property
    def __density_estimator(self) -> Union[DensityEstimators, None]:
        """

        Returns
        -------

        """
        return self._density_estimator

    @__density_estimator.setter
    def __density_estimator(self, density_estimator: np.ndarray):
        """

        Parameters
        ----------
        density_estimator: np.ndarray

        """
        assert isinstance(density_estimator, np.ndarray), 'The value in `density_estimator` must be of the ' \
                                                          'type `np.ndarray`!'

        self._density_estimator = density_estimator

    def _cluster_like_tomato(self, cluster_index: int) -> tuple:
        """

        Parameters
        ----------
        cluster_index

        Returns
        -------

        """
        # calculate the cluster indices
        _cluster_indices = filter(lambda x: x[1] == cluster_index, self._union_find.object_id_to_parent_id.items())
        _cluster_indices = tee(map(lambda x: x[0], _cluster_indices))

        # pick those data points that have the correct indices
        _ord_cluster_data = self._ordered_data[list(_cluster_indices[0])]

        _ord_cluster_metadata = None if self._X_metadata is None else self._X_metadata[list(_cluster_indices[1])]

        return _ord_cluster_data, _ord_cluster_metadata




















