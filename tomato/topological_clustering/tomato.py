#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019 October 26 20:37:09 (EST) 

@author: KanExtension
"""

import os
import numpy as np
import networkx as nx

from itertools import product, combinations
from collections import Counter

from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, NearestNeighbors

import tomato_utils as tu
import tomato_warnings as tw

from itertools import tee, repeat
from typing import Iterator, Tuple, Set, Union, Callable, List, Dict

from functools import reduce

'''

C L A S S E S

'''


class DensityEstimators:
    """This class implements one particular density estimator

    Attributes
    ----------
    density_estimators: Tuple[str]
        is a tuple of restricted strings. The restricted string might have the following values:
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
    def __init__(self, density_estimators: Dict[str, dict]):

        self._density_estimators = None

        # @property handling
        self.__density_estimators = density_estimators


    '''
    HELPER FUNCTIONS
    '''
    @property
    def __density_estimator(self) -> Union[Dict[str, dict], None]:
        """

        Returns
        -------

        """
        return self._density_estimators

    @__density_estimator.setter
    def __density_estimator(self, density_estimators: Dict[str, dict]):
        """

        Parameters
        ----------
        density_estimators

        Returns
        -------

        """
        assert isinstance(density_estimators, dict)

        _check_set = set(density_estimators).difference(DensityEstimators._allowed_density_estimators)
        assert not _check_set, f'The following names in `density_estimators` ' \
            f'are not allowed: {_check_set}!'

        self._density_estimators = density_estimators

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



    '''
    M A I N  F U N C T I O N S
    '''
    def densities(self, X: np.ndarray) -> Dict[str, Iterator[np.ndarray]]:
        """
        Parameters
        ----------
        X: np.ndarray

        Returns
        -------

        Notes
        -----
        - every Iterator under key, is part of itertools.repeat, so its infinite iterator over
        the np.ndarrays

        """

        return reduce(lambda d, key_val: {**d, **{key_val[0]: repeat(self.__dict__[key_val[0]](X, **key_val[1]))}},
                      self._density_estimators.items(), {})


class SimplicalComplex:
    """
    ADD DOCSTRING

    """

    def __init__(self, simplices: tuple = ()):

        self._simplices = None
        self._face_set = None

        # do not be fooled, this is handled via @property setter
        self.simplices = simplices

    '''
    H E L P E R   M E T H O D S
    '''
    @property
    def simplices(self) -> np.ndarray:
        """
        Returns
        -------

        """
        return self._simplices

    @simplices.setter
    def simplices(self, simplices: Union[Tuple[int], Iterator[int]]):
        """Is a setter for the simplices

        Parameters
        ----------
        simplices

        """
        self._simplices = np.array(list(map(lambda simplex: tuple(sorted(simplex)), simplices)))

    '''
    M A I N   F U N C T I O N S
    '''
    def faces(self) -> set:
        """This function returns the set of faces of the given Simplical complex

        Returns
        -------

        Notes
        -----
        Note, the notion of face is different from the notion of `boundary`.

        Examples
        --------

        """
        self._face_set = set()

        for simplex in self._simplices:
            _num_nodes = len(simplex)

            for r in range(_num_nodes, 0, -1):
                for face in combinations(simplex, r):
                    self._face_set.add(face)

        return self._face_set

    def n_faces(self, dim: int) -> Iterator[Tuple[int]]:
        """This function returns generator over faces of simplical complex.

        Parameters
        ----------
        dim:int
            is positive integer determining dimension of returned faces of the graph.

        Returns
        -------
            iterator over faces of dimension `dim`, each face is represented by tuple.

        """
        yield from filter(lambda face: len(face) == dim + 1, self._face_set)


class VietorisRipsComplex(SimplicalComplex):
    """
    ADD DOCSTRING
    """

    def __init__(self, points: np.ndarray, epsilon: float, labels: Union[np.ndarray, None],
                 distance_function: Union[Callable[[List[float], List[float]], float], None]):

        # initialize the superclass
        super().__init__()

        # attributes definitions
        # do not worry about nones, we will fix it through the @property
        self._pts = None
        self._epsilon = None
        self._labels = None

        self._distfcn = None

        # note, the cliques will be dealt later.
        self._cliques = None

        # @property setters
        self.__points = points
        self.__labels = labels
        self.__epsilon = epsilon
        self.__distance_function = distance_function

        # different derived attributes
        self.network = self._construct_network(self._pts, self._labels, self._epsilon, self._distfcn)

        # do not be fooled, this is handled via @property setter inherited from superclass
        self.simplices = map(tuple, nx.find_cliques(self.network))

    '''
    H E L P E R   M E T H O D S
    '''
    @property
    def __points(self) -> Union[np.ndarray, None]:
        """

        Returns
        -------

        """
        return self._pts

    @__points.setter
    def __points(self, points: np.ndarray):
        """

        Parameters
        ----------
        points: np.ndarray

        Returns
        -------

        """
        self._pts = points

    @property
    def __labels(self) -> Union[Iterator[int], np.ndarray, None]:
        """

        Returns
        -------

        """
        return self._labels

    @__labels.setter
    def __labels(self, labels: Union[np.ndarray, None]):
        """

        Parameters
        ----------
        labels

        Returns
        -------

        """
        self._labels = range(len(self._pts)) if labels is None or len(labels) != len(self._pts)\
            else labels

    @property
    def __epsilon(self) -> Union[float, None]:
        """

        Returns
        -------

        """
        return self._epsilon

    @__epsilon.setter
    def __epsilon(self, epsilon: float):
        """

        Parameters
        ----------
        epsilon

        Returns
        -------
        None

        """
        self._epsilon = epsilon

    @property
    def __distance_function(self) -> Union[Callable, None]:
        """

        Returns
        -------

        """
        return self._distfcn

    @__distance_function.setter
    def __distance_function(self, distance_function: Union[Callable, None]):
        """

        Parameters
        ----------
        distance_function

        Returns
        -------

        """
        assert isinstance(distance_function, (None, Callable)),\
            f'The `distance_function` parameter: {distance_function} must be either None or Callable.'

        if distance_function is None:
            tw.user_is_using_default_distance_function()
            self._distfcn = distance.euclidean

        else:
            self._distfcn = distance_function

    '''
    M A I N   F U N C T I O N S
    '''

    def generate_vietoris_rips_complex(self):
        """This function generates full `Vietoris_Rips_complex`.
        Note, this function might take nontrivial time and resources.

        Returns
        -------
        None

        Examples
        --------

        """
        if self._cliques is None:
            tw.operation_might_take_long_to_finish('generate_vietoris_rips_complex')

            self._cliques = np.ndarray(list(nx.find_cliques(self.network)))

    def _construct_network(self, points: np.ndarray, labels: np.ndarray, epsilon: float,
                           distfcn: Callable) -> nx.Graph:
        """This function constructs a `networkx graph` object that is a representation
        of the `Vietoris_Rips_complex`.

        Parameters
        ----------
        points: np.ndarray
            is an ndarray of the data points (of shape: n_examples, dimensions),
            distance of any two members must be computable via `distfcn` function passed as argument

        labels: np.ndarray
            is the array of `lables` associated with each data point in `points`

        epsilon: float
            is a positive `float`, i.e. the distance, if two data points are under that distance, they are connected

        distfcn: 'scipy distance function'
            is a scipy distance function, that measures distances of two data points

        Returns
        -------
        'networkx graph'
            is the instance of the networkx graph

        """
        # create an instance of `networkx` (simple:: no self loops and multiple edges) Graph
        g = nx.Graph()
        g.add_nodes_from(labels)

        # creating 2 copies of iterators
        zips, spiz = tee(zip(points, labels))

        # this is a bottleneck, you must iterate over each possible product
        # !! improve in the future (should be able to parallelize this)
        for pair in product(zips, spiz):

            # this means that pair must be distinct to connect it
            if pair[0][1] != pair[1][1]:
                dist = distfcn(pair[0][0], pair[1][0])

                # Vietoris_Rips_complex condition: consider pair to be an edge
                # only if a distance is smaller than `epsilon`
                if dist and dist < epsilon:
                    g.add_edge(pair[0][1], pair[1][1])
        return g

    def print_complex(self):
        """

        Returns
        -------

        """
        self.generate_vietoris_rips_complex()

        print(f'The Vietoris-Rips complex is: {self._cliques}')



class Tomato:
    """
    <ADD DOCSTRING>
    """
    def __init__(self):
        pass





















