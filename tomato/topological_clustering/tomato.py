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

from itertools import tee
from typing import Iterator, Tuple, Set, Union, Callable, List

'''

C L A S S E S

'''


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





















