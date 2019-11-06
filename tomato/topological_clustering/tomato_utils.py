#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2019 October 26 20:37:25 (EST) 

@author: KanExtension
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.datasets import make_blobs, make_moons, make_circles
from typing import Iterator, Tuple, Union, Callable, List
from itertools import tee
import networkx as nx
import topological_clustering.tomato_warnings as tw
from itertools import product, combinations
from scipy.spatial import distance

__all__ = ["ClusterGenerator"]


'''

F U N C T I O N S

'''


def fit_gaussian_kde_estimator(input_data: np.ndarray, bandwidth: np.ndarray = None, cross_validation: int = 2,
                               verbose: bool = False, **kwargs) -> 'sklearn estimator':
    """
    
    
    Parameters
    ----------
    input_data: np.ndarray
    
    bandwidth: np.ndarray
    
    cross_validation: int
        -- default -- is 2
        
    verbose: bool
    
    kwargs
        are optional argumenst solely used in sklearn function `KernelDensity`
        after specifying `kernel='gaussian', so do not specify kernel.
        For definition of `KernelDensity` argument see:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
        
    Returns
    -------
    sklearn estimator
    
    Raises
    ------
    ValueError
    
    Examples
    --------

    """
    if not (bandwidth is None or isinstance(bandwidth, np.ndarray)):
        raise ValueError('The `bandwidth` parameter can either be None or np.ndarray.')

    _default_bandwidth_searchspace = np.linspace(0.1, 1.0, 30)

    _bandwidth = _default_bandwidth_searchspace if bandwidth is None else bandwidth

    _grid = GridSearchCV(KernelDensity(kernel='gaussian', **kwargs),{'bandwidth':_bandwidth},
                         cv=cross_validation)

    # fit the `input data` to produce the best kde.
    _grid.fit(input_data)

    if verbose:
        print(f'Best bandwidth: {_grid.best_params_}')

    return _grid.best_estimator_


def fit_single_kde_estimator(input_data: np.ndarray, **kwargs) -> 'sklearn `kde` estimator':
    """

    Parameters
    ----------
    input_data: np.ndarray
        is the train data of the shape: (n_samples, n_dimensions)

    kwargs
        are parameters passed into the sklearn `KernelDensity`, in this case
        the argument `kernel` is not specified.
        For definition of `KernelDensity` argument see:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html

    Returns
    -------
    sklearn `kde` estimator

    """
    _kde = KernelDensity(**kwargs)

    _kde.fit(input_data)

    return _kde


def evaluate_kde_estimator(fitted_estimator: 'sklearn `kde` estimator', input_data: np.ndarray,
                           exponentialize: bool = True) -> np.ndarray:
    """

    Parameters
    ----------
    fitted_estimator: 'sklearn `kde` estimator'

    input_data: np.ndarray

    exponentialize: bool


    Returns
    -------

    """
    _result = fitted_estimator.score_samples(input_data)

    return np.exp(_result) if exponentialize else _result


'''

C L A S S E S

'''


class UnionFind:
    """
    ADD DOCSTRING

    Attributes
    ----------
    """
    def __init__(self, root_weights: np.ndarray):
        self._weight_of_root = root_weights
        self._object_id_to_parent_id = {}

        self._id_to_object = {}
        self._object_to_id = {}

    '''
    H E L P E R   F U N C T I O N S
    '''
    @property
    def object_id_to_parent_id(self) -> dict:
        return self._object_id_to_parent_id

    def insert_objects(self, objects: Iterator['object']):
        """

        Parameters
        ----------
        objects

        Returns
        -------

        """

        for _object in objects:
            _ = self.find(_object)

    '''
    M A I N  F U N C T I O N S
    '''

    def find(self, object_1: object) -> object:
        """

        Parameters
        ----------
        object_1

        Returns
        -------

        Notes
        -----

        Examples
        --------

        """
        if object_1 not in self._object_to_id:

            # this will determine the unique ID for a new object
            # since the object was not among `_object_to_id` set, it must be a new root
            _new_root_id = len(self._object_to_id)
            self._object_to_id[object_1] = _new_root_id

            # this creates an inverse dictionary, for given unique id you will get an object
            self._id_to_object[_new_root_id] = object_1

            # this creates the `_parent_ident` pointer
            # i.e. the new object is a self root
            self._object_id_to_parent_id[_new_root_id] = _new_root_id

            return object_1

        # ok if the object is in the `self._object_to_id` dictionary
        # this will basically look for the root
        # _list_of_nodes_id: holds a list of nodes you are searching for root
        # you start with the current id
        _list_of_nodes_id = [self._object_to_id[object_1]]

        # now look for the parent id
        _parent_id = self._object_id_to_parent_id[_list_of_nodes_id[-1]]

        # this looks for the biggest _parent_ident id
        while _parent_id != _list_of_nodes_id[-1]:
            _list_of_nodes_id += [_parent_id]

            _parent_id = self._object_id_to_parent_id[_parent_id]

        # it is an instance of lazy unions since it flattens up all tree members during the search
        for _node_id in _list_of_nodes_id:
            self._object_id_to_parent_id[_node_id] = _parent_id

        return self._id_to_object[_parent_id]

    def union(self, object_1: object, object_2: object):
        """

        Parameters
        ----------
        object_1: object

        object_2: object

        Returns
        -------

        """

        # -- step 0 -- look for the roots of the object_1 and object_2
        _root_1, _root_2 = self.find(object_1), self.find(object_2)

        # -- step 1 -- if roots are equal you are done
        # if they are not equal you must merge them
        # who is the root after the merge depends on the weight of the root
        # the smaller root weight is merged into the bigger root weight
        if _root_1 != _root_2:

            # look for the root metadata, like root's id and weights
            _root_1_id = self._object_to_id[_root_1]
            _root_2_id = self._object_to_id[_root_2]

            _root_weight_1 = self._weight_of_root[_root_1_id]
            _root_weight_2 = self._weight_of_root[_root_2_id]

            # -- step 2 -- do the swap between the roots if _root_weight_1 is smaller
            if _root_weight_1 < _root_weight_2:
                _root_1, _root_2, _root_1_id, _root_2_id =\
                _root_2, _root_1, _root_2_id, _root_1_id

            # after the swap we have a guarantee that whatever is stored in _root_1 is
            # a correct biggest root.

            # also in principle we can erase _root_2 from the set of roots
            #del(self._weight_of_root[_root_2_id])

            # -- step 3 -- point _root_2_id to a new root == _root_1_id
            self._object_id_to_parent_id[_root_2_id] = _root_1_id


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
                 distance_function: Union[Callable[[List[float], List[float]], float], None] = None):

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
        assert isinstance(distance_function, (type(None), Callable)),\
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
            is the array of `labels` associated with each data point in `points`

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


class ClusterGenerator:
    """
    ADD DOCSTRING
    """
    # here you keep the allowed structures
    _allowed_structures = {'anisotropy', 'variance', 'circles', 'moons', 'random',
                           'default'}

    def __init__(self, n_samples: int = 1500, randomize: int = 42):
        self._n_samples = n_samples
        self._randomize = randomize

    '''
    M A I N  F U N C T I O N S
    '''
    def generate(self, structure: str) -> np.ndarray:

        assert structure in ClusterGenerator._allowed_structures, f'Your structure {structure} ' \
            f'is not among the allowed structures'

        if structure == 'anisotropy':

            x, y = make_blobs(n_samples=self._n_samples, random_state=self._randomize)
            vec = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]

            return np.dot(x, vec), y

        elif structure == 'variances':

            _std = [1.0, 2.5, 0.5]
            return make_blobs(n_samples=self._n_samples, cluster_std=_std, random_state=self._randomize)

        elif structure == 'circles':

            return make_circles(n_samples=self._n_samples, factor=0.5, noise=0.05)

        elif structure == 'moons':

            return make_moons(n_samples=self._n_samples, noise=0.05)

        elif structure == 'random':

            return np.random.rand(self._n_samples, 2), None

        elif structure == 'default':

            return make_blobs(n_samples=self._n_samples, random_state=self._randomize)
        else:
            raise ValueError('Something horrible has happened, notify the project owner.')



