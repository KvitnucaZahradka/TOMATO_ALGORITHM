import os
import sys
# import scipy as sp
from scipy.spatial import distance

from sklearn.datasets import make_blobs, make_moons, make_circles

import numpy as np

import networkx as nx
from itertools import product

from itertools import combinations
#from scipy.s_parent_idse import dok_matrix
from operator import add

from sklearn.neighbors import BallTree

# ---- FUNCTIONS ----


def fit_kde_estimator(input_data: np.ndarray,
                      bandwidth: 'np.linspace' = np.linspace(0.1, 1.0, 30),
                      cross_validation: int = 2,
                      verbose: bool = False,
                      **kwargs) -> np.ndarray:
    '''
    [] DESCRIPTION []
        []
    <> _parent_idAMETERS <>
        <> X:np.ndarray = X numpy array of shape (n_samples, n_features)
        <>** cross_validation: int = is the number of cross validation to do during the fitting
            DEFAULT: 20
        <>** bandwidth_searchspace: np.linspace = np.linspace(0.01, 1.0, 30) is the linspace for bandwidth,
            DEFAULT: np.linspace(0.1, 1.0, 30)
        <>** verbose: bool = whether the algorithm should be verbose
            DEFAULT: False
        <>** other__parent_idams = the same as sklearn constructor for `KernelDensity`

    >< RETURNS><
        >< fitted sklearn KDE estimator
    ! NOTES !
        ! NOTE, we were following: https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    '''
    # ---- IMPORT -----------------------------------
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    # ---- CODE -------------------------------------
    _grid = GridSearchCV(KernelDensity(kernel='gaussian', **kwargs),
                         {'bandwidth': bandwidth},
                         cv=cross_validation)

    # FIT THE `input_data` TO PRODUCE BEST `KDE`
    _grid.fit(input_data)

    if verbose:
        print('BEST BANDWIDTH: {}'.format(_grid.best__parent_idams_))

    return _grid.best_estimator_


def fit_single_kde_estimator(input_data: np.ndarray, **kwargs):
    # ---- IMPORT -----------------------------------
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    # ---- CODE -------------------------------------
    _kde = KernelDensity(**kwargs)

    _kde.fit(input_data)

    return _kde


def evaluate_kde_estimator(fitted_estimator: 'sklearn `kde` estimator',
                           input_data: np.ndarray)-> np.ndarray:
    '''
    <> _parent_idAMETERS <>
        <> fitted_estimator: sklearn `kde` estimator
        <> input_data: np.ndarray

    >< RETURNS ><
        >< np.ndarray =  is exponentialized

    '''
    # ----- CODE --------------------------------------
    # since it is the `kde` estimator you must exponentialize it
    return np.exp(fitted_estimator.score_samples(input_data))


class Simplicial_complex():

    def __init__(self, simplices: list = []):
        '''
        ADD DOCSTRING
        '''
        # ----- CODE --------------------------------
        self.import_simplices(simplices=simplices)

    '''
    HELPER FUNCTIONS
    '''

    def import_simplices(self, simplices: list = []):
        '''
        ADD DOCSTRING
        '''
        # ----- CODE --------------------------------

        self._simplices = map(lambda simplex: tuple(sorted(simplex)), simplices)

    '''
    MAIN FUNCTIONS
    '''

    def faces(self):
        '''
        ADD DOCSTRING
        ! NOTE:
            the notion of `faces` is different from the notion of `boundary`

        '''
        # ----- CODE --------------------------------
        self._faceset = set()
        for simplex in self._simplices:
            _numnodes = len(simplex)

            for r in range(_numnodes, 0, -1):
                for face in combinations(simplex, r):
                    self._faceset.add(face)

        return self._faceset

    def n_faces(self, dim: int):
        '''
        ADD DOCSTRING

        '''
        # ----- CODE --------------------------------
        return filter(lambda face: len(face) == dim+1, self.face_set)


class Vietoris_Rips_complex(Simplicial_complex):

    def __init__(self, points,
                 epsilon,
                 labels=None,
                 distfcn=distance.euclidean):
        '''
        ADD DOCSTRING
        '''
        # ------ CODE --------------------------------------------------------
        self._pts = points
        self._labels = range(len(self._pts)) if labels == None or\
            len(labels) != len(self._pts) else labels

        self._epsilon = epsilon
        self._distfcn = distfcn

        self.network = self.construct_network(
            self._pts, self._labels, self._epsilon, self._distfcn)

        #self.import_simplices(map(tuple, list(nx.find_cliques(self.network) )))
        self.import_simplices(map(tuple, nx.find_cliques(self.network)))

    '''
    HELPER FUNCTIONS
    '''

    def print_complex(self):
        print(list(nx.find_cliques(self.network)))

    def construct_network(self,
                          points,
                          labels,
                          epsilon,
                          distfcn):
        '''
        ADD DOCSTRING
        '''
        g = nx.Graph()
        g.add_nodes_from(labels)

        zips, spiz = zip(points, labels), zip(points, labels)

        for pair in product(zips, spiz):

            if pair[0][1] != pair[1][1]:
                dist = distfcn(pair[0][0], pair[1][0])
                if dist and dist < epsilon:
                    g.add_edge(pair[0][1], pair[1][1])

        return g


class Union_find():
    '''
    this class implements the Union - find data structure


    '''
    # Initialization

    def __init__(self):

        self.weight_of_root = {}
        self._object_id_to_parent_id = {}

        self.id_to_object = {}
        self.objects_to_id = {}

    # Insert objects among the already existing ones
    def insert_objects(self, objects: "iterable over `objects`"):
        '''
        add docstring
        '''
        # ------- CODE ---------------------------
        for object in objects:
            _ = self.find(object)

    def is_object_in(self, object)->bool:

        return object in self.objects_to_id

    # Find a given object / build it if non-existing

    def find(self, object)->'object':
        '''

        <> object: must be hashable//lookable object

        >< ALWAYS RETURNS THE OBJECT
        this finds an object and returns the object if exists
        if object does not exist, it will put the object into the data structure
        '''
        # ------ CODE ------------------------------
        if not object in self.objects_to_id:

            # this will determine the unique ID for a new object
            # since the object was not among `object_to_id` set, it must be a new root
            _new_root_id = len(self.objects_to_id)
            self.objects_to_id[object] = _new_root_id

            # this means that the new root has only one (self) branch
            # in general the weight of the root of the tree measures how many
            # objects are attached to the root
            self.weight_of_root[_new_root_id] = 1

            # this creates the inverse dictionary, for given unique id you will get the object
            self.id_to_object[_new_root_id] = object

            # this creates the _parent_ident pointer
            # since this is a new root, roots _parent_ident is always root
            self._object_id_to_parent_id[_new_root_id] = _new_root_id

            return object

        # ok if the object is in the object dictionary
        # this basically looks for the root
        #
        # _list_of_nodes_id: stores the list of nodes as you are searching for root
        # you start with the current id
        _list_of_nodes_id = [self.objects_to_id[object]]

        # now look for parent id
        _parent_id = self._object_id_to_parent_id[_list_of_nodes_id[-1]]

        # this looks for biggest _parent_ident id
        while _parent_id != _list_of_nodes_id[-1]:
            _list_of_nodes_id += [_parent_id]

            _parent_id = self._object_id_to_parent_id[_parent_id]

        # it is the lazy type of union since it flattens up all the tree members
        # and all of them ave the same root
        #
        # this basically flattens the search tree
        for _node_id in _list_of_nodes_id:
            self._object_id_to_parent_id[_node_id] = _parent_id

        # this returns the root object
        return self.id_to_object[_parent_id]

    # Link two different objects in a same distinct set
    def union(self, object_1, object_2):
        '''
        add docstring
        '''

        # this looks for a roots of object_1 and object_2
        _root_1, _root_2 = self.find(object_1), self.find(object_2)

        # if roots are equal ... you are done
        # if the are different, you must to merge them
        # who is the root after merge depends on the weight of the root
        # the smaller root weight is merged into the bigger root weight
        if _root_1 != _root_2:

            # this looks for the root metadata, like root id`s and roots weights
            _root_1_id = self.objects_to_id[_root_1]
            _root_2_id = self.objects_to_id[_root_2]

            _root_weight_1 = self.weight_of_root[_root_1_id]
            _root_weight_2 = self.weight_of_root[_root_2_id]

            # just doing swap between roots if _root_weight_1 is smaller
            # SWAP
            if _root_weight_1 < _root_weight_2:
                _root_1, _root_2, _root_1_id, _root_2_id, _root_weight_1, _root_weight_2 =\
                    _root_2, _root_1, _root_2_id, _root_1_id, _root_weight_2, _root_weight_1

            # after swap we have guaranteed that whatever is stored in `root_1` is corerct biggest root
            self.weight_of_root[_root_1_id] = _root_weight_1 + _root_weight_2

            # we know that we can erase `_root_2` from the set that is reserved only for roots
            del(self.weight_of_root[_root_2_id])

            # and point _root_2_id to a new root == root_1_id
            self._object_id_to_parent_id[_root_2_id] = _root_1_id


class Tomato():

    def __init__(self, X: np.ndarray, density_estimation: str, **kwargs):
        '''
        <> PARAMETERS <>
            <> X: np.ndarray = data should be normalized to have values between <-1, 1> in all dimensions
                                reason for that is that then the largest scale is
            <> density_estimation: str =  is a restricted string {'kde', 'local_kde'}
                `kde` = is the global kde estimator
                `local_kde` = is knn based kde estimator, you need to provide `n_neighbors` parameter
        '''

        # ------ CODE ------------------------
        '''
        (I) YOU MUST FIT THE THE KDE DENSITIES
        '''
        #
        print('fitting densities')
        if density_estimation == 'kde_gauss_grid_search':
            _densities = evaluate_kde_estimator(fit_kde_estimator(
                X, cross_validation=kwargs.get('cross_validation', 2)), X)

        elif density_estimation == 'kde_tophat':
            _densities = evaluate_kde_estimator(fit_single_kde_estimator(X, kernel='tophat'), X)

        else:
            raise NotImplementedError

        '''
        (II) FIND THE ORDERING ON THE DATA, I.E. CREATE THE \tilde{f} FUNCTION
        '''
        # `_data_store` stores a list of ordered data by the point density estimates in non-increasing fashion
        _data_store = sorted(zip(X.tolist(), _densities.tolist()),
                             key=lambda x: -x[-1])

        '''
        (III) CREATE ORDERED DATA AS WELL AS THE `VR` COMPLEX
        '''

        # lets get just ordered densities, that will serve for quick lookup of pseudogradinets
        # as `tilde_f`
        self._tilde_f = np.array(list(map(lambda x: x[1], _data_store)))

        # extract only the ordered data
        self._ordered_data = np.array(list(map(lambda x: x[0], _data_store)))

        # FIX THIS ::: CREATE AN ALGORITHM THAT CAN FIND EPSILON AUTOMATICALLY
        # this can be done by looking for such an epsilon that ::: <- look into the paper
        #self.refit_vietoris_rips_graph(epsilon = kwargs.get('VR_EPSILON', 0.8))

    def fit_vietoris_rips_graph(self, epsilon: float):
        '''
        ADD DOCSTRING
        '''
        self._graph_type = 'vietoris_rips_complex'
        self._graph = Vietoris_Rips_complex(
            self._ordered_data, epsilon=epsilon,
            labels=list(range(self._ordered_data.shape[0])))

    def fit_knn_graph(self, n: int, **kwargs):
        '''
        ADD DOCSTRING
        '''
        self._graph_type = 'knn_complex'
        self._num_neighbors = n
        self._graph = BallTree(self._ordered_data, leaf_size=kwargs.get('leaf_size', 42))

    # actual

    def fit(self, tau: float = 1e-2)->'union find object':
        '''
        ADD DOCSTRING
        '''
        # ------ CODE ------------------------------
        # create UNION-FIND data sctructure
        _U = Union_find()

        # create the `g` vector == the pseudo gradient vector
        #_g_vector = np.full((len(self._tilde_f),), -1, np.int64)

        # create the `r` vector == the root vector
        #_r_vector = np.full((len(self._tilde_f), ), -1, np.int64)

        # at the beginning is every index its own root
        #_r_vector = np.array(list(range(len(self._tilde_f))))

        # this will be useful for plotting
        # well I will store (dens, True) = it means cluster was born
        # well I will store (dens, False) = it means that some density died
        _persistence_data = {}
        for idx in range(len(self._tilde_f)):

            '''
            (I) FIND NEIGHBORHOOD SET
            '''
            # returns the neighborhood of indices that have HIGHER densities than current idx :: I.E. PSEUDO-GRADIENTS
            # i.e. they have lower indices than the current index

            if self._graph_type == 'vietoris_rips_complex':
                print('using {} graph'.format(self._graph_type))
                _N = np.array(
                    list(filter(lambda ind: ind < idx, self._vietoris_rips_graph.network[idx])))

            elif self._graph_type == 'knn_complex':
                _dist, _ind = self._graph.query(self._ordered_data[idx: idx + 1],
                                                k=self._num_neighbors)

                _N = np.array(list(filter(lambda ind: ind < idx, _ind[0])))
            else:
                raise ValueError('GRAPH NOT FOUND.')

            '''
            (II) CREATE UNION FIND // UPDATE UNION FIND
            '''
            # cluster is born

            if self._tilde_f[idx] not in _persistence_data:

                # the if statement should avoid zeroing something that was there before,
                # mathematically this should `almost` never happen, but because comp, it might
                _persistence_data[self._tilde_f[idx]] = 0.0

            if _N.size > 0:

                # if _N is not empty ::: then find the largest root and by neighbor gradients
                _pseudogradient = _N[np.argmax(self._tilde_f[_N])]

                # find root for `_pseudogradient`
                _parent = _U.find(_pseudogradient)

                # do `UNION` of idx and _parent
                _U.union(_parent, idx)

                # also means that `idx` density dies, at the level _tilde_f[idx]
                _persistence_data[self._tilde_f[idx]] = self._tilde_f[idx]

                for j in _N:
                    # find root for j
                    _parent_j = _U.find(j)

                    # this is the condition when you decided to what root current node belongs
                    _parents_root_densities = [self._tilde_f[_parent], self._tilde_f[_parent_j]]

                    # what this means?
                    # this means that parent densities are different and
                    if _parent != _parent_j and min(_parents_root_densities) < self._tilde_f[idx] + tau:

                        # only in this case conglomerate `parent_j` and `_parent`
                        # means that `_parent` density dies
                        _U.union(_parent_j, _parent)

                        # this means that _parent density was killed at the level _tilde_f[idx]
                        _persistence_data[self._tilde_f[_parent]] = self._tilde_f[idx]

                        # update `_parent`
                        _parent = _U.find(_parent_j)

            else:
                # if _N is empty :: then add `idx` into the union find data structure

                _U.insert_objects([idx])

                # also store the root for `idx` in root vector
                #_r_vector[idx] = idx

        return _U, _persistence_data


class ClusterGenerator:

    # Initialization
    # structure refers to the type of data to generate
    # n_samples refers to the amount of data to deal with
    # randomize is the random state for reproducibility
    def __init__(self, structure='blobs', n_samples=1500, randomize=42):

        self.structure = structure
        self.n_samples = n_samples
        self.randomize = randomize

    # Function aiming at generating samples
    def generate(self):

        if self.structure == 'anisotropy':

            x, y = make_blobs(n_samples=self.n_samples, random_state=self.randomize)
            vec = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
            return np.dot(x, vec), y

        elif self.structure == 'variances':

            std = [1.0, 2.5, 0.5]
            return make_blobs(n_samples=self.n_samples, cluster_std=std, random_state=self.randomize)

        elif self.structure == 'circles':

            return make_circles(n_samples=self.n_samples, factor=0.5, noise=0.05)

        elif self.structure == 'moons':

            return make_moons(n_samples=self.n_samples, noise=0.05)

        elif self.structure == 'random':

            return np.random.rand(self.n_samples, 2), None

        else:

            return make_blobs(n_samples=self.n_samples, random_state=self.randomize)
