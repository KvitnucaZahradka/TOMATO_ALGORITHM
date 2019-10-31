
from scipy.spatial import distance

from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.neighbors import BallTree, NearestNeighbors

import numpy as np
import networkx as nx

from itertools import product
from itertools import combinations

'''
F U N C T I O N S
'''


def fit_gaussian_kde_estimator(input_data: np.ndarray,
                               bandwidth: 'np.linspace' = np.linspace(0.1, 1.0, 30),
                               cross_validation: int = 2,
                               verbose: bool = False,
                               **kwargs) -> 'sklearn estimator':
    """This convenience function is taking the input data and returns the best gaussian
    kde estimator after running the grid search with desired number of cross validations.
    We were following blog `https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/`

    Parameters
    ----------
    input_data:np.ndarray
        is the numpy array with the data of the shape: (n_samples, n_dimensions)

    bandwidth:'np.linspace'
        is an instance of numpy linspace, it is the parameter of gridsearch for
        gaussian kde parameter `bandwidth`, default is np.linspace(0.1, 1.0, 30)

    cross_validation:int
        is the number of folds to consider during the grid searchin
        default is 2

    verbose:bool
        arameter indicating whether this method should print some messages
        default is False

    **kwagrs
        is the optional parameter(s) dictionary for the gaussian kde fitting
        (see sklearn kde gaussian kernel fitting)

    Returns
    -------
        grid searched and best gaussian kde estimator sklearn estimator

    Examples
    --------
        <ADD DOCTEST RUNNABLE EXAMPLES>

    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    _grid = GridSearchCV(KernelDensity(kernel='gaussian', **kwargs),
                         {'bandwidth': bandwidth},
                         cv=cross_validation)

    # FIT THE `input_data` TO PRODUCE BEST `KDE`
    _grid.fit(input_data)

    if verbose:
        print('Best bandwidth: {}'.format(_grid.best__parent_idams_))

    return _grid.best_estimator_


def fit_single_kde_estimator(input_data: np.ndarray, **kwargs)->'sklearn `kde` estimator':
    """This convenience function fits single (NOT grid search) sklearn `kde` estimator

    Parameters
    ----------
    input_data:np.ndarray
        is the train data of the shape (n_samples, n_dimensions)

    **kwargs
        optional parameters dictionary, see `sklearn.neighbors.KernelDensity` for options

    Returns
    -------
    'sklearn `kde` estimator'

    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    _kde = KernelDensity(**kwargs)

    _kde.fit(input_data)

    return _kde


def evaluate_kde_estimator(fitted_estimator: 'sklearn `kde` estimator',
                           input_data: np.ndarray,
                           exponentialize: bool = True)-> np.ndarray:
    """This function evaluates fitted `kde` estimator.

    Parameters
    ----------
    fitted_estimator:'sklearn `kde` estimator'
        is fitted `sklearn` `kde` estimator

    input_data:np.ndarray
        is the data set of the shape (n_examples, n_dimensions)

    exponentialize:bool
        is boolean, if True then we will exponentialize the density estimator

    Returns
    -------
    np.ndarray
        the final version of the density estimator
    """
    if exponentialize:
        # since it is the `kde` estimator is logarithmic, you must exponentialize it
        return np.exp(fitted_estimator.score_samples(input_data))
    else:
        # this returns the (logarithmic) estimator
        return fitted_estimator.score_samples(input_data)


'''
C L A S S E S
'''


class SimplicialComplex():
    """This class implements simple realization of `Simplical complex`
    """

    def __init__(self, simplices: tuple = ()):
        """Constructor of Simplical complex

        Parameters
        ----------
        simplices:tuple
            is a tuple of tuples, where individual tuple determine individual p-simplex (unoriented)

        """
        self.import_simplices(simplices=simplices)

    '''
    H E L P E R   F U N C T I O N S
    '''

    def import_simplices(self, simplices: tuple = ()):
        """
        Parameters
        ----------
        simplices:tuple
            is a tuple of tuples, where individual tuple determine individual p-simplex (unoriented)

        Returns
        -------
        None
            this function returns None, but changes the state of `self._simplices`
            to a generator over Simplical complex

        Examples
        --------
        """
        self._simplices = map(lambda simplex: tuple(sorted(simplex)), simplices)

    '''
    M A I N   F U N C T I O N S
    '''

    def faces(self)->set:
        """
        Returns
        -------
        This function returns the set of faces of given Simplical complex.

        Note
        ----
        Note, the notion of face is DIFFERENT from the notion of `boundary`!
        """
        self._faceset = set()
        for simplex in self._simplices:
            _num_nodes = len(simplex)

            for r in range(_num_nodes, 0, -1):
                for face in combinations(simplex, r):
                    self._faceset.add(face)

        return self._faceset

    def n_faces(self, dim: int)->'iterator over faces':
        """This function returns generator over faces of simplical complex.

        Parameters
        ----------
        dim:int
            is positive integer determining dimension of returned faces of the graph.

        Returns
        -------
            iterator over faces of dimension `dim`

        """
        return filter(lambda face: len(face) == dim+1, self.face_set)


class VietorisRipsComplex(SimplicialComplex):
    """Simple implemntation of the Vietoris_Rips_complex
    """

    def __init__(self,
                 points: np.ndarray,
                 epsilon: float,
                 labels: np.ndarray = None,
                 distfcn: 'scipy distance' = distance.euclidean):
        """

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


        """
        self._pts = points
        self._labels = range(len(self._pts)) if labels == None or\
            len(labels) != len(self._pts) else labels

        self._epsilon = epsilon
        self._distfcn = distfcn

        self.network = self.construct_network(self._pts, self._labels,
                                              self._epsilon, self._distfcn)

        # here you are keeping the true realization of Vietoris Rips complex
        self._cliques = None

        #self.import_simplices(map(tuple, list(nx.find_cliques(self.network) )))
        self.import_simplices(map(tuple, nx.find_cliques(self.network)))

    '''
    H E L P E R  F U N C T I O N S
    '''

    def generate_vietoris_rips_complex(self):
        """This function generates full `Vietoris_Rips_complex`.
        Note, this function might take nontrivial time and resources.
        """
        if self._cliques == None:
            self._cliques = np.ndarray(list(nx.find_cliques(self.network)))

    def print_complex(self):
        """This function just prints the full `Vietoris_Rips_complex` and if
        complex is not yet calculated, it calculates it.
        """
        if self._cliques == None:
            self._cliques = self.generate_vietoris_rips_complex()

        print('Vietoris_Rips_complex is :\n\n', self._cliques)

    '''
    M A I N  F U N C T I O N S
    '''

    def construct_network(self,
                          points: np.ndarray,
                          labels: np.ndarray,
                          epsilon: float,
                          distfcn: 'scipy distance function')->'networkx graph':
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
        zips, spiz = zip(points, labels), zip(points, labels)

        # this is a bottleneck, you must itterate over each possible product
        # !! improve in the future (should be able to paralelize this)
        for pair in product(zips, spiz):

            # this means that pair must be distinct to connect it
            if pair[0][1] != pair[1][1]:
                dist = distfcn(pair[0][0], pair[1][0])

                # Vietoris_Rips_complex condition: consider pair to be an edge
                # only if a distance is smaller than `epsilon`
                if dist and dist < epsilon:
                    g.add_edge(pair[0][1], pair[1][1])
        return g


class Union_find():
    """This class implements the Union - find data structure, specifically adjusted
    for usage in `Tomato` algorithm.
    """

    def __init__(self, root_weights: np.ndarray):
        """Constructor of `Tomato algorithm` specific realization of the
        `Union_find` data structure.

        Parameters
        ----------
        root_weights:np.ndarray
            is the numpy ndarray with `root weights`. Root weights
            is the numpy array of the size and relation to the future members
            (possibly) inserted into particular `Union_find` instance.

        """
        self.weight_of_root = root_weights
        self._object_id_to_parent_id = {}

        self.id_to_object = {}
        self.objects_to_id = {}

    '''
    H E L P E R  F U N C T I O N S
    '''

    # Insert objects among the already existing ones
    def insert_objects(self, objects: 'iterable over `objects`'):
        """This function iterates over iterable container of objects and
        add them into the `Union_find` instance memmory

        Parameters
        ----------
        objects:'iterable over `objects`'
            is the iterable over objects that need to be inserted into the
            `Union_find` instance memmory

        Returns
        -------

        Examples
        --------
        """
        for object in objects:
            _ = self.find(object)

    '''
    M A I N  F U N C T I O N S
    '''

    # Find a given object / build it if non-existing
    def find(self, object: 'object')->'object':
        """This function finds the root `object` for a lookup member (also called object)
        If the object (the function key) is not found then this function !ADDS!
        the object (the function key) into the `Union_find` instance memmory as a
        new member (that is its own root, until merge with some other object).

        Parameters
        ----------
        object:'object'
            is the hashable and lookable object (i.e. can be a key in dictionary)

        Returns
        -------
        `object`: is the root of the searched object (the function key) or self if not fond

        Examples
        --------

        """
        if not object in self.objects_to_id:

            # this will determine the unique ID for a new object
            # since the object was not among `object_to_id` set, it must be a new root
            _new_root_id = len(self.objects_to_id)
            self.objects_to_id[object] = _new_root_id

            # this creates the inverse dictionary, for given unique id you will get the object
            self.id_to_object[_new_root_id] = object

            # this creates the _parent_ident pointer
            # since this is a new root, roots _parent_ident is always root
            self._object_id_to_parent_id[_new_root_id] = _new_root_id

            return object

        # ok if the object is in the object dictionary
        # this basically looks for the root
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
    def union(self, object_1: 'object', object_2: 'object'):
        """This function implements the `union` operation of the data structure
        `Union find`. It takes two objects and assign them common `root`, i.e.
        they will forever be in the same `union`.

        Parameters
        ----------
        object_1:'object'
            is the lookable and hashable object
        object_2:'object'
            is lookable and hashable object

        Returns
        -------

        Examples
        --------

        """
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
                _root_1, _root_2, _root_1_id, _root_2_id =\
                    _root_2, _root_1, _root_2_id, _root_1_id

            # after swap we have guaranteed that whatever is stored in `root_1` is corerct biggest root
            #self.weight_of_root[_root_1_id] = _root_weight_1 + _root_weight_2

            # we know that we can erase `_root_2` from the set that is reserved only for roots
            # del(self.weight_of_root[_root_2_id])

            # and point _root_2_id to a new root == root_1_id
            self._object_id_to_parent_id[_root_2_id] = _root_1_id


class Tomato():
    """This class implements the `Tomato` clustering algorithm.
    """

    def __init__(self, X: np.ndarray, density_estimation: str, **kwargs):
        """ The constructor for the `Tomato` clustering algorithm

        Parameters
        ----------
            X:np.ndarray
                is the numpy array of the vectors we want to group of the shape: (n_examples, n_dimensions)

            density_estimation:str
                is the `morse function` estimator, in this implementation it can be
                a true local density estimate, logarithmic density estimate, or their generalisations.
                Implemented density estimators are:
                    `kde_gauss_grid_search`: is kde with Gaussian kernel and grid searched and cross validated.
                    `kde_logarithmic_gauss_grid_search`: is the same as `kde_gauss_grid_search`, except is NOT
                    exponentialized
                    `kde_tophat`: is kde estimator with `tophat` kernel
                    `kde_logarithmic_tophat`: is the same as `tophat` except not exponentialized
                    `kde_logarithmic_knn`: is the logarithmic type of density estimator
                    with density based on k nearest neighbors graph. See paper (section 7.1):
                        http://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf

            **kwargs: depending on kernel density estimator:
                - for estimators `kde_gauss_grid_search`, `kde_tophat` (and their logarithmic versions)
                user can setup: `bandwidth` that is `np.linspace`, default is: np.linspace(0.1, 1.0, 30)
                and `cross_validation`:int, default is 2.
                - for estimators `kde_tophat` (and its log) one can set up
                kwargs that are coonsistent with sklearn `KernelDensity` class constructor.
                - for estimator `kde_logarithmic_knn` user can set up
                log_regularizer:float, positive float, that serves as convenience for
                problematic data points, when np.log(data_point) tend to be ill defined
                default is 0.0
                leaf_size: int, positive int that determines the shape of the `BallTree` in the knn algorithm
                default is 42
                knn_estimator_n_neighbors: int, positive int determining how many neighbors you will consider
                bool_filter: callable, is the boolean filter that takes float and returns bool value,
                default is: lambda x: True

        Returns
        -------

        Examples
        --------
        """

        print('fitting densities')

        # fit the `kde` (or custom) densities
        if density_estimation == 'kde_gauss_grid_search':
            _densities = evaluate_kde_estimator(fit_gaussian_kde_estimator(X, cross_validation=kwargs.get('cross_validation', 2),
                                                                           bandwidth=kwargs.get('bandwidth', np.linspace(0.1, 1.0, 30))), X)

        elif density_estimation == 'kde_logarithmic_gauss_grid_search':
            _densities = evaluate_kde_estimator(fit_gaussian_kde_estimator(X, cross_validation=kwargs.get('cross_validation', 2),
                                                                           bandwidth=kwargs.get('bandwidth', np.linspace(0.1, 1.0, 30))), X, False)

        elif density_estimation == 'kde_tophat':
            _densities = evaluate_kde_estimator(
                fit_single_kde_estimator(X, kernel='tophat', **kwargs), X)

        elif density_estimation == 'kde_logarithmic_tophat':
            _densities = evaluate_kde_estimator(
                fit_single_kde_estimator(X, kernel='tophat', **kwargs), X, False)

        elif density_estimation == 'kde_logarithmic_knn':
            # if necessary, introduce log regularizer
            _log_regularizer = kwargs.get('log_regularizer', 0.0)

            # this is the knn-based density estimation
            _leaf_size = kwargs.get('leaf_size', 42)

            # we need to know from how many neighbors we will do the estimator
            _n_neighbors = kwargs.get('knn_estimator_n_neighbors', 6)

            # create local graph
            _graph = NearestNeighbors(n_neighbors=_n_neighbors, leaf_size=_leaf_size,
                                      algorithm='ball_tree', p=2, n_jobs=os.cpu_count() - 1)
            # fit the graph
            _graph.fit(X)

            # define query function, so to be consistent with other functions
            _graph.query = lambda X, k: _graph.kneighbors(X=X, n_neighbors=k)

            # define log density estimator
            def _log_density_estimator(in_vector, k=_n_neighbors): return np.log(k) - np.log(X.shape[0])\
                - X.shape[1]*np.log(_log_regularizer + _graph.query(in_vector, k=k)[0][0][-1])

            # it might happen that the `_log_density_estimator` is negative, you must shift it to positive values
            _densities = np.array(list(map(lambda ind: _log_density_estimator(X[ind:ind+1]),
                                           range(X.shape[0]))))

            # this shifts the densities to be nonzero
            _densities += np.abs(np.min((0.0, np.min(_densities))))

        else:
            raise NotImplementedError

        # get the boolean filter that works on densities
        # this is useful if your densities have some hyper significant peak
        # and you want to get rid of it
        # default is True
        _bool_filter = kwargs.get('bool_density_filter', lambda x: True)

        # `_data_store` stores a list of ordered data by the point density estimates in non-increasing fashion
        _data_store = sorted(zip(X.tolist(), _densities.tolist()),
                             key=lambda x: -x[-1])

        # apply bool filter, if you have one
        _data_store = list(filter(lambda x: _bool_filter(x[-1]), _data_store))

        # lets get just ordered densities, that will serve for quick lookup of pseudogradinets
        # as `tilde_f`
        self._tilde_f = np.array(list(map(lambda x: x[1], _data_store)))

        # extract only the ordered data
        self._ordered_data = np.array(list(map(lambda x: x[0], _data_store)))

    '''
    H E L P E R   F U N C T I O N S
    '''

    def fit_vietoris_rips_graph(self, epsilon: float):
        """This function fits data with the Vietoris - Rips complex

        Parameters
        ----------
        epsilon:float
            is the positive float parameter that determines the `close` neighborhood of a data point
            all the other data are connected together as well as with the point
            (since the epsilon - `sphere` is supposed to be convex in our Euclidean-like metric)

        Returns
        -------

        Examples
        --------
        """
        self._graph_type = 'vietoris_rips_complex'
        self._graph = Vietoris_Rips_complex(self._ordered_data, epsilon=epsilon,
                                            labels=list(range(self._ordered_data.shape[0])))

    def fit_knn_graph(self, n: int, **kwargs):
        """This function fits the data with the knn-type of graph

        Parameters
        ----------
        n:int
            positive integer that determines the number of neighbors we will consider
            for given data point

        **kwargs
            optional parameter dictionary
            - user can specify:
            `leaf_size`, is a positive integer that determines shape of search tree for `ball tree` algorithm
            default is 42

        Returns
        -------

        Examples
        --------
        """
        self._graph_type = 'knn_complex'
        self._num_neighbors = n

        self._graph = NearestNeighbors(n_neighbors=n, leaf_size=kwargs.get('leaf_size', 42),
                                       algorithm='ball_tree', p=2, n_jobs=os.cpu_count() - 1)
        #self._graph = BallTree(self._ordered_data, leaf_size=kwargs.get('leaf_size', 42))

        self._graph.fit(self._ordered_data)

        # add locally the `query` function so you can be consistent with other algorithms
        self._graph.query = lambda X, k: self._graph.kneighbors(X=X, n_neighbors=k)

    '''
    M A I N   F U N C T I O N S
    '''
    # actual fitting using the `Tomato` algorithm

    def fit(self, tau: float, verbose: bool = False)->tuple:
        """
        Parameters
        ----------
        tau:float
            is the parameter that determines at what level of `prevalence` the
            peaks should be merged

        Returns
        -------
        tuple: (Union_find instance, persistence_data:dict),
            where the `Union_find` instance is the instance of union find with
            root_weights = density estimates
            and `persistence_data` is a dictionary with the density at which the clusters
            are born and densities at which clusters die.


        Examples
        --------
        """
        # create UNION-FIND data sctructure
        _union_find = Union_find(root_weights=self._tilde_f)

        # in persistence_data we will keep information when clusters are born (key)
        # and when they die (corresponding values)
        _persistence_data = {}

        if verbose:
            print('We are using using {} graph'.format(self._graph_type))

        # iterating through the sorted indices and merging clusters as given in `Tomato` paper
        for idx in range(len(self._tilde_f)):
            if verbose:
                print('Calculating index: {}'.format(idx))

            # returns the neighborhood of indices that have HIGHER densities than current idx :: I.E. PSEUDO-GRADIENTS
            # i.e. they have lower indices than the current index
            if self._graph_type == 'vietoris_rips_complex':

                # `_neighborhood` stores the neighborhood of `idx`
                _neighborhood = np.array(list(filter(lambda ind: ind < idx,
                                                     self._vietoris_rips_graph.network[idx])))

            elif self._graph_type == 'knn_complex':

                _dist, _ind = self._graph.query(self._ordered_data[idx:idx + 1],
                                                k=self._num_neighbors)

                _neighborhood = np.array(list(filter(lambda ind: ind < idx, _ind[0])))
            else:
                raise ValueError('GRAPH NOT FOUND.')

            # for persistence diagram you need to remember start peaks
            _start_peaks = set(_union_find._object_id_to_parent_id.values())

            # cluster is born
            if _neighborhood.size > 0:

                # if _neighborhood is not empty ::: then find the largest root and by neighbor gradients
                _pseudogradient = _neighborhood[np.argmax(self._tilde_f[_neighborhood])]

                # find root for `_pseudogradient`
                _parent = _union_find.find(_pseudogradient)

                # do `UNION` of idx and _parent
                _union_find.union(_parent, idx)

                for j in _neighborhood:
                    # find root for j
                    _parent_j = _union_find.find(j)

                    # this is the condition when you decided to what root current node belongs
                    _parents_root_densities = [self._tilde_f[_parent], self._tilde_f[_parent_j]]

                    # what this means?
                    # this means that parent densities are different and you are merging only if persistence
                    # is big enough
                    if _parent != _parent_j and min(_parents_root_densities) < self._tilde_f[idx] + tau:

                        # only in this case conglomerate `parent_j` and `_parent`
                        # means that `_parent` density dies
                        _union_find.union(_parent_j, _parent)

                        # update `_parent`
                        _parent = _union_find.find(_parent_j)
            else:
                # if _neighborhood is empty :: then add `idx` into the union find data structure
                _union_find.insert_objects([idx])

                # now for persistent diagram, you need peeks that survived
                _stop_peaks = set(_union_find._object_id_to_parent_id.values())

                # calclulate what peaks has been killed in the process
                _killed = _start_peaks.difference(_stop_peaks)

                # update the persistence data
                _persistence_data = {**_persistence_data,
                                     **{self._tilde_f[peak]: self._tilde_f[idx]
                                        for peak in _killed}}

        # final persistence data update
        # you need to find out which peaks survived the merging, i.e.
        # are totaly persistent
        _persistence_data = {**_persistence_data, **{self._tilde_f[peak]: 0.0 for
                                                     peak in set(_union_find._object_id_to_parent_id.values())}}

        return _union_find, _persistence_data


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
