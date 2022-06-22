import warnings
import numpy as np
import networkx as nx
import dionysus as ds

from scipy.spatial.distance import squareform, pdist
from typing import List, Set, Tuple, Dict, Union
from datatypes import Number, Node, Edge, Vertex

import graphutils as gu
import bacchus as bs


###############################################################################
#     RIPS FILTRATION FROM GRAPH OR MATRIX
###############################################################################
def fill_rips_from_graph(G, **kwargs):
    # OPTIONS -------------------------------------------------------
    distance_func = kwargs.get('distance_func', gu.get_distance_matrix)
    weight = kwargs.setdefault('weight', 'weight')      # edge field that contains weights
    retdist = kwargs.get('retdist', False)              # return distance matrix

    # LOGIC ---------------------------------------------------------
    if G.is_directed():
        warnings.warn(
            "The graph needs to be undirected for the construction of a Rips complex"
        )

    dist_matrix = distance_func(G, **kwargs)
    simplices = fill_rips_from_distances(dist_matrix, **kwargs)

    # return the distance matrix
    if retdist:
        return simplices, dist_matrix

    return simplices


def fill_rips_from_distances(dist, **kwargs):
    # OPTIONS -------------------------------------------------------
    maxdim = kwargs.get('maxdim', 1)
    thresh = kwargs.get('thresh', None)
    values = sorted(np.unique(np.asarray(dist)))

    # LOGIC ---------------------------------------------------------
    if dist.shape[0] != dist.shape[1]:
        raise ValueError("The data should be a square distance matrix")

    if len(values) <= 1:
        raise ValueError(f"Distance matrix has one unique value, vals={values}")

    if not thresh:
        # values are already sorted
        thresh = values[-2] if np.isinf(values[-1]) else values[-1]

    # Generate rips filtration
    dist = squareform(dist)
    simplices = ds.fill_rips(dist, maxdim + 1, thresh)
    simplices = bs.SlicingFiltration(simplices)
    return simplices


###############################################################################
#     CYCLER
###############################################################################
class Cycler:
    """ Build a rips diagram of the data and and provide access to cycles.

        This class wraps much of the functionality in Dionysus, giving it a clean interface and providing a way to access cycles.

        Warning: This has only been tested to work for 1-cycles. 
    """
    def __init__(self, **kwargs):
        self._maxdim = kwargs.get('maxdim', 1)
        self._thresh = kwargs.get('thresh', None)
        self._coeff: int = kwargs.get('coeff', 2)

        self._dist_matrix = None
        self._diagram = None
        self._filtration: 'SlicingFiltration' = None    # filtration of Rips complex
        self._persistence: ds.ReducedMatrix = None      # RM for <complex + cone vertex>
        self._nodes: List[Node] = None                  # list : vertices -> nodes : list(nodes)[vertices]
        self._vertices: Dict[Node, Vertex] = None       # dict : nodes -> vertices : {nodes: vertices}

        self.data = None  # distance matrix
        self.data_type: str = None  # {'graph', 'distance_matrix', 'point_cloud'}
        self.barcode = None
        self.cycles: Dict['PersistenceInterval', List[ds.Simplex]] = None
        pass

    @property
    def thresh(self):
        if self._thresh is not None:
            return self._thresh
        if self._dist_matrix is not None:
            self._thresh = gu.get_max_dist(self._dist_matrix)
            return self._thresh
        return np.inf

    @property
    def maxdim(self):
        return self._maxdim

    @property
    def coeff_field(self):
        return self._coeff

    @property
    def nodes(self):
        return self._nodes

    def fit(self, data, distance_matrix=False, graph=False, **kwargs):
        """ Generate Rips filtration and cycles for data.
        """
        simplices = []
        self.data = data

        if not distance_matrix and not graph:
            self.data_type = 'point_cloud'
            self._dist_matrix = squareform(pdist(data))
            self._nodes = kwargs.get('nodes', [f"P{i:02d}" for i in range(len(self._dist_matrix))])
            simplices = fill_rips_from_distances(
                self._dist_matrix, maxdim=self.maxdim, thresh=self.thresh
            )

        if distance_matrix and graph:
            raise ValueError(
                f"Both 'distance_matrix' and 'graph' options are `True`. "
                f"Only one should be `True`."
            )

        if distance_matrix:
            self.data_type = 'distance_matrix'
            self.data = data
            self._dist_matrix = data
            self._nodes = kwargs.get('nodes', [f"N{i:02d}" for i in range(len(data))])
            simplices = fill_rips_from_distances(
                self._dist_matrix, maxdim=self.maxdim, thresh=self.thresh
            )

        if graph:
            self.data_type = 'graph'
            self.data = data
            simplices, dist_matrix = fill_rips_from_graph(data, retdist=True, **kwargs)
            self._dist_matrix = dist_matrix
            labels = kwargs.get('labels', 'nodes')      # G.nodes field with node labels
            if labels == 'nodes':
                self._nodes = list(data.nodes)
            else:
                self._nodes = list(kwargs.get('nodes', [l for _, l in data.nodes(labels)]))

        self._vertices = {n: i for i, n in enumerate(self.nodes)}
        self._thresh = gu.get_max_dist(self.data) if not self._thresh else self._thresh
        self.from_simplices(simplices)


    def from_simplices(self, simplices):
        if not isinstance(simplices, bs.SlicingFiltration):
            simplices = bs.SlicingFiltration(simplices)

        # Compute persistence diagram
        self._add_cone_vertex(simplices)
        persistence = ds.homology_persistence(simplices, self._coeff)
        diagrams = ds.init_diagrams(persistence, simplices)

        # Set all the results
        self._filtration = simplices
        self._persistence = persistence
        self._diagram = [
            bs.PersistenceInterval(i, persistence=persistence)
            for i in diagrams[self.maxdim]
        ]
        self.barcode = np.array([(d.birth, d.death) for d in self._diagram])

        self._build_cycles()

    @staticmethod
    def _add_cone_vertex(simplices):
        """
        Add cone point to force homology to finite length;
        Dionysus only gives out cycles of finite intervals
        """
        spxs = [ds.Simplex([-1])] + [c.join(-1) for c in simplices]
        for spx in spxs:
            spx.data = np.inf       # was: spx.data = 1
            simplices.append(spx)
        return simplices

    def _build_cycles(self):
        """Create cycles from the diagram of order=self.order
        """
        cycles = {}

        intervals = sorted(self._diagram, key=lambda d: d.death-d.birth, reverse=True)

        for interval in self._diagram:
            if self._persistence.pair(interval.data) != self._persistence.unpaired:
                cycle_raw = self._persistence[self._persistence.pair(interval.data)]

                # Break dionysus iterator representation so it becomes a list
                cycle = [s for s in cycle_raw]
                cycle = self._data_representation_of_cycle(cycle)
                cycles[interval.data] = cycle

        self.cycles = cycles

    def _data_representation_of_cycle(self, cycle_raw: List[ds.ChainEntry]) -> List['IndexedSimplex']:
        cycle = [bs.IndexedSimplex(self._filtration[s.index], index=s.index) for s in cycle_raw]
        return cycle

    def get_cycle(self, interval):
        """Get a cycle for a particular interval. Must be same type returned from `longest_intervals` or entry in `_diagram`.
        """

        return self.cycles[interval.data]
    
    def get_all_cycles(self):
        return list(self.cycles.values())
    
    def longest_intervals(self, n):
        """Return the longest n intervals. For all intervals, just access diagram directly from _diagram.
        """

        intervals = sorted(self._diagram, key=lambda d: d.death-d.birth, reverse=True)
        return intervals[:n]
    
    def order_vertices(self, cycle):
        """ Take a cycle and generate an ordered list of vertices.

            This representation is much more useful for analysis.
        """
        ordered_vertices = [cycle[0][0], cycle[0][1]]
        next_row = 0

        # TODO: how do I make this better? It seems so hacky
        for _ in cycle[1:]:
            next_vertex = ordered_vertices[-1]
            rows, cols = np.where(cycle == next_vertex)
            which = np.where(rows != next_row)
            next_row, next_col = rows[which], (cols[which] + 1) % 2

            ordered_vertices.append(cycle[next_row,next_col][0])
        
        return ordered_vertices

