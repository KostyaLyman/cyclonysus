import numpy as np
import networkx as nx
import math

from typing import List, Set, Tuple, Dict, Union


###############################################################################
#     DISTANCE MATRICES
###############################################################################
def is_symmetric_matrix(A, **kwargs):
    # OPTIONS -------------------------------------------------------
    decimals = kwargs.get('decimals', 4)
    close = kwargs.get('close', False)

    # LOGIC ---------------------------------------------------------
    if A.shape[0] != A.shape[1]:
        return False
    if 'decimals' in kwargs:
        # do rounding only if 'decimals' was passed
        A = np.array(np.round(A, decimals))

    if close:
        return np.allclose(np.array(A.T), np.array(A))
    else:
        return (A.T == A).all()


def get_distance_matrix(G, weight='weight', **kwargs):
    # OPTIONS -------------------------------------------------------
    force_undir = kwargs.get('force_undir', False)
    decimals = kwargs.get('decimals', 4)

    # LOGIC ---------------------------------------------------------
    if force_undir:
        G = nx.Graph(G)

    # TODO: can we do it without transforming into a dense numpy array?
    D = nx.to_scipy_sparse_array(G, weight=weight).todense().astype(float)
    D[D == 0] = np.inf
    np.fill_diagonal(D, [0]*len(G))
    return np.array(np.round(D, decimals))


def get_path_distance_matrix(G, weight='weight', **kwargs):
    # OPTIONS -------------------------------------------------------
    L0 = kwargs.get('L0', False)
    force_undir = kwargs.get('force_undir', False)
    decimals = kwargs.get('decimals', 4)

    # LOGIC ---------------------------------------------------------
    if force_undir:
        G = nx.Graph(G)

    node_idx = {n: i for i, n in enumerate(G.nodes)}
    D = np.full((len(G), len(G)), np.inf)
    if L0:
        all_pairs_paths = nx.all_pairs_dijkstra(G, weight=None)
    else:
        all_pairs_paths = nx.all_pairs_dijkstra(G, weight=weight)

    for n, (dists, paths) in all_pairs_paths:
        for k, dist in dists.items():
            D[node_idx[n], node_idx[k]] = dist
    np.fill_diagonal(D, [0] * len(G))
    return np.array(np.round(D, decimals))


###############################################################################
#     GRAPH WEIGHTS UTILS
###############################################################################
def minmax(*args) -> Tuple:
    return (min(args), max(args))


def get_weights(G, weight='weight') -> List:
    return [d[2] for d in G.edges.data(weight)]


def get_max_weight(G, weight='weight'):
    M = np.array([d[2] for d in G.edges.data(weight)]).max()
    return M


def get_min_weight(G, weight='weight'):
    m = np.array([d[2] for d in G.edges.data(weight)]).min()
    return m


def get_max_dist(D):
    if len(D) == 0:
        return 0

    values = sorted(np.unique(np.asarray(D)))
    M = values[-2] if np.isinf(values[-1]) else values[-1]
    return M


def set_reversed_weights(G, weight='weight', target_weight=None):
    """
    Reverses the order of edge weights:
        w -> M - w + 1
    where M = max(w)
    """
    target_weight = f"td_{weight}" if not target_weight else target_weight

    M = get_max_weight(G, weight)
    rev = {(u, v): M - w + 1 for u, v, w in G.edges.data(weight)}
    nx.set_edge_attributes(G, rev, target_weight)
    return G


def set_normalized_weights(G, weight='weight', target_weight=None, max_norm_weight=100):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(1, max_norm_weight))
    target_weight = f"norm_{weight}" if not target_weight else target_weight

    n_edges = len(G.edges)
    G_w = {(u, v): w for u, v, w in G.edges.data(weight)}
    G_w = dict(zip(
        G_w.keys(),
        scaler.fit_transform(
            np.array(list(G_w.values()), ndmin=2).reshape(n_edges, 1)
        ).reshape(n_edges)
    ))
    nx.set_edge_attributes(G, G_w, target_weight)

    return G


def set_log_normalized_weights(G, weight='weight', target_weight=None):
    target_weight = f"norm_{weight}" if not target_weight else target_weight
    m = get_min_weight(G, weight)
    G_w = {
        (u, v): math.log(w - m + 1) + 1
        for u, v, w in G.edges.data(weight)
    }
    nx.set_edge_attributes(G, G_w, target_weight)

    return G


def set_chain_weights(G, chain, target_weight='chain_weight'):
    G_w = {
        (u, v): chain.get(minmax(u, v), 0)
        for u, v in G.edges()
    }
    nx.set_edge_attributes(G, G_w, target_weight)
    return G

