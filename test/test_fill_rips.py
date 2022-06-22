import networkx as nx
import pytest
import numpy as np
import dionysus as ds

from cyclonysus import graphutils as gu
from cyclonysus import fill_rips_from_distances
from cyclonysus import fill_rips_from_graph


class TestFillRips:

    def test_rips_from_distance_matrix(self):
        G = nx.read_edgelist(
            "data/simple-graph.edgelist.csv",
            delimiter=",",
            nodetype=str,
            data=[('weight', int)]
        )

        print(f"\n{G.nodes}")
        print(f"{G.edges.data()}")

        # cycles
        D = gu.get_distance_matrix(G, weight='weight')
        simplices = fill_rips_from_distances(D, maxdim=1)

        assert simplices
        assert len(simplices) == 27

        # simplices
        for spx in simplices:
            print(spx)
        pass

    def test_rips_from_graph(self):
        G = nx.read_edgelist(
            "data/simple-graph.edgelist.csv",
            delimiter=",",
            nodetype=str,
            data=[('weight', int)]
        )

        simplices = fill_rips_from_graph(
            G,
            distance_func=gu.get_distance_matrix,
            weight='weight',
            maxdim=1
        )
        for spx in simplices:
            print(f"{str(spx):<11} :: {simplices.index(spx):>2}")
        pass
