"""
Our heavy graph type.
Wraps the original networkx graph in a interface that fits us and
adds all the precomputed things in there.
"""
from uuid import uuid4 as uuid_gen
import networkx as nx
from more_itertools import grouper
import pact.nautyhelper as nautyhelper


def _is_nx_star(nxg):
    degrees = sorted([d for v, d in nxg.degree])
    # all except highest deg must be 1 == second highest deg is 1 (we
    # assume connected graphs with no isolated vertices).
    if degrees[-2] == 1:
        return degrees[-1]
    return None


def _is_Knm(G, n, m):
    if nx.is_bipartite(G.graph) and len(G.E) == n * m and len(G.V) == n + m:
        return True
    return False

def _is_cycle(nxg):
    degs = [d for _, d in nxg.degree]
    if min(degs) == 2 and max(degs) == 2:
        return len(nxg.edges)
    return None


class GraphWrapper:
    def __init__(self, nx_graph):
        self.id = uuid_gen().int

        self.graph = nx_graph

        self.hombase = None
        self.td = None
        self.plan = None

        self.star = _is_nx_star(nx_graph)
        self.cycle = _is_cycle(nx_graph)

        # Nauty graphs don't work with serialisation so make sure not to store them
        self.nauty_graph = None

        self.biclique = None
        for n, m in [(2, 2), (2, 3), (3, 3)]:
            if _is_Knm(self, n, m):
                self.biclique = (n, m)

        self.clique = None
        n = len(self.V)
        if (n * (n - 1)) / 2 == len(self.E):
            self.clique = n

    def from_g6str(g6str, sparse=False, directed=False):
        assert (not (directed and sparse))
        if directed:
            rawdata = map(int, g6str.split())
            data = list(grouper(rawdata, 2, fillvalue='strict'))
            # nv, ne = data[0]
            edges = data[1:]
            graph = nx.from_edgelist(edges, create_using=nx.DiGraph)
        elif sparse:
            graph = nx.from_sparse6_bytes(str.encode(g6str))
        else:
            graph = nx.from_graph6_bytes(str.encode(g6str))

        return GraphWrapper(graph)

    def from_nauty(nauty_graph):
        nxg = nx.from_dict_of_lists(nauty_graph.adjacency_dict)
        return GraphWrapper(nxg)

    def guarantee_nauty_graph(self):
        if self.nauty_graph is not None:
            return True
        self.nauty_graph = nautyhelper.nx_to_pynauty(self.graph)
        return True

    def cleanup_for_storage(self):
        self.nauty_graph = None

    def vertex_labels_dict(self):
        ret = dict()
        for n in self.graph:
            nodeinfo = self.graph.nodes[n]
            ret[n] = nodeinfo.get('labels', [])
        return ret

    @property
    def is_directed(self):
        return nx.is_directed(self.graph)

    @property
    def V(self):
        return list(self.graph.nodes)

    @property
    def E(self):
        return (self.graph.edges)

    @property
    def hombase_known(self):
        return not (self.hombase is None)

    @property
    def ghw(self):
        return self.td.ghw if self.td is not None else None

    @property
    def tw(self):
        return self.td.tw if self.td is not None else None
