"""
A SpasmSpace assumed that no two graphs in the space are isomorphic.
This is never checked for cost reasons but assumed as an invariant throughout.
"""
from pact.graphwrapper import GraphWrapper


class SpasmSpace:
    def __init__(self):
        self._graphs = dict()
        self._ev_index = dict()

    def add_from_g6lines(self, lines, wrapper_params):
        for line in lines:
            g = GraphWrapper.from_g6str(line, **wrapper_params)
            self.add_wrapped_graph(g)

    def add_wrapped_graph(self, G):
        self._graphs[G.id] = G

        idx_key = (len(G.E), len(G.V))
        if idx_key not in self._ev_index:
            self._ev_index[idx_key] = list()
        self._ev_index[idx_key].append(G.id)
        return G.id

    def add_nx_graph(self, nxg):
        G = GraphWrapper(nxg)
        self.add_wrapped_graph(G)

    def iter_by_properties(self, properties):
        for G in self._graphs.values():
            if any(getattr(G, propname, None) != propval
                   for propname, propval in properties.items()):
                continue
            else:
                yield G

    def find(self, properties):
        return next(self.iter_by_properties(properties), None)

    def items(self):
        return self._graphs.items()

    def graphs_iter(self):
        return self._graphs.values()

    def __getitem__(self, gid):
        return self._graphs[gid]

    def __len__(self):
        return len(self._graphs)

    def iter_by_ev(self, num_edges, num_vertices):
        idx_key = (num_edges, num_vertices)
        idx_entry = self._ev_index.get(idx_key, [])
        for gid in idx_entry:
            yield self._graphs[gid]

    def cleanup_for_storage(self):
        for G in self._graphs.values():
            G.cleanup_for_storage()
