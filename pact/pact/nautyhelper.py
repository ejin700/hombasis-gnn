import pynauty
import networkx as nx


def nx_to_pynauty(nxg):
    nv = len(nxg.nodes())
    adj = {v: list(adjd.keys()) for v, adjd in nxg.adjacency()}
    pyn = pynauty.Graph(nv,
                        directed=nx.is_directed(nxg),
                        adjacency_dict=adj)
    return pyn


def nauty_graph_edgenum(ng):
    acc = 0
    for _, v in ng.adjacency_dict.items():
        acc += len(v)
    if not ng.directed:
        acc //= 2
    return acc


def nauty_quotient_fail_on_selfloop(ng, rho):
    adjd = {k: set(v) for k, v in ng.adjacency_dict.items()}

    newdict = dict()

    for i, Bi in enumerate(rho):
        newdict[i] = []
        for j, Bj in enumerate(rho):
            for vi in Bi:
                if set(Bj).intersection(adjd[vi]) != set():
                    if i == j:
                        return None
                    newdict[i].append(j)
                    break

    adjl = {k: list(v) for k, v in newdict.items()}
    # number of vertices is suboptimal but exact would require relabling
    return pynauty.Graph(len(rho),
                         directed=ng.directed, adjacency_dict=adjl)


def nauty_has_bidirected_edge(ng):
    adjlist = {k: set(v) for k, v in ng.adjacency_dict.items()}
    for k, kadj in adjlist.items():
        for j in kadj:
            if k in adjlist[j]:
                return True
    return False
