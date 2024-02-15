import warnings
from subprocess import Popen, PIPE
import networkx as nx
import json
import multiprocess

from pact.hypergraph import HyperGraph
import pact.jointree as jt
import pact.treedecomp as td


BALGO_PATH = 'lib/BalancedGo/BalancedGo'


def _make_edge_conversion_map(G):
    Gedges = list(G.E)
    edge_conversion = dict()

    for i, e in enumerate(Gedges):
        name = f'E_{i}'
        edge_conversion[name] = e
    return edge_conversion


def _rawHTD(G, ecmap):
    """Gets a raw hypertree decomp from BalancedGo."""
    balgo = Popen([BALGO_PATH, '-shellio',
                   '-exact',  # '-width', '2',
                   '-heuristic', '1',
                   '-local', '-complete'],
                  shell=False, stdin=PIPE, stdout=PIPE)

    # write hg on stdin
    for ename, e in ecmap.items():
        l = str.encode(f'{ename}({e[0]}, {e[1]})\n')
        balgo.stdin.write(l)
    balgo.stdin.close()

    # get output
    # TODO: add error handling
    htd = json.loads(balgo.stdout.readline())

    decomp = td.td_from_BalGo_json(htd, ecmap)
    for node in decomp.nodes():
        node.bag = set(map(int, node.bag))
        node.cover_map = {en: ecmap[en] for en in node.cover}

    return decomp


def _cover_is_connected(G, node):
    subg_edges = list(node.cover_map.values())
    cover_subg = nx.edge_subgraph(G.graph, subg_edges)
    if nx.is_directed(G.graph):
        return nx.is_weakly_connected(cover_subg)
    else:
        return nx.is_connected(cover_subg)


def _find_shortest_path_between_edges(G, e1, e2):
    """Returns a shortest path (as a list of vertices) that connects the two input edges"""
    a, b = e1
    c, d = e2

    if nx.is_directed(G.graph):
        H = G.graph.to_undirected(reciprocal=False, as_view=True)
    else:
        H = G.graph
    paths = [nx.shortest_path(H, a, c),
             nx.shortest_path(H, b, c),
             nx.shortest_path(H, a, d),
             nx.shortest_path(H, b, d)]

    shortest = None
    for p in paths:
        if shortest is None:
            shortest = p
            continue
        if len(p) < len(shortest):
            shortest = p
    return shortest


def _connect_cover(G, node, ecmap):
    """If cover is not connected in graph then return an improved cover that is connected."""
    if _cover_is_connected(G, node):
        return node.cover
    if len(node.cover) > 2:
        warnings.warn('Connected covers for high widths are not great', RuntimeWarning)
    # Note that for the directed case we care only about weak connectedness
    #print(cover, 'is not connected!')

    cover_edges = list(node.cover_map.values())
    edge_pairs = zip(cover_edges, cover_edges[1:])
    path = []
    for e1, e2 in edge_pairs:
        # should be shortst path from current path in better version
        local_path = _find_shortest_path_between_edges(G, e1, e2)
        #print(f'local path beween {e1} and {e2}', local_path)

        for a, b in zip(local_path, local_path[1:]):
            ab_name = [k for k, v in ecmap.items() if v == (a, b) or v == (b, a)][0]
            #print(a,b, f'is {ab_name}')
            path.append(ab_name)

    #print('gives', path)
    # get rid of duplicates and add base cover
    path = list(set(node.cover) | set(path))
    #print('final', path)
    # this is a mess
    return path


def _G_to_HG(ecmap):
    """Converts an edge conversion map to a HyperGraph object"""
    hg = HyperGraph()
    for en, e in ecmap.items():
        hg.add_edge(set(e), en)
    return hg


def _get_refined_decomp(G, ecmap, refine_covers):
    """Get a HTD plus refinements for graph G."""
    htd = _rawHTD(G, ecmap)

    overhead = 0
    for n in htd.nodes():
        if refine_covers:
            n.con_cover = _connect_cover(G, n, ecmap)
        else:
            n.con_cover = list(n.cover)

        n.con_cover_map = {en: ecmap[en] for en in n.con_cover}

        delta = len(n.con_cover) - len(n.cover)
        overhead += delta**2

    return htd, overhead


def balgo_multitry_for_cheapest_decomp(G, times=1, threads=1, refine_covers=True):
    ecmap = _make_edge_conversion_map(G)

    hg = _G_to_HG(ecmap)
    try:
        # needs to add bags to jt
        decomp = jt.get_jt(hg, ecmap)
        return decomp, 0
    except RuntimeError:
        pass

    # Avoid the use of multiprocess in the future for threads=1 and times=1 cases
    pool = multiprocess.Pool(threads)
    params = zip([G] * times, [ecmap] * times, [refine_covers] * times)
    x = pool.starmap(_get_refined_decomp, params)

    best, bestcost = min(x, key=lambda pair: pair[1])

    return best, bestcost
