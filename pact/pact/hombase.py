"""
Functions for computing the homomorphism base
"""
import networkx as nx
# we use igraph for computing the automorphism number, kinda ugly but fast
import igraph as ig
import math
import pact.util as util
import pynauty
from pact.nautyhelper import nx_to_pynauty, nauty_has_bidirected_edge, \
    nauty_quotient_fail_on_selfloop, nauty_graph_edgenum
from pact.graphwrapper import GraphWrapper
from gmpy2 import mpq


def num_automorphisms(G):
    igG = ig.Graph.from_networkx(G.graph)
    return ig.automorphisms._count_automorphisms_vf2(igG)


def partition_product(rho):
    return util.prod(map(lambda B: math.factorial(len(B) - 1), rho))


def calc_coeff(G, F, partition_base, automorphisms):
    vdiff = len(G.V) - len(F.V)
    sign = (-1)**vdiff
    return mpq(sign * partition_base) / automorphisms


def loop_in_partition(E, rho):
    # kick out loop stuff
    for e in E:
        for B in rho:
            if e.issubset(set(B)):
                return True
    return False


def hombase_coeffs(G, spasm_space, skip_bidirected=True):
    # Maintains the sum term over all partition products for
    # partitions that are isomorphic to graph in key of dictionary
    partition_base = dict()
    E = list(map(set, G.E))

    for rho in util.partitions_mit(G.V):
        # We assume no loops in the host graph and therefore only
        # loop-free graphs are of interest in the spasm.
        if loop_in_partition(E, rho):
            continue

        quot = nx.quotient_graph(G.graph, rho,
                                 create_using=G.graph.__class__)

        # If skip_bidirected is set true, we assume that all directed
        # hosts have no bidirected edges. In that case we can skip all
        # patterns in the spasm with bidirected edges as the
        # homomorphism count will necessarily be 0.
        if (skip_bidirected and nx.is_directed(quot) and
            any((True for (u, v) in quot.edges() if u in quot[v]))):
            continue

        quot_edges = len(quot.edges)
        quot_vertices = len(rho)

        found = False
        for F in spasm_space.iter_by_ev(quot_edges, quot_vertices):
            # Seems to be worth it to pre-filter by the quicker check
            if not nx.faster_could_be_isomorphic(quot, F.graph):
                continue

            # If we actually found a graph in the base: compute the
            # product term for the partition and add it to the
            # 'partition base' for F
            if nx.is_isomorphic(quot, F.graph):
                cur = partition_base.get(F.id, 0)
                partition_base[F.id] = cur + partition_product(rho)
                found = True
                # We can stop here as we assume that the spasm space
                # contains no duplicates (under isomorphism). Hence,
                # at most one graph in the spasm space can be
                # ismorphic to `quot`
                break

        if not found:
            raise RuntimeError(f'Graph {G.id} with partition {rho} is not in given spasm space')

    # after we created all sum terms from partitions, compute final coefficient
    autos = num_automorphisms(G)
    return {graphid: calc_coeff(G, spasm_space[graphid], part_base, autos)
            for graphid, part_base in partition_base.items()}


def hombase_coeffs_nauty(G, spasm_space,
                         skip_bidirected=True,
                         expand_space=False):
    # Maintains the sum term over all partition products for
    # partitions that are isomorphic to graph in key of dictionary
    partition_base = dict()
    E = list(map(set, G.E))

    Gnauty = nx_to_pynauty(G.graph)

    for rho in util.partitions_mit(G.V):
        # On the graphs I tested this early detection is faster than
        # computing the quotient and checking then
        if loop_in_partition(E, rho):
            continue

        quot = nauty_quotient_fail_on_selfloop(Gnauty, rho)
        if quot is None:
            raise RuntimeError('Already checked, shouldnt happen')
        if skip_bidirected and nx.is_directed(G.graph) and nauty_has_bidirected_edge(quot):
            continue

        quot_edges = nauty_graph_edgenum(quot)
        quot_vertices = len(rho)

        found = False
        for F in spasm_space.iter_by_ev(quot_edges, quot_vertices):
            F.guarantee_nauty_graph()
            if pynauty.isomorphic(quot, F.nauty_graph):
                cur = partition_base.get(F.id, 0)
                partition_base[F.id] = cur + partition_product(rho)
                found = True
                break

        if not found:
            if not expand_space:
                raise RuntimeError(
                    f'Graph {G.id} with partition {rho} is not in given spasm space')
            else:
                new_quot = GraphWrapper.from_nauty(quot)
                new_quot_id = spasm_space.add_wrapped_graph(new_quot)
                partition_base[new_quot_id] = partition_product(rho)

    # after we created all sum terms from partitions, compute final coefficient
    autos = int(pynauty.autgrp(Gnauty)[1])
    return {graphid: calc_coeff(G, spasm_space[graphid], part_base, autos)
            for graphid, part_base in partition_base.items()}
