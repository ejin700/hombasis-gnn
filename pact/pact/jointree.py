"""
Computes join trees from HyperGraphs
"""
import pact.treedecomp as td


def tree_from_rels(rels, ecmap):
    nodes = dict()
    for ch, p in rels:
        chnode = nodes.get(ch,
                           td.TDNode(ecmap[ch], cover_map={ch: ecmap[ch]}))
        pnode = nodes.get(p,
                          td.TDNode(ecmap[p], cover_map={p: ecmap[p]}))

        pnode.children.append(chnode)
        nodes[ch] = chnode
        nodes[p] = pnode

    # Set all con cover maps, singleton covers are trivially connected
    for x in nodes.values():
        x.set_con_cover_map(x.cover_map)

    # Find root
    as_child = {k: 0 for k in nodes.keys()}
    for ch, _ in rels:
        as_child[ch] += 1

    rootname = min(as_child.items(), key=lambda p: p[1])[0]
    root = nodes[rootname]
    return root


def boring_vertices(hg):
    degs = hg.degrees()
    isolated = [v for v, k in degs.items() if k <= 1]
    return isolated


def build_subedges_list(hg):
    subset_rels = list()
    for en, e in hg.edge_dict.items():
        for fn, f in hg.edge_dict.items():
            if fn == en:
                continue
            if e.issubset(f):
                subset_rels.append((en, fn))

    return list(sorted(subset_rels))


def greedy_shallower_jt(orig_hg):
    if len(orig_hg.E) == 0:
        return []
    if len(orig_hg.E) == 1:
        en = list(orig_hg.edge_dict.keys())[0]
        return []

    to_del = set(boring_vertices(orig_hg))
    leftover = orig_hg.V - to_del
    hg = orig_hg.vertex_induced_subg(leftover)

    subedges = build_subedges_list(hg)
    removed_edges = set()
    jt = list()

    for en, fn in subedges:
        if en in removed_edges or fn in removed_edges:
            continue

        hg.remove_edge(en)
        jt.append((en, fn))
        removed_edges.add(en)

        "prioritise subedges of fn at this point"
        for gn, fn2 in subedges:
            if gn in removed_edges or fn2 != fn:
                continue

            hg.remove_edge(gn)
            jt.append((gn, fn))
            removed_edges.add(gn)

    if len(to_del) == 0 and len(removed_edges) == 0:
        """cycle"""
        raise RuntimeError('GYO non-empty fixpoint reached -> cyclic hypergraph')
    chjt = greedy_shallower_jt(hg)
    jt += chjt
    return jt


def _singleton_td(hg, ecmap):
    en = list(hg.edge_dict.keys())[0]
    cover_map = {en: ecmap[en]}
    decomp = td.TDNode(ecmap[en], cover_map=cover_map)
    decomp.set_con_cover_map(cover_map)
    return decomp


def get_jt(hg, ecmap):
    if len(hg.E) == 1:
        return _singleton_td(hg, ecmap)
    jtedges = greedy_shallower_jt(hg)
    return tree_from_rels(jtedges, ecmap)
