"""
Creates the execution plan from a given Tree Decomposition.
TODO: needs some refactoring to make the interface clear (only node_to_ops should really be called externally)
TODO: possible further refactoring to allow for optimised planners / alternative plannings
"""
from collections import deque
from pact.operation import Operation
from pact.treedecomp import TDNode


def find_join_path(con_cover_map):
    path = []
    cur_vars = set()
    working_map = dict(con_cover_map)

    start = list(working_map.keys())[0]
    path.append(start)
    cur_vars = set(working_map[start])
    del working_map[start]

    while len(working_map) > 0:
        next_en, next_e = None, None
        for en, e in working_map.items():
            if cur_vars.intersection(set(e)) != set():
                next_en, next_e = en, e
                break
        # for safety, actually impossible if cover is connected
        assert (next_en is not None and next_e is not None)
        path.append(next_en)
        cur_vars = cur_vars.union(set(next_e))
        del working_map[en]
    return path


def binary_join_op(Rn, Sn, cover_map, nodename):
    R, S = cover_map[Rn], cover_map[Sn]
    joinatts = set.intersection(set(R), set(S))
    o = Operation(Operation.JOIN, nodename,
                  A=Rn, B=Sn, key=joinatts)
    return o


# TODO refactor this and binary join op to con_cover_map for clarity
def path_join_ops(path, tdnode, nodename):
    cover_map = tdnode.con_cover_map

    # first join the first two parts of the path
    start = binary_join_op(path[0], path[1], cover_map, nodename)
    all_joins = [start]
    cur_vars = set.union(set(cover_map[path[0]]), set(cover_map[path[1]]))

    # then continue joining in the path edges
    for en in path[2:]:
        e = set(cover_map[en])
        joinatts = cur_vars.intersection(e)
        cur_vars = cur_vars.union(e)

        en_join = Operation(Operation.JOIN, nodename,
                            A=nodename, B=en, key=joinatts)
        all_joins.append(en_join)

    # finally project only to bag to make counting work cleanly
    project = Operation(Operation.PROJECT, nodename,
                        A=nodename, key=tdnode.bag)
    all_joins.append(project)
    return all_joins


def cover_join_ops(tdnode, nodename):
    con_cover_map = tdnode.con_cover_map
    if len(con_cover_map) == 2:
        k1, k2 = con_cover_map.keys()

        # TODO: fix this on a depper level.
        # The issue is that some decompositions have a 2 edge cover despite the bag being covered fully by one edge
        # It's not trivial to fix by ignoring the redundandt edge as we have to make sure that it
        # doesn't get lost overall, i.e., that the decomp remains complete.
        # For now this fixes correctness, it's very inefficient to compute these useless joins though.
        # THe issue is quite rare though.
        paranoid_project = Operation(Operation.PROJECT, nodename,
                                     A=nodename, key=tdnode.bag)
        return [binary_join_op(k1, k2, con_cover_map, nodename),
                paranoid_project]
    else:
        path = find_join_path(con_cover_map)  # really needs to be connected for this to work
        return path_join_ops(path, tdnode, nodename)


def rename_op(edgename, edge):
    """
    Note for the directed case that we globally assume that edge
    tuples everywhere are of the form (source, target).
    """
    o = Operation(Operation.RENAME, edgename,
                  A=Operation.BASERELNAME,
                  rename_key={'s': edge[0], 't': edge[1]})
    return o


def semijoin_op(parent, child, parentname, childname):
    joinatts = set.intersection(parent.bag, child.bag)
    o = Operation(Operation.SEMIJOIN, parentname,
                  A=parentname, B=childname, key=joinatts)
    return o


def is_semijoin_child(p, c):
    """Given nodes p and c in a TD, decide whether c should just semijoin into p"""
    return c.bag.issubset(p.bag) and c.isLeaf


def count_ops(parent, child, parentname, childname):
    overlap = set.intersection(parent.bag, child.bag)
    childcount = Operation(Operation.COUNT_EXT, childname,
                           A=childname, key=overlap)
    merge_count = Operation(Operation.SUM_COUNT, parentname,
                            A=parentname, B=childname, key=overlap)
    return [childcount, merge_count]


def node_to_ops(node, index=0):
    def node_name_from_index(index):
        return f'node${index}'
    plan = deque()
    nodename = node_name_from_index(index)
    for en, e in node.con_cover_map.items():
        plan.append(rename_op(en, e))

    # compute the node join
    # todo deal with name of resulting relation
    if len(node.con_cover) > 1:
        plan.extend(cover_join_ops(node, nodename))
    else:
        # make the renamed node the noderel directly in the renaming
        # be careful this is fragile to changes in plan building
        plan[-1].new_name = nodename

    # attach child plans
    child_map = dict()
    for child in node.children:
        index += 1
        plan.extend(node_to_ops(child, index))
        child_map[child] = node_name_from_index(index)

    # first do semijoin children
    # loop to small to be worth the complexity of doing all in one loop
    for child in node.children:
        if not is_semijoin_child(node, child):
            continue
        childname = child_map[child]
        o = semijoin_op(node, child, nodename, childname)
        plan.append(o)

    # do the exciting counting ops for other children
    for child in node.children:
        if is_semijoin_child(node, child):
            continue
        childname = child_map[child]
        count_plan = count_ops(node, child, nodename, childname)
        plan.extend(count_plan)

    return plan


"""
Early Sj magic
"""


def _find_en(e, G):
    for n in G.td.nodes():
        for k, v in n.con_cover_map.items():
            if set(v) == e:
                return k
    raise RuntimeError(f'Name for edge {e} not found')


def _make_unary_tdn(e, G):
    en = _find_en(e, G)
    cover_map = {en: tuple(e)}
    n = TDNode(e, cover_map)
    n.set_con_cover_map(cover_map)
    return n


def _has_sj_child(tdnode, e):
    """Does tdnode have a child that is just the edge e"""
    for c in tdnode.children:
        if len(c.cover) > 1 or not is_semijoin_child(tdnode, c):
            continue
        if c.bag == e:
            return True
    return False


def opportunistic_traingle_sj_add(G, thres=2, max_add=3):
    add = dict()
    for n in G.td.nodes():
        add[n] = []
        if len(n.con_cover) >= thres:

            cover_edges = list(map(set, n.con_cover_map.values()))
            cover_vars = set.union(*cover_edges)
            # print(cover_edges)
            # print(cover_vars)
            for e in map(set, G.E):
                if e not in cover_edges and e.issubset(cover_vars) and not _has_sj_child(n, e):
                    # print(e, 'SJ possible')
                    sjchild = _make_unary_tdn(e, G)
                    # print(n)
                    add[n].append(sjchild)
    for node, newsjs in add.items():
        node.children.extend(newsjs)


# TODO refactor this and binary join op to con_cover_map for clarity
def path_join_ops_earlysj(path, tdnode, nodename, child_map):
    cover_map = tdnode.con_cover_map

    # first join the first two parts of the path
    start = binary_join_op(path[0], path[1], cover_map, nodename)
    all_joins = [start]
    cur_vars = set.union(set(cover_map[path[0]]), set(cover_map[path[1]]))

    # prepare children to semijoin as soon as possible
    sjcands = [c for c in tdnode.children if is_semijoin_child(tdnode, c)]
    handled_sj = [c for c in sjcands if c.bag.issubset(cur_vars)]

    for c in handled_sj:
        childname = child_map[c]
        sj = Operation(Operation.SEMIJOIN, nodename,
                       A=nodename, B=childname, key=c.bag)
        all_joins.append(sj)
        sjcands.remove(c)
    del handled_sj

    # then continue joining in the path edges
    for en in path[2:]:
        e = set(cover_map[en])
        joinatts = cur_vars.intersection(e)
        cur_vars = cur_vars.union(e)

        en_join = Operation(Operation.JOIN, nodename,
                            A=nodename, B=en, key=joinatts)
        all_joins.append(en_join)

        # semijoin in any new possible rels
        handled_sj = [c for c in sjcands if c.bag.issubset(cur_vars)]

        for c in handled_sj:
            childname = child_map[c]
            sj = Operation(Operation.SEMIJOIN, nodename,
                           A=nodename, B=childname, key=c.bag)
            all_joins.append(sj)
            sjcands.remove(c)

    # finally project only to bag to make counting work cleanly
    project = Operation(Operation.PROJECT, nodename,
                        A=nodename, key=tdnode.bag)
    all_joins.append(project)

    assert (len(sjcands) == 0)
    return all_joins


def cover_join_ops_earlysj(tdnode, nodename, child_map):
    con_cover_map = tdnode.con_cover_map
    if len(con_cover_map) == 2:
        k1, k2 = con_cover_map.keys()

        # TODO: fix this on a depper level.
        # The issue is that some decompositions have a 2 edge cover despite the bag being covered fully by one edge
        # It's not trivial to fix by ignoring the redundandt edge as we have to make sure that it
        # doesn't get lost overall, i.e., that the decomp remains complete.
        # For now this fixes correctness, it's very inefficient to compute these useless joins though.
        # THe issue is quite rare though.

        # TODO FIX
        sjcands = [c for c in tdnode.children if is_semijoin_child(tdnode, c)]
        sjs = [Operation(Operation.SEMIJOIN, nodename, A=nodename, B=child_map[c], key=c.bag)
               for c in sjcands]

        e1, e2 = tdnode.cover_map[k1], tdnode.cover_map[k2]
        cols_after_join = set(e1) | set(e2)
        join_ops = [binary_join_op(k1, k2, con_cover_map, nodename)] + sjs
        if cols_after_join == tdnode.bag:
            return join_ops
        paranoid_project = Operation(Operation.PROJECT, nodename,
                                     A=nodename, key=tdnode.bag)
        return join_ops + [paranoid_project]
    elif len(tdnode.cover) == 2 and len(tdnode.con_cover) > 4:
        k1, k2 = tdnode.cover
        e1, e2 = tdnode.cover_map[k1], tdnode.cover_map[k2]
        cols_after_join = set(e1) | set(e2)

        op = Operation(Operation.JOIN, nodename,
                       A=k1, B=k2, key=[])

        sjcands = [c for c in tdnode.children if is_semijoin_child(tdnode, c)]
        sjs = [Operation(Operation.SEMIJOIN, nodename, A=nodename, B=child_map[c], key=c.bag)
               for c in sjcands]

        join_ops = [op] + sjs
        if cols_after_join == tdnode.bag:
            return join_ops

        paranoid_project = Operation(Operation.PROJECT, nodename,
                                     A=nodename, key=tdnode.bag)
        return join_ops + [paranoid_project]
    else:
        path = find_join_path(con_cover_map)  # really needs to be connected for this to work
        return path_join_ops_earlysj(path, tdnode, nodename, child_map)


def node_to_ops_earlysj(node, index=0):
    def node_name_from_index(index):
        return f'node${index}'
    plan = deque()

    nodename = node_name_from_index(index)
    for en, e in node.con_cover_map.items():
        plan.append(rename_op(en, e))
        if len(node.con_cover) == 1:
            # make the renamed node the noderel directly in the renaming
            # be careful this is fragile to changes in plan building
            plan[-1].new_name = nodename

    # attach child plans
    child_map = dict()
    for child in node.children:
        index += 1
        plan.extend(node_to_ops_earlysj(child, index))
        child_map[child] = node_name_from_index(index)

    # compute the node join
    # todo deal with name of resulting relation
    if len(node.con_cover) > 1:
        plan.extend(cover_join_ops_earlysj(node, nodename, child_map))

    # do the exciting counting ops for other children
    for child in node.children:
        if is_semijoin_child(node, child):
            continue
        childname = child_map[child]
        count_plan = count_ops(node, child, nodename, childname)
        plan.extend(count_plan)

    return plan
