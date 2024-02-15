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


def label_semijoin_op(edgename, vertex, label_info):
    """
    Filters edgename according to labels
    """
    ops = []
    for label in label_info[vertex]:
        label_base_rel = Operation.LABELREL_PREFIX + label
        label_renamed_rel = f'label_{label}${vertex}'
        ren = Operation(Operation.RENAME, label_renamed_rel,
                        A=label_base_rel,
                        rename_key={'vertex': vertex})
        sj = Operation(Operation.SEMIJOIN, edgename,
                       A=edgename, B=label_renamed_rel,
                       key=[vertex])
        ops += [ren, sj]
    return ops


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


def plan_for_pattern(wrapped_graph):
    label_info = wrapped_graph.vertex_labels_dict()
    return _node_to_ops(wrapped_graph.td, 0, label_info)

   
def _node_to_ops(node, index, label_info):
    def node_name_from_index(index):
        return f'node${index}'
    plan = deque()
    nodename = node_name_from_index(index)

    for en, e in node.con_cover_map.items():
        plan.append(rename_op(en, e))
        for v in e:
            plan.extend(label_semijoin_op(en, v, label_info))

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
        plan.extend(_node_to_ops(child, index, label_info))
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


