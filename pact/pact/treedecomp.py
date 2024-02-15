from collections import deque


class TDNode:
    def __init__(self, bag, cover_map=[]):
        self.bag = set(bag)
        self.children = []
        self.cover_map = cover_map
        self.cover = list(cover_map.keys())
        self.con_cover_map = None
        self.con_cover = None
        # self.parent = None

    def set_con_cover_map(self, con_cover_map):
        self.con_cover_map = con_cover_map
        self.con_cover = list(con_cover_map.keys())

    def nodes(self):
        return td_nodes_it(self)

    @property
    def isLeaf(self):
        return len(self.children) == 0

    @property
    def ghw(self):
        return max([len(self.cover)] + [c.ghw for c in self.children])

    @property
    def tw(self):
        return max([len(self.bag) - 1] + [c.tw for c in self.children])

    @property
    def depth(self):
        return 1 + max([0] + [c.depth for c in self.children])

    def _to_str(self, depth=0):
        space = '  ' * depth
        return space + f'B: {str(self.bag)}\tλ: {self.cover}'

    def _to_repr_lines(self, depth=0):
        lines = [self._to_str(depth)]
        for c in self.children:
            lines.extend(c._to_repr_lines(depth + 1))
        return lines

    def __repr__(self):
        lines = self._to_repr_lines(0)
        return '\n'.join(lines)


def td_nodes_it(root):
    frontier = deque([root])
    while len(frontier) > 0:
        n = frontier.popleft()
        yield n
        if n.children is not None:
            frontier.extendleft(n.children)


def bfs_find_exact_bag(tree, bag):
    for node in tree.nodes():
        if node.bag == bag:
            return node
    return None


def td_from_BalGo_json(jsonhtd, ecmap):
    def json_node_to_tdn(jsonnode):
        cover_map = {en: ecmap[en] for en in jsonnode['Cover']}
        n = TDNode(jsonnode['Bag'], cover_map)
        if jsonnode['Children'] is None:
            return n
        n.children = list(map(json_node_to_tdn, jsonnode['Children']))
        return n

    jsonroot = jsonhtd['Root']
    root = json_node_to_tdn(jsonroot)
    return root
