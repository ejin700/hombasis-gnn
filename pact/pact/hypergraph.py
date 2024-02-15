"""Reused version of a hypergraph helper library for use with the
join tree algorithm. 

 We use barely any of these features and we
could further simplify to a graph library or merge with the Graph type
used as a wrapper for networkx objects in the future.

"""

import networkx as nx
import re
import itertools


class HyperGraph(object):
    def __init__(self):
        self.V = set()
        self.E = list()
        self.edge_dict = dict()

    def grid(n, m):
        h = HyperGraph()
        hc, vc = 0, 0
        for col in range(m - 1):
            for row in range(n):
                vi = '{}.{}'.format(row, col)
                vright = '{}.{}'.format(row, col + 1)
                horz_name = 'H{}'.format(hc)
                hc = hc + 1
                h.add_edge(set([vi, vright]),
                           horz_name)
        for col in range(m):
            for row in range(n - 1):
                vi = '{}.{}'.format(row, col)
                vdown = '{}.{}'.format(row + 1, col)
                vert_name = 'V{}'.format(vc)
                vc = vc + 1
                h.add_edge(set([vi, vdown]),
                           vert_name)
        return h

    def copy(self):
        h = HyperGraph()
        for en, e in self.edge_dict.items():
            h.add_edge(e.copy(), name=en)
        return h

    def join_copy(self, x, y):
        """Copy of self with vertices x and y joined"""
        if x not in self.V or y not in self.V:
            raise ValueError('Join vertices need to be in hypergraph')
        h = HyperGraph()
        for en, e in self.edge_dict.items():
            e2 = e.copy()
            if y in e2:
                e2.remove(y)
                e2.add(x)
            h.add_edge(e2, name=en)
        return h

    def toHyperbench(self):
        s = []
        for en, e in sorted(self.edge_dict.items()):
            s.append('{}({}),'.format(en, ','.join(e)))
        return '\n'.join(s)

    def vertex_induced_subg(self, U):
        """Induced by vertex set U"""
        h = HyperGraph()
        for en, e in self.edge_dict.items():
            e2 = e.copy()
            e2 = e2 & U
            if e2 != set():
                h.add_edge(e2, name=en)
        return h

    def bridge_subg(self, U):
        EC = [en for en, e in self.edge_dict.items() if
              (e & U) != set()]
        C = self.edge_subg(EC)

        # for each component C_i of rest, compute a special edge Sp_i
        for C_i in self.separate(U):
            print(C_i)
            Sp_i_parts = [(e - U) for e in C.E if (e & C_i.V) != set()]
            Sp_i = set.union(*Sp_i_parts)
            C.add_special_edge(Sp_i)
        return C

    def edge_subg(self, edge_names):
        h = HyperGraph()
        for en in edge_names:
            if en not in self.edge_dict:
                raise ValueError('Edge >{}< not present in hypergraph'.format(en))
            h.add_edge(self.edge_dict[en].copy(), en)
        return h

    def fromHyperbench(fname):
        EDGE_RE = re.compile('\s*([\w:]+)\s?\(([^\)]*)\)')

        def split_to_edge_statements(s):
            x = re.compile('\w+\s*\([^\)]+\)')
            return list(x.findall(s))

        def cleanup_lines(rl):
            a = map(str.rstrip, rl)
            b = filter(lambda x: not x.startswith('%') and len(x) > 0, a)
            return split_to_edge_statements(''.join(b))

        def line_to_edge(line):
            m = EDGE_RE.match(line)
            name = m.group(1)
            e = m.group(2).split(',')
            e = set(map(str.strip, e))
            return name, e

        with open(fname) as f:
            raw_lines = f.readlines()
        lines = cleanup_lines(raw_lines)

        hg = HyperGraph()
        for line in lines:
            edge_name, edge = line_to_edge(line)
            hg.add_edge(edge, edge_name)
        return hg

    def add_edge(self, edge, name):
        assert (type(edge) == set)
        self.edge_dict[name] = edge
        self.V.update(edge)
        self.E.append(edge)

    def add_special_edge(self, sp):
        SPECIAL_NAME = 'Special'
        # find a name first
        sp_name = None
        for i in itertools.count():
            candidate = SPECIAL_NAME + str(i)
            if candidate not in self.edge_dict:
                sp_name = candidate
                break
        self.add_edge(sp, sp_name)

    def remove_edge(self, name):
        e = self.edge_dict[name]
        del self.edge_dict[name]
        self.E.remove(e)

    def degrees(self):
        deg_dict = {v: 0 for v in self.V}
        for e in self.E:
            for v in e:
                deg_dict[v] += 1
        return deg_dict

    def primal_nx(self):
        G = nx.Graph()
        G.add_nodes_from(self.V)
        for i, e in enumerate(self.E):
            for a, b in itertools.combinations(e, 2):
                G.add_edge(a, b)
        return G

    def incidence_nx(self, without=[]):
        G = nx.Graph()
        G.add_nodes_from(self.V)
        G.add_nodes_from(self.edge_dict.keys())
        for n, e in self.edge_dict.items():
            if n in without:
                continue
            for v in e:
                G.add_edge(n, v)
        return G

    def toPACE(self, special=[]):
        buf = list()
        vertex2int = {v: str(i) for i, v in enumerate(self.V, start=1)}
        buf.append('p htd {} {}'.format(len(self.V),
                                        len(self.E)))
        for i, ei in enumerate(sorted(self.edge_dict.items()), start=1):
            en, e = ei
            edgestr = ' '.join(map(lambda v: vertex2int[v], e))
            line = '{} {}'.format(i, edgestr)
            buf.append(line)

        if special is None:
            special = []
        for sp in special:
            if sp is None:
                continue
            edgestr = ' '.join(map(lambda v: vertex2int[v], sp))
            buf.append('s ' + edgestr)
        return '\n'.join(buf)

    def separation_subg(self, U, sep):
        C = HyperGraph()
        cover = U | sep
        for en, e in self.edge_dict.items():
            if e.issubset(cover) and not e.issubset(sep):
                C.add_edge(e, en)
        return C

    def separate(self, sep):
        """Returns list of components"""
        assert (type(sep) == set)
        primal = self.primal_nx()
        primal.remove_nodes_from(sep)
        comp_vertices = nx.connected_components(primal)
        comps = [self.separation_subg(U, sep)
                 for U in comp_vertices]
        return comps

    # def fancy_repr(self, hl=[]):
    #     edge_style = colorama.Fore.RED + colorama.Style.NORMAL
    #     vertex_style = colorama.Fore.YELLOW + colorama.Style.NORMAL
    #     hl_style = colorama.Fore.WHITE + colorama.Back.GREEN + colorama.Style.BRIGHT
    #     _reset = colorama.Style.RESET_ALL

    #     def color_vertex(v):
    #         if v in hl:
    #             return hl_style + v + _reset
    #         else:
    #             return vertex_style + v + _reset
    #     s = ''
    #     for en, e in sorted(self.edge_dict.items()):
    #         s += edge_style + en + _reset + '('
    #         s += ','.join(map(color_vertex, e))
    #         s += ')\n'
    #     return s

    # def __repr__(self):
    #     return self.fancy_repr()
