import doctest
from unionfind import UnionFind
from collections import deque
from heapdict import heapdict


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


class Graph:

    @classmethod
    def new(cls, weighted = False, directed = False):
        assert isinstance(weighted, bool)
        assert isinstance(directed, bool)

        if weighted and directed:
            cons = WeightedDirectedGraph
        
        elif weighted and not directed:
            cons = WeightedUndirectedGraph

        elif not weighted and directed:
            cons = UnweightedDirectedGraph
        
        elif not weighted and not directed:
            cons = UnweightedUndirectedGraph

        return cons()


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ABSTRACT BASE CLASSES


class UnweightedGraph:
    def __init__(self):
        self.vs = set()
        self.es = set()
        self.adj_dict = dict()

    def addV(self, v):
        # add v to set of vertices
        sv = str(v)
        self.vs.add(sv)
        # add v to adjacency dict
        self.adj_dict.setdefault(sv, set())

    @property
    def V(self):
        return list(self.vs)
             
    @property
    def E(self):
        return list(self.es)
     
    def neighbors(self, v):
        return set(self.adj_dict[v])

    def _construct_path(self, src, dst, p):
        path = []
        curr = dst
        while p[curr] is not None and curr != src:
            path.append(curr)
            curr = p[curr]

        if len(path) == 0:
            return None
        path.append(curr)
        return path[::-1]

    def bfs_path(self, source, destination, get_seen = False):
        """
        Return a list of vertices representing a path from `source`
        to `destination` if such a path exists, and None if it doesn't
        """
        paths = deque()
        paths.append( [source] )
        seen = { source }
        while len(paths) != 0:
            path = paths.popleft()
            for v in self.neighbors(path[-1]):
                if v not in seen: 
                    seen.add(v)
                    npath = path + [ v ]
                    if v == destination:
                        if get_seen:
                            return npath, seen
                        else:
                            return npath
                    paths.append(npath)
        return None

    def dfs(self, source):
        s = str(source)
        c = { v : 'W'  for v in self.vs } # color
        p = { v : None for v in self.vs } # parent
        d = { v : None for v in self.vs } # discovery time
        f = { v : None for v in self.vs } # finish time
        time = 0

        def dfs_visit(u, time):
            time += 1
            d[u] = time
            c[u] = 'G'
            for v in self.neighbors(u):
                if c[v] == 'W':
                    p[v] = u
                    dfs_visit(v, time)
            c[u] = 'B'
            time += 1
            f[u] = time

        dfs_visit(s, time)

        return c, p, d, f

    def dfs_path(self, source, destination):
        src, dst = str(source), str(destination)
        c, p, _, _ = self.dfs(source)

        if c[dst] != 'B':
            return None

        return self._construct_path(src, dst, p)

    def is_connected(self):
        start = list(self.vs)[0]
        c, _, _, _ = self.dfs(start)
        return all(c[v] == 'B' for v in self.vs)

    def is_cyclic(self):
        visited = set()
        working = set()

        def detect_cycle(u):
            visited.add(u)
            working.add(u)
            for v in self.neighbors(u):
                if v in working:
                    return True
                elif v not in visited:
                    if detect_cycle(v):
                        return True
            working.remove(u)
            return False
        
        for u in self.vs:
            if u not in visited:
                if detect_cycle(u):
                    return True
        return False

    def floyd_warshall_paths(self):

        def init_matrix():
            D = dict()
            for i in self.vs:
                for j in self.vs:
                    D[(i, j)] = float('inf')
            return D

        D = init_matrix()
        for i, j in self.es:
            D[(i, j)] = 1

        for k in range(len(self.vs)):
            sk = str(k)
            for i in self.vs:
                for j in self.vs:
                    D[(i, j)] = min(D[(i, j)], D[(i, sk)] + D[(sk, j)])
        return D



class WeightedGraph(UnweightedGraph):
    def __init__(self):
        UnweightedGraph.__init__(self)
        self.weights = dict()

    def weight_of(self, u, v):
        su, sv = str(u), str(v)
        if (su, sv) not in self.weights:
            return None
        return self.weights[(su, sv)]

    def _init_single_source(self, src):
        if src not in self.vs:
            raise Exception('Source vertex not in graph')
        d = { v : float('inf') for v in self.vs }
        p = { v :     None     for v in self.vs }
        d[src] = 0
        return d, p

    def _relax(self, u, v, d, p, do_if_relax = tuple()):
        if d[v] > d[u] + self.weights[(u, v)]:
            d[v] = d[u] + self.weights[(u, v)]
            p[v] = u
            for task in do_if_relax:
                task()

    def dijkstra_path(self, source, destination):
        src, dst = str(source), str(destination)
        if src == dst:
            return [src] if src in self.adj_dict[src] else None

        # initialize single source
        d, p = self._init_single_source(src)

        hd = heapdict()
        for v, dv in d.items():
            hd[v] = dv

        for _ in range(len(self.vs)):
            u, _ = hd.popitem()
            for v in self.neighbors(u):
                change_key = lambda : hd.__setitem__(v, d[u] + self.weights[(u, v)])
                self._relax(u, v, d, p, [change_key])

        # reconstruct shortest path
        return self._construct_path(src, dst, p)

    def bellman_ford_path(self, source, destination):
        src, dst = str(source), str(destination)
        if src == dst:
            return [src] if src in self.adj_dict[src] else None

        # initialize single source
        d, p = self._init_single_source(src)

        # relax each edge at most |V| - 1 times
        for _ in range(len(self.vs) - 1):
            for u, v in self.es:
                self._relax(u, v, d, p)

        # detect negative weight cycles
        for u, v in self.es:
            if d[v] > d[u] + self.weights[(u, v)]:
                raise Exception('Found negative weight cycle')

        # reconstruct shortest path
        return self._construct_path(src, dst, p)

    def floyd_warshall_paths(self):

        def init_matrix():
            D = dict()
            for i in self.vs:
                for j in self.vs:
                    D[(i, j)] = float('inf')
            return D

        D = init_matrix()
        for i, j in self.weights:
            D[(i, j)] = self.weights[(i, j)]

        for k in range(len(self.vs)):
            sk = str(k)
            for i in self.vs:
                for j in self.vs:
                    D[(i, j)] = min(D[(i, j)], D[(i, sk)] + D[(sk, j)])
        return D

        
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# TEST HELPERS


def check_valid_path(G, src, dst, p, is_none = False):
    if is_none:
        assert p is None
    else:
        assert src == p[0]
        assert dst == p[-1]
        for i in range(len(p) - 1):
            assert (p[i], p[i+1]) in G.E


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


class UnweightedDirectedGraph(UnweightedGraph):
    def addE(self, u, v):
        # add u, v to vertices, edges, and adjacency dict
        su, sv = str(u), str(v)
        self.addV(su)
        self.addV(sv)
        self.es.add((su, sv))
        self.adj_dict[su].add(sv)


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# TESTS


def test_unweighted_directed_graph_init():
    G = UnweightedDirectedGraph()
    # A
    G.addV('A')
    assert sorted(G.V) == ['A']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    # A, B
    G.addV('B')
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    assert sorted(G.neighbors('B')) == []
    # A --> B
    G.addE('A', 'B')
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == [('A', 'B')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == []
    # A --> B, B --> C
    G.addE('B', 'C')
    assert sorted(G.V) == ['A', 'B', 'C'], sorted(G.V)
    assert sorted(G.E) == [('A', 'B'), ('B', 'C')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['C']
    assert sorted(G.neighbors('C')) == []
    # A --> B, B --> C, C --> D
    G.addE('C', 'D')
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('B', 'C'), ('C', 'D')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['C']
    assert sorted(G.neighbors('C')) == ['D']
    assert sorted(G.neighbors('D')) == []
    # A --> B, B --> C, C --> D, A --> D
    G.addE('A', 'D')
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]
    assert sorted(G.neighbors('A')) == ['B', 'D']
    assert sorted(G.neighbors('B')) == ['C']
    assert sorted(G.neighbors('C')) == ['D']
    assert sorted(G.neighbors('D')) == []


def test_unweighted_directed_graph_cycles():
    G1 = UnweightedDirectedGraph()
    # A
    G1.addV('A')
    assert not G1.is_cyclic()
    # B
    G1.addV('B')
    assert not G1.is_cyclic()
    # A --> B
    G1.addE('A', 'B')
    assert not G1.is_cyclic()
    # A --> B, B --> A
    G1.addE('B', 'A')
    assert G1.is_cyclic()

    G2 = UnweightedDirectedGraph()
    # 0, 1, 2, 3, 4, 5, 6
    for i in range(7):
        G2.addV(i)
        assert not G2.is_cyclic()
    # 0 --> 6
    G2.addE(0, 6)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1
    G2.addE(0, 1)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2
    G2.addE(1, 2)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3
    G2.addE(2, 3)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4
    G2.addE(3, 4)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5
    G2.addE(4, 5)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5, 5 --> 6
    G2.addE(5, 6)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5, 5 --> 6, 6 --> 1
    G2.addE(6, 1)
    assert G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5, 5 --> 6, 6 --> 1, 5 --> 2
    G2.addE(5, 2)
    assert G2.is_cyclic()


def test_unweighted_directed_graph_path():
    path_algs = [
        lambda G, src, dst : G.bfs_path(src, dst),
        lambda G, src, dst : G.dfs_path(src, dst)
    ]

    for pa in path_algs:
        G = UnweightedDirectedGraph()
        
        # A
        G.addV('A')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        
        # A, B
        G.addV('B')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'), is_none = True)
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'), is_none = True)
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        
        # A --> B
        G.addE('A', 'B')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'), is_none = True)
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)

        # A --> B, B --> C
        G.addE('B', 'C')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
         
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'), is_none = True)
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
         
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'), is_none = True)
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'), is_none = True)
        check_valid_path(G, 'C', 'C', pa(G, 'C', 'C'), is_none = True)
        
        # A --> B, B --> C, C --> D
        G.addE('C', 'D')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
        check_valid_path(G, 'A', 'D', pa(G, 'A', 'D'))
        
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'), is_none = True)
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
        check_valid_path(G, 'B', 'D', pa(G, 'B', 'D'))
        
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'), is_none = True)
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'), is_none = True)
        check_valid_path(G, 'C', 'C', pa(G, 'C', 'C'), is_none = True)
        check_valid_path(G, 'C', 'D', pa(G, 'C', 'D'))
        
        check_valid_path(G, 'D', 'A', pa(G, 'D', 'A'), is_none = True)
        check_valid_path(G, 'D', 'B', pa(G, 'D', 'B'), is_none = True)
        check_valid_path(G, 'D', 'C', pa(G, 'D', 'C'), is_none = True)
        check_valid_path(G, 'D', 'D', pa(G, 'D', 'D'), is_none = True)
        
        # A --> B, B --> C, C --> D, A --> D
        G.addE('A', 'D')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
        check_valid_path(G, 'A', 'D', pa(G, 'A', 'D'))
        
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'), is_none = True)
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
        check_valid_path(G, 'B', 'D', pa(G, 'B', 'D'))
        
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'), is_none = True)
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'), is_none = True)
        check_valid_path(G, 'C', 'C', pa(G, 'C', 'C'), is_none = True)
        check_valid_path(G, 'C', 'D', pa(G, 'C', 'D'))
        
        check_valid_path(G, 'D', 'A', pa(G, 'D', 'A'), is_none = True)
        check_valid_path(G, 'D', 'B', pa(G, 'D', 'B'), is_none = True)
        check_valid_path(G, 'D', 'C', pa(G, 'D', 'C'), is_none = True)
        check_valid_path(G, 'D', 'D', pa(G, 'D', 'D'), is_none = True)


def test_unweighted_directed_graph():
    test_unweighted_directed_graph_init()
    test_unweighted_directed_graph_cycles()
    test_unweighted_directed_graph_path()


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


class WeightedDirectedGraph(WeightedGraph):
    def addE(self, u, v, w):
        # add u, v to vertices, edges, and adjacency dict
        su, sv = str(u), str(v)
        self.addV(su)
        self.addV(sv)
        self.es.add((su, sv))
        self.adj_dict[su].add(sv)
        self.weights[(su, sv)] = float(w)


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# TESTS


def test_weighted_directed_graph_init():
    G = WeightedDirectedGraph()
    # A
    G.addV('A')
    assert sorted(G.V) == ['A']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    assert G.weight_of('A', 'A') is None
    # A, B
    G.addV('B')
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    assert sorted(G.neighbors('B')) == []
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') is None
    assert G.weight_of('B', 'A') is None
    assert G.weight_of('B', 'B') is None
    # A --> B
    G.addE('A', 'B', 1)
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == [('A', 'B')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == []
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('B', 'A') is None
    assert G.weight_of('B', 'B') is None
    # A --> B, B --> C
    G.addE('B', 'C', 4)
    assert sorted(G.V) == ['A', 'B', 'C'], sorted(G.V)
    assert sorted(G.E) == [('A', 'B'), ('B', 'C')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['C']
    assert sorted(G.neighbors('C')) == []
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('A', 'C') is None
    assert G.weight_of('B', 'A') is None
    assert G.weight_of('B', 'B') is None
    assert G.weight_of('B', 'C') == 4
    assert G.weight_of('C', 'A') is None
    assert G.weight_of('C', 'B') is None
    assert G.weight_of('C', 'C') is None
    # A --> B, B --> C, C --> D
    G.addE('C', 'D', 9)
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('B', 'C'), ('C', 'D')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['C']
    assert sorted(G.neighbors('C')) == ['D']
    assert sorted(G.neighbors('D')) == []
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('A', 'C') is None
    assert G.weight_of('A', 'D') is None
    assert G.weight_of('B', 'A') is None
    assert G.weight_of('B', 'B') is None
    assert G.weight_of('B', 'C') == 4
    assert G.weight_of('B', 'D') is None
    assert G.weight_of('C', 'A') is None
    assert G.weight_of('C', 'B') is None
    assert G.weight_of('C', 'C') is None
    assert G.weight_of('C', 'D') == 9
    assert G.weight_of('D', 'A') is None
    assert G.weight_of('D', 'B') is None
    assert G.weight_of('D', 'C') is None
    assert G.weight_of('D', 'D') is None

    # A --> B, B --> C, C --> D, A --> D
    G.addE('A', 'D', 16)
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('A', 'D'), ('B', 'C'), ('C', 'D')]
    assert sorted(G.neighbors('A')) == ['B', 'D']
    assert sorted(G.neighbors('B')) == ['C']
    assert sorted(G.neighbors('C')) == ['D']
    assert sorted(G.neighbors('D')) == []
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('A', 'C') is None
    assert G.weight_of('A', 'D') == 16
    assert G.weight_of('B', 'A') is None
    assert G.weight_of('B', 'B') is None
    assert G.weight_of('B', 'C') == 4
    assert G.weight_of('B', 'D') is None
    assert G.weight_of('C', 'A') is None
    assert G.weight_of('C', 'B') is None
    assert G.weight_of('C', 'C') is None
    assert G.weight_of('C', 'D') == 9
    assert G.weight_of('D', 'A') is None
    assert G.weight_of('D', 'B') is None
    assert G.weight_of('D', 'C') is None
    assert G.weight_of('D', 'D') is None


def test_weighted_directed_graph_cycles():
    G1 = WeightedDirectedGraph()
    # A
    G1.addV('A')
    assert not G1.is_cyclic()
    # B
    G1.addV('B')
    assert not G1.is_cyclic()
    # A --> B
    G1.addE('A', 'B', 1)
    assert not G1.is_cyclic()
    # A --> B, B --> A
    G1.addE('B', 'A', 2)
    assert G1.is_cyclic()

    G2 = WeightedDirectedGraph()
    # 0, 1, 2, 3, 4, 5, 6
    for i in range(7):
        G2.addV(i)
        assert not G2.is_cyclic()
    # 0 --> 6
    G2.addE(0, 6, 3)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1
    G2.addE(0, 1, 4)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2
    G2.addE(1, 2, 5)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3
    G2.addE(2, 3, 6)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4
    G2.addE(3, 4, 7)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5
    G2.addE(4, 5, 8)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5, 5 --> 6
    G2.addE(5, 6, 9)
    assert not G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5, 5 --> 6, 6 --> 1
    G2.addE(6, 1, 10)
    assert G2.is_cyclic()
    # 0 --> 6, 0 --> 1, 1 --> 2, 2 --> 3, 3 --> 4, 4 --> 5, 5 --> 6, 6 --> 1, 5 --> 2
    G2.addE(5, 2, 11)
    assert G2.is_cyclic()


def test_weighted_directed_graph_single_source_shortest_path():
    shortest_path_algs = [
        lambda G, src, dst : G.bellman_ford_path(src, dst), 
        lambda G, src, dst : G.dijkstra_path(src, dst)
    ]

    for spa in shortest_path_algs:
        G = WeightedDirectedGraph()
        
        # A
        G.addV('A')
        assert spa(G, 'A', 'A') is None
        
        # A, B
        G.addV('B')
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') is None
        assert spa(G, 'B', 'A') is None
        assert spa(G, 'B', 'B') is None
        
        # A --> B
        G.addE('A', 'B', 1)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'B', 'A') is None
        assert spa(G, 'B', 'B') is None

        # A --> B, B --> C
        G.addE('B', 'C', 4)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'A', 'C') == ['A', 'B', 'C']
         
        assert spa(G, 'B', 'A') is None
        assert spa(G, 'B', 'B') is None
        assert spa(G, 'B', 'C') == ['B', 'C']
         
        assert spa(G, 'C', 'A') is None
        assert spa(G, 'C', 'B') is None
        assert spa(G, 'C', 'C') is None
        
        # A --> B, B --> C, C --> D
        G.addE('C', 'D', 9)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'A', 'C') == ['A', 'B', 'C']
        assert spa(G, 'A', 'D') == ['A', 'B', 'C', 'D']
        
        assert spa(G, 'B', 'A') is None
        assert spa(G, 'B', 'B') is None
        assert spa(G, 'B', 'C') == ['B', 'C']
        assert spa(G, 'B', 'D') == ['B', 'C', 'D']
        
        assert spa(G, 'C', 'A') is None
        assert spa(G, 'C', 'B') is None
        assert spa(G, 'C', 'C') is None
        assert spa(G, 'C', 'D') == ['C', 'D']
        
        assert spa(G, 'D', 'A') is None
        assert spa(G, 'D', 'B') is None
        assert spa(G, 'D', 'C') is None
        assert spa(G, 'D', 'D') is None
        
        # A --> B, B --> C, C --> D, A --> D
        G.addE('A', 'D', 16)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'A', 'C') == ['A', 'B', 'C']
        assert spa(G, 'A', 'D') == ['A', 'B', 'C', 'D']
        
        assert spa(G, 'B', 'A') is None
        assert spa(G, 'B', 'B') is None
        assert spa(G, 'B', 'C') == ['B', 'C']
        assert spa(G, 'B', 'D') == ['B', 'C', 'D']
        
        assert spa(G, 'C', 'A') is None
        assert spa(G, 'C', 'B') is None
        assert spa(G, 'C', 'C') is None
        assert spa(G, 'C', 'D') == ['C', 'D']
        
        assert spa(G, 'D', 'A') is None
        assert spa(G, 'D', 'B') is None
        assert spa(G, 'D', 'C') is None
        assert spa(G, 'D', 'D') is None


def test_weighted_directed_graph():
    test_weighted_directed_graph_init()
    test_weighted_directed_graph_cycles()
    test_weighted_directed_graph_single_source_shortest_path()


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


class UnweightedUndirectedGraph(UnweightedDirectedGraph):
    def addE(self, u, v):
        UnweightedDirectedGraph.addE(self, u, v)
        UnweightedDirectedGraph.addE(self, v, u)


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# TESTS


def test_unweighted_undirected_graph_init():
    G = UnweightedUndirectedGraph()
    # A
    G.addV('A')
    assert sorted(G.V) == ['A']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    # A, B
    G.addV('B')
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    assert sorted(G.neighbors('B')) == []
    # A --- B
    G.addE('A', 'B')
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == [('A', 'B'), ('B', 'A')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['A']
    # A --- B, B --- C
    G.addE('B', 'C')
    assert sorted(G.V) == ['A', 'B', 'C'], sorted(G.V)
    assert sorted(G.E) == [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['A', 'C']
    assert sorted(G.neighbors('C')) == ['B']
    # A --- B, B --- C, C --- D
    G.addE('C', 'D')
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), ('D', 'C')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['A', 'C']
    assert sorted(G.neighbors('C')) == ['B', 'D']
    assert sorted(G.neighbors('D')) == ['C']
    # A --> B, B --> C, C --> D, A --> D
    G.addE('A', 'D')
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('A', 'D'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), ('D', 'A'), ('D', 'C')]
    assert sorted(G.neighbors('A')) == ['B', 'D']
    assert sorted(G.neighbors('B')) == ['A', 'C']
    assert sorted(G.neighbors('C')) == ['B', 'D']
    assert sorted(G.neighbors('D')) == ['A', 'C']


def test_unweighted_undirected_graph_path():
    path_algs = [
        (lambda G, src, dst : G.bfs_path(src, dst), 'bfs'),
        (lambda G, src, dst : G.dfs_path(src, dst), 'dfs')
    ]

    for pa, name in path_algs:
        G = UnweightedUndirectedGraph()
        
        # A
        G.addV('A')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        
        # A, B
        G.addV('B')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'), is_none = True)
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'), is_none = True)
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        
        # A --- B
        G.addE('A', 'B')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'))
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'))
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)

        # A --- B, B --- C
        G.addE('B', 'C')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
         
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'))
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'))
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
         
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'))
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'))
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'))
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'))
        check_valid_path(G, 'C', 'C', pa(G, 'C', 'C'), is_none = True)
        
        # A --- B, B --- C, C --- D
        G.addE('C', 'D')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
        check_valid_path(G, 'A', 'D', pa(G, 'A', 'D'))
        check_valid_path(G, 'A', 'D', pa(G, 'A', 'D'))
        
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'))
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'))
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
        check_valid_path(G, 'B', 'D', pa(G, 'B', 'D'))
        check_valid_path(G, 'B', 'D', pa(G, 'B', 'D'))
        
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'))
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'))
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'))
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'))
        check_valid_path(G, 'C', 'C', pa(G, 'C', 'C'), is_none = True)
        check_valid_path(G, 'C', 'D', pa(G, 'C', 'D'))
        check_valid_path(G, 'C', 'D', pa(G, 'C', 'D'))
        
        check_valid_path(G, 'D', 'A', pa(G, 'D', 'A'))
        check_valid_path(G, 'D', 'A', pa(G, 'D', 'A'))
        check_valid_path(G, 'D', 'B', pa(G, 'D', 'B'))
        check_valid_path(G, 'D', 'B', pa(G, 'D', 'B'))
        check_valid_path(G, 'D', 'C', pa(G, 'D', 'C'))
        check_valid_path(G, 'D', 'C', pa(G, 'D', 'C'))
        check_valid_path(G, 'D', 'D', pa(G, 'D', 'D'), is_none = True)
        
        # A --- B, B --- C, C --- D, A --- D
        G.addE('A', 'D')
        check_valid_path(G, 'A', 'A', pa(G, 'A', 'A'), is_none = True)
        check_valid_path(G, 'A', 'B', pa(G, 'A', 'B'))
        check_valid_path(G, 'A', 'C', pa(G, 'A', 'C'))
        check_valid_path(G, 'A', 'D', pa(G, 'A', 'D'))
        
        check_valid_path(G, 'B', 'A', pa(G, 'B', 'A'))
        check_valid_path(G, 'B', 'B', pa(G, 'B', 'B'), is_none = True)
        check_valid_path(G, 'B', 'C', pa(G, 'B', 'C'))
        check_valid_path(G, 'B', 'D', pa(G, 'B', 'D'))
        
        check_valid_path(G, 'C', 'A', pa(G, 'C', 'A'))
        check_valid_path(G, 'C', 'B', pa(G, 'C', 'B'))
        check_valid_path(G, 'C', 'C', pa(G, 'C', 'C'), is_none = True)
        check_valid_path(G, 'C', 'D', pa(G, 'C', 'D'))
        
        check_valid_path(G, 'D', 'A', pa(G, 'D', 'A'))
        check_valid_path(G, 'D', 'B', pa(G, 'D', 'B'))
        check_valid_path(G, 'D', 'C', pa(G, 'D', 'C'))
        check_valid_path(G, 'D', 'D', pa(G, 'D', 'D'), is_none = True)


def test_unweighted_undirected_graph():
    test_unweighted_undirected_graph_init()
    test_unweighted_undirected_graph_path()


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


class WeightedUndirectedGraph(WeightedDirectedGraph):
    def addE(self, u, v, w):
        WeightedDirectedGraph.addE(self, u, v, w)
        WeightedDirectedGraph.addE(self, v, u, w)

    def kruskal_mst(self):
        assert self.is_connected(), "Can only find MST of a connected graph"
        uf = UnionFind()
        mst = set()

        for v in self.vs:
            uf.make_set(v)

        half = set()
        for u, v in sorted(self.es):
            if (v, u) not in half:
                half.add((u, v))

        w = 0
        vs = set()
        for u, v in sorted(half, key = lambda e : self.weights[e]):
            if len(vs) == len(self.vs):
                return mst, w

            if uf.find_set(u) != uf.find_set(v):
                uf.union(u, v)
                mst.add((u, v))
                vs.add(u)
                vs.add(v)
                w += self.weights[(u, v)]



# ////////////////////////////////////////////////////////////////////////////////////////////////////
# TESTS


def test_weighted_undirected_graph_init():
    G = WeightedUndirectedGraph()
    # A
    G.addV('A')
    assert sorted(G.V) == ['A']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    assert G.weight_of('A', 'A') is None
    # A, B
    G.addV('B')
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == []
    assert sorted(G.neighbors('A')) == []
    assert sorted(G.neighbors('B')) == []
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') is None
    assert G.weight_of('B', 'A') is None
    assert G.weight_of('B', 'B') is None
    # A --> B
    G.addE('A', 'B', 1)
    assert sorted(G.V) == ['A', 'B']
    assert sorted(G.E) == [('A', 'B'), ('B', 'A')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['A']
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('B', 'A') == 1
    assert G.weight_of('B', 'B') is None
    # A --> B, B --> C
    G.addE('B', 'C', 4)
    assert sorted(G.V) == ['A', 'B', 'C'], sorted(G.V)
    assert sorted(G.E) == [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['A', 'C']
    assert sorted(G.neighbors('C')) == ['B']
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('A', 'C') is None
    assert G.weight_of('B', 'A') == 1
    assert G.weight_of('B', 'B') is None
    assert G.weight_of('B', 'C') == 4
    assert G.weight_of('C', 'A') is None
    assert G.weight_of('C', 'B') == 4
    assert G.weight_of('C', 'C') is None
    # A --> B, B --> C, C --> D
    G.addE('C', 'D', 9)
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), ('D', 'C')]
    assert sorted(G.neighbors('A')) == ['B']
    assert sorted(G.neighbors('B')) == ['A', 'C']
    assert sorted(G.neighbors('C')) == ['B', 'D']
    assert sorted(G.neighbors('D')) == ['C']
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('A', 'C') is None
    assert G.weight_of('A', 'D') is None
    assert G.weight_of('B', 'A') == 1
    assert G.weight_of('B', 'B') is None
    assert G.weight_of('B', 'C') == 4
    assert G.weight_of('B', 'D') is None
    assert G.weight_of('C', 'A') is None
    assert G.weight_of('C', 'B') == 4
    assert G.weight_of('C', 'C') is None
    assert G.weight_of('C', 'D') == 9
    assert G.weight_of('D', 'A') is None
    assert G.weight_of('D', 'B') is None
    assert G.weight_of('D', 'C') == 9
    assert G.weight_of('D', 'D') is None
    # A --> B, B --> C, C --> D, A --> D
    G.addE('A', 'D', 16)
    assert sorted(G.V) == ['A', 'B', 'C', 'D']
    assert sorted(G.E) == [('A', 'B'), ('A', 'D'), ('B', 'A'), ('B', 'C'), ('C', 'B'), ('C', 'D'), ('D', 'A'), ('D', 'C')]
    assert sorted(G.neighbors('A')) == ['B', 'D']
    assert sorted(G.neighbors('B')) == ['A', 'C']
    assert sorted(G.neighbors('C')) == ['B', 'D']
    assert sorted(G.neighbors('D')) == ['A', 'C']
    assert G.weight_of('A', 'A') is None
    assert G.weight_of('A', 'B') == 1
    assert G.weight_of('A', 'C') is None
    assert G.weight_of('A', 'D') == 16
    assert G.weight_of('B', 'A') == 1
    assert G.weight_of('B', 'B') is None
    assert G.weight_of('B', 'C') == 4
    assert G.weight_of('B', 'D') is None
    assert G.weight_of('C', 'A') is None
    assert G.weight_of('C', 'B') == 4
    assert G.weight_of('C', 'C') is None
    assert G.weight_of('C', 'D') == 9
    assert G.weight_of('D', 'A') == 16
    assert G.weight_of('D', 'B') is None
    assert G.weight_of('D', 'C') == 9
    assert G.weight_of('D', 'D') is None


def test_weighted_undirected_graph_mst():
    G = WeightedUndirectedGraph()

    # A --(1)-- B
    G.addE('A', 'B', 1)
    assert {('A', 'B')}, 1 == G.kruskal_mst()

    # A --(1)-- B, A --(2)-- C
    G.addE('A', 'C', 2)
    assert {('A', 'B'), ('A', 'C')}, 1+2 == G.kruskal_mst()

    # A --(1)-- B, A --(2)-- C, B --(3)-- C
    G.addE('C', 'B', 3)
    assert {('A', 'B'), ('A', 'C')}, 1+2 == G.kruskal_mst()

    # A --(1)-- B, A --(2)-- C, B --(3)-- C, C --(4)-- D, B --(5)-- D
    G.addE('C', 'D', 4)
    G.addE('B', 'D', 5)
    assert {('A', 'B'), ('A', 'C'), ('C', 'D')}, 1+2+4 == G.kruskal_mst()


def test_weighted_undirected_graph_single_source_shortest_path():
    shortest_path_algs = [
        lambda G, src, dst : G.bellman_ford_path(src, dst), 
        lambda G, src, dst : G.dijkstra_path(src, dst)
    ]

    for spa in shortest_path_algs:
        G = WeightedUndirectedGraph()
        
        # A
        G.addV('A')
        assert spa(G, 'A', 'A') is None
        
        # A, B
        G.addV('B')
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') is None
        assert spa(G, 'B', 'A') is None
        assert spa(G, 'B', 'B') is None
        
        # A --> B
        G.addE('A', 'B', 1)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'B', 'A') == ['B', 'A']
        assert spa(G, 'B', 'B') is None

        # A --> B, B --> C
        G.addE('B', 'C', 4)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'A', 'C') == ['A', 'B', 'C']
         
        assert spa(G, 'B', 'A') == ['B', 'A']
        assert spa(G, 'B', 'B') is None
        assert spa(G, 'B', 'C') == ['B', 'C']
         
        assert spa(G, 'C', 'A') == ['C', 'B', 'A']
        assert spa(G, 'C', 'B') == ['C', 'B']
        assert spa(G, 'C', 'C') is None
        
        # A --> B, B --> C, C --> D
        G.addE('C', 'D', 9)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'A', 'C') == ['A', 'B', 'C']
        assert spa(G, 'A', 'D') == ['A', 'B', 'C', 'D']
        
        assert spa(G, 'B', 'A') == ['B', 'A']
        assert spa(G, 'B', 'B') is None
        assert spa(G, 'B', 'C') == ['B', 'C']
        assert spa(G, 'B', 'D') == ['B', 'C', 'D']
        
        assert spa(G, 'C', 'A') == ['C', 'B', 'A']
        assert spa(G, 'C', 'B') == ['C', 'B']
        assert spa(G, 'C', 'C') is None
        assert spa(G, 'C', 'D') == ['C', 'D']
        
        assert spa(G, 'D', 'A') == ['D', 'C', 'B', 'A']
        assert spa(G, 'D', 'B') == ['D', 'C', 'B']
        assert spa(G, 'D', 'C') == ['D', 'C']
        assert spa(G, 'D', 'D') is None
        
        # A --> B, B --> C, C --> D, A --> D
        G.addE('A', 'D', 16)
        assert spa(G, 'A', 'A') is None
        assert spa(G, 'A', 'B') == ['A', 'B']
        assert spa(G, 'A', 'C') == ['A', 'B', 'C']
        assert spa(G, 'A', 'D') == ['A', 'B', 'C', 'D']
        
        assert spa(G, 'B', 'A') == ['B', 'A']
        assert spa(G, 'B', 'B') is None
        assert spa(G, 'B', 'C') == ['B', 'C']
        assert spa(G, 'B', 'D') == ['B', 'C', 'D']
        
        assert spa(G, 'C', 'A') == ['C', 'B', 'A']
        assert spa(G, 'C', 'B') == ['C', 'B']
        assert spa(G, 'C', 'C') is None
        assert spa(G, 'C', 'D') == ['C', 'D']
        
        assert spa(G, 'D', 'A') == ['D', 'C', 'B', 'A']
        assert spa(G, 'D', 'B') == ['D', 'C', 'B']
        assert spa(G, 'D', 'C') == ['D', 'C']
        assert spa(G, 'D', 'D') is None


def test_weighted_undirected_graph():
    test_weighted_undirected_graph_init()
    test_weighted_undirected_graph_mst()
    test_weighted_undirected_graph_single_source_shortest_path()


# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


def test_graph():
    tests = [test_unweighted_directed_graph, test_unweighted_undirected_graph, \
             test_weighted_directed_graph, test_weighted_undirected_graph]
    num_runs = 10
    s = ''
    for t in tests:
        for _ in range(num_runs):
            t()
        s += '.'
    print(s)    


G = Graph.new(directed = False, weighted = False)
G.addV(0)
print(sorted(G.floyd_warshall_paths().items()))
G.addV(1)
print(sorted(G.floyd_warshall_paths().items()))
G.addE(0, 1)
print(sorted(G.floyd_warshall_paths().items()))


if __name__ == '__main__':
    doctest.testmod()
    try:
        test_graph()
        print('Graph tests pass')
    except AssertionError as e:
        print('Graph tests fail')
        raise e