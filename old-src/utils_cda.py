# pip install synthetic-data-generation
import numpy as np
import matplotlib.pyplot as plt
from causallearn.graph.GraphClass import GeneralGraph, Edge, Endpoint, GraphNode
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io

from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCMBased import lingam
import castle

# Declared just for export clarity:
ges = ges
pc = pc
lingam = lingam


def img(G, labels=None):
    pydot_object = GraphUtils.to_pydot(G, labels=labels)
    png = io.BytesIO(pydot_object.create_png(f="png"))  # type: ignore
    return plt.imread(png, format="png")


def adj_matrix(G):
    idx = {label: i for i, label in enumerate(G.get_node_names())}
    n = len(idx)
    mat = np.zeros((n, n), dtype=int)
    E = G.get_graph_edges()
    for e in E:
        i = idx[e.node1.get_name()]
        j = idx[e.node2.get_name()]
        if not e.points_toward(e.node1) and not e.points_toward(e.node2):
            mat[i][j] = mat[j][i] = 1
        elif e.points_toward(e.node2):
            mat[i][j] = 1
        elif e.points_toward(e.node1):
            mat[j][i] = 1
    return mat


def dist_adj_matrix(adj_ref, adj):
    n = adj_ref.shape[0]
    assert adj_ref.shape == (n, n)
    assert adj.shape == (n, n)
    total = 0
    edge = lambda lr, rl: ("--" if lr else "<-") if rl else ("->" if lr else "  ")
    for i in range(n):
        for j in range(i + 1, n):
            first = edge(adj_ref[i, j], adj_ref[j, i])
            second = edge(adj[i, j], adj[j, i])
            if first == second:
                cost = 0
            elif (first + second).count("-") == 3:
                cost = 1  # Missing/added direction
            elif first == "  " or second == "  ":
                cost = 2  # Missing/added edge
            else:
                assert first + second in ("<-->", "-><-")
                cost = 3  # Opposite directions
            total += cost
    return total


class PostCDA:
    def __init__(self, result, labels=None, threshold=None, **kwargs):
        self.result = result
        self.labels = labels

        def ignore_errors(f):
            try:
                return f()
            except:
                return None

        G = ignore_errors(lambda: result["G"]) or ignore_errors(lambda: result.G)
        if G is None:
            adj = ignore_errors(lambda: result.causal_matrix)
            if adj is None:
                # LinGAM case...
                adjW = result.adjacency_matrix_
                assert adjW is not None
                adj = (np.abs(adjW) > threshold).astype(int)
                adj = adj.T  # Carlos: the output seems to be transposed!
            m = len(adj)
            node_names = [("X%d" % (i + 1)) for i in range(m)]
            G = GeneralGraph([GraphNode(name) for name in node_names])
            for i in range(m):
                for j in range(i + 1, m):
                    if adj[i, j] or adj[j, i]:
                        end1 = Endpoint.TAIL if adj[i, j] else Endpoint.ARROW
                        end2 = Endpoint.TAIL if adj[j, i] else Endpoint.ARROW
                        e = Edge(G.nodes[i], G.nodes[j], end1, end2)
                        G.add_edge(e)
        else:
            adj = adj_matrix(G)

        self.adj = adj
        self.img = img(G, self.labels)
