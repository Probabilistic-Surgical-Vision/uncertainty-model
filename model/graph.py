import networkx as nx
import collections

from networkx import Graph
from typing import List, Optional, Tuple

Node = collections.namedtuple('Node', ['id', 'type', 'inputs'])


def get_graph_info(graph: Graph) -> Tuple[List[Node], List[int], List[int]]:
    inputs, outputs, nodes = list(), list(), list()

    for i in range(graph.number_of_nodes()):
        neighbours = list(graph.neighbors(i))
        type = "intermediate"

        if i < min(neighbours):
            inputs.append(i)
            type = "input"
        elif i > max(neighbours):
            outputs.append(i)
            type = "output"

        node = Node(i, type, [n for n in neighbours if n < i])
        nodes.append(node)

    return nodes, inputs, outputs

def build_graph(nodes: int, k: float, p: float, seed: Optional[int] = None):
    return nx.connected_watts_strogatz_graph(nodes, k, p, seed=seed)

def save_graph(graph, path):
    nx.write_yaml(graph, path)

def load_graph(path):
    return nx.read_yaml(path)
