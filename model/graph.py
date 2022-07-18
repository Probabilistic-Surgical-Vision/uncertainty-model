import networkx as nx
import collections

Node = collections.namedtuple('Node', ['id', 'inputs', 'type'])


def get_graph_info(graph):
    input_nodes = []
    output_nodes = []
    nodes = []
    for node in range(graph.number_of_nodes()):
        tmp = list(graph.neighbors(node))
        tmp.sort()
        type = -1
        if node < tmp[0]:
            input_nodes.append(node)
            type = 0
        if node > tmp[-1]:
            output_nodes.append(node)
            type = 1
        nodes.append(Node(node, [n for n in tmp if n < node], type))
    return nodes, input_nodes, output_nodes

def build_graph(nodes, args, seed= None):
    if seed == None:
        seed= args.seed

    if args.graph_model == 'ER':
        return nx.erdos_renyi_graph(nodes, args.P, seed)
    elif args.graph_model == 'BA':
        return nx.barabasi_albert_graph(nodes, args.M, seed)
    elif args.graph_model == 'WS':
        return nx.connected_watts_strogatz_graph(nodes, args.K, args.P, tries=200, seed=seed)

def save_graph(graph, path):
    nx.write_yaml(graph, path)

def load_graph(path):
    return nx.read_yaml(path)
