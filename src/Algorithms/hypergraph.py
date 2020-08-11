from collections.abc import Iterable

class hypergraph:

    def __init__(self, nodes=None):
        assert nodes is None or isinstance(nodes, Iterable), \
            'Parameter `nodes` must either be None or an Iterable object.'

        self.nodes = set() if nodes is None else set(nodes)
        self.edges = {}
        self.n_edges = 0
        self._node_edgesets = {node: set() for node in self.nodes}


    def add_node(self, node_id=None):
        assert node_id not in self.nodes, \
            '`node_id` is already in the hypergraph.'

        if node_id is None:
            node_id = len(self.nodes)
            self.nodes.add(node_id)
            self._node_edgesets[node_id] = set()
        else:
            self.nodes.add(node_id)
            self._node_edgesets[node_id] = set()

    
    def add_nodes(self, node_set):
        assert isinstance(node_set, Iterable), 'Parameter `node_set` must be Iterable.'

        for node_id in node_set:
            self.add_node(node_id)

    
    def remove_node(self, node_id):
        assert node_id in self.nodes, \
            '`node_id` is not in hypergraph.'

        for edge_id in self._node_edgesets[node_id]:
            self.edges[edge_id].remove(node_id)

        del self._node_edgesets[node_id]
        self.nodes.remove(node_id)


    def add_edge(self, node_set, edge_label=None):
        assert isinstance(node_set, Iterable), \
            'Parameter `node_set` must be Iterable.'

        ## Get the edge label/ID.
        if edge_label is None:
            edge_label = self.n_edges
        while edge_label in self.edges:
            edge_label += 1

        ## Add the edge to each node's list of edges.
        for node in node_set:
            self._node_edgesets[node].add(edge_label)

        self.edges[edge_label] = set(node_set)
        self.n_edges += 1


    def remove_edge(self, edge_id):
        for node in self.edges[edge_id]:
            self._node_edgesets.remove(edge_id)

        del self.edges[edge_id]


    def reset_node_ids(self):
        node_id = 0
        mapping = {}
        for node in self.nodes:
            mapping[node] = node_id
            node_id += 1

        translate = lambda edge, mapping: set([mapping[node] for node in edge])
        for edge_id in self.edges:
            self.edges[edge_id] = translate(self.edges[edge_id], mapping)

        self.nodes = set(mapping.values())


    def reset_edge_ids(self):
        edge_id = 0
        new_edges = {}
        for edge in self.edges.values():
            new_edges[edge_id] = edge
            edge_id += 1
        self.edges = new_edges


    def delete_node_and_incident_edges(self, node_id):
        assert node_id in self.nodes, 'Parameter `node_id` must be in the hypergraph.'

        for edge_id in self._node_edgesets[node_id]:
            self.remove_edge(edge_id)
        self.remove_node(node_id)


    def degree(self, node_id):
        return len(self._node_edgesets[node_id])


if __name__ == '__main__':
    hg = hypergraph()
    for i in range(10):
        hg.add_node()

    print(f'hg._node_edgesets -> {hg._node_edgesets}')

    hg.remove_node(3)

    print(f'hg.nodes -> {hg.nodes}')
    print(f'hg.edges -> {hg.edges}')
