import networkx as nx

print('>>> Normalizing edgelist for the Amazon topology... ', end='')
g = nx.read_edgelist('com-amazon.ungraph.txt', create_using=nx.Graph())
g = nx.relabel.convert_node_labels_to_integers(g)
nx.write_edgelist(g, 'amazon-graph.txt', data=False)
print('Done!')


print('>>> Initializing node mapping Amazon community normalization... ', end='')
amazon_map = {}
i = 0
with open('com-amazon.ungraph.txt', 'r') as f:
    lines = f.readlines()
    original_nodes = set()
    for line in lines:
        if line[0] == '#':
            continue
        u, v = line.split()
        original_nodes.add(int(u))
        original_nodes.add(int(v))
        
    original_nodes = sorted(list(original_nodes))
    amazon_map = {original_nodes[i]: i for i in range(len(original_nodes))}

with open('com-amazon.top5000.cmty.txt', 'r') as f:
    lines = f.readlines()
    text = []
    for line in lines:
        string = ''
        for node in line.split():
            string += str(amazon_map[int(node)]) + ' '
        string += '\n'
        text.append(string)
    out = open('amazon-cmty.txt', 'w')
    out.writelines(text)
print('Done!')
    

print('>>> Normalizing edgelist for the DBLP topology... ', end='')
g = nx.read_edgelist('com-dblp.ungraph.txt', create_using=nx.Graph())
g = nx.relabel.convert_node_labels_to_integers(g)
nx.write_edgelist(g, 'dblp-graph.txt', data=False)
print('Done!')


print('>>> Initializing node mapping DBLP community normalization... ', end='')
dblp_map = {}
i = 0
with open('com-dblp.ungraph.txt', 'r') as f:
    lines = f.readlines()
    original_nodes = set()
    for line in lines:
        if line[0] == '#':
            continue
        u, v = line.split()
        original_nodes.add(int(u))
        original_nodes.add(int(v))
        
    original_nodes = sorted(list(original_nodes))
    dblp_map = {original_nodes[i]: i for i in range(len(original_nodes))}

with open('com-dblp.top5000.cmty.txt', 'r') as f:
    lines = f.readlines()
    text = []
    for line in lines:
        string = ''
        for node in line.split():
            string += str(dblp_map[int(node)]) + ' '
        string += '\n'
        text.append(string)
    out = open('dblp-cmty.txt', 'w')
    out.writelines(text)
print('Done!')
