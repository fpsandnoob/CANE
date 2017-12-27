import random
import numpy as np

node2vec = {}
f = open('embed.txt', 'rb')
for i, j in enumerate(f):
    if str(j, encoding='utf-8') != '\n':
        node2vec[i] = list(map(float, str(j, encoding='utf-8').strip().split(' ')))
# f1 = open('test_graph.txt', 'rb')
edges = []
graph_file = open("../datasets/{}/graph.txt".format("zhihu"), "rb").readlines()
for i in graph_file:
    edges.append(list(map(int, str(i.strip(), encoding='utf-8').replace("\r\n", "").split('\t'))))
nodes = list(set([i for j in edges for i in j]))
a = 0
b = 0
for i, j in edges:
    if i in node2vec.keys() and j in node2vec.keys():
        dot1 = np.dot(node2vec[i], node2vec[j])
        random_node = random.sample(nodes, 1)[0]
        while random_node == j or random_node not in node2vec.keys():
            random_node = random.sample(nodes, 1)[0]
        dot2 = np.dot(node2vec[i], node2vec[random_node])
        if dot1 > dot2:
            a += 1
        elif dot1 == dot2:
            a += 0.5
        b += 1

print(float(a) / b)
