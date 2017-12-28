import random
graph_path = '../datasets/zhihu/graph.txt'
f = open(graph_path,'r')
edges = [i for i in f]
selected = random.sample(edges, int(len(edges)*0.15))
remain = [i for i in edges if i not in selected]
fw1 = open('graph.txt','w')
fw2 = open('test_graph.txt','w')

for i in selected:
	fw1.write(i)
for i in remain:
	fw2.write(i)