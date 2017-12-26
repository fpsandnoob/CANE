import sys
from imp import reload

reload(sys)
import config

# sys.setdefaultencoding("utf-8")
import numpy as np
from tensorflow.contrib import learn
from negativeSample import InitNegTable
import random


class dataSet:
    def __init__(self, text_path, graph_path):

        text_file, graph_file = self.load(text_path, graph_path)

        self.edges = self.load_edges(graph_file)

        self.text, self.num_vocab, self.num_nodes = self.load_text(text_file)

        self.negative_table = InitNegTable(self.edges)

    def load(self, text_path, graph_path):
        text_file = open(text_path, 'rb').readlines()
        graph_file = open(graph_path, 'rb').readlines()

        return text_file, graph_file

    def load_edges(self, graph_file):
        edges = []
        for i in graph_file:
            edges.append(list(map(int, str(i.strip(), encoding='utf-8').replace("\r\n", "").split('\t'))))

        return edges

    def load_text(self, text_file):
        vocab = learn.preprocessing.VocabularyProcessor(config.MAX_LEN)
        text = np.array(list(vocab.fit_transform(list(map(str, text_file)))))
        num_vocab = len(vocab.vocabulary_)
        num_nodes = len(text)

        return text, num_vocab, num_nodes

    def negative_sample(self, edges):
        node1, node2 = zip(*edges)
        sample_edges = []
        func = lambda: self.negative_table[random.randint(0, config.neg_table_size - 1)]
        for i in range(len(edges)):
            neg_node = func()
            while node1[i] == neg_node or node2[i] == neg_node:
                neg_node = func()
            sample_edges.append([node1[i], node2[i], neg_node])

        return sample_edges

    def generate_batches(self, mode=None):

        num_batch = len(self.edges) / config.batch_size
        edges = self.edges
        if mode == 'add':
            num_batch += 1
            edges.extend(edges[:(config.batch_size - len(self.edges) // config.batch_size)])
        if mode != 'add':
            random.shuffle(edges)
        sample_edges = edges[:int(num_batch * config.batch_size)]
        sample_edges = self.negative_sample(sample_edges)

        batches = []
        for i in range(int(num_batch)):
            batches.append(sample_edges[i * config.batch_size:(i + 1) * config.batch_size])
        # print sample_edges[0]
        return batches

if __name__ == '__main__':
    graph_path = '../datasets/{}/graph.txt'.format("zhihu")
    text_path = '../datasets/{}/data.txt'.format("zhihu")
    c = dataSet(text_path, graph_path)
    print(c.load_edges(graph_path))
