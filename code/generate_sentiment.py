from snownlp import SnowNLP
import os

dir_path = '../datasets/zhihu/data.txt'

with open(os.path.join("..", "datasets", "zhihu", "sentiments.txt"), "w", encoding='utf-8') as f:
    with open(dir_path, encoding='utf-8') as f1:
        data = f1.readline()
        while data != '':
            s = SnowNLP(data)
            f.write(str(s.sentiments) + '\n')
            data = f1.readline()
