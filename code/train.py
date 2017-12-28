import numpy as np
import tensorflow as tf
import sys
import config
import rescane
from DataSet import dataSet
import time


# load data
graph_path = '../datasets/zhihu/graph.txt'
text_path = '../datasets/zhihu/data.txt'
sentiment_path = '../datasets/zhihu/sentiments.txt'

data = dataSet(text_path, graph_path, sentiment_path)
# start session


def normal_train():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            model = rescane.Modelv1(data.num_vocab, data.num_nodes)
            opt = tf.train.AdamOptimizer(config.lr)
            train_op = opt.minimize(model.loss)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # training
            print('start training.......')
            # saver.recover_last_checkpoints("/ckpt/model.ckpt")
            for epoch in range(config.num_epoch):
            #     if epoch % 10 == 0:
            #         saver.save(sess, "/ckpt/model.ckpt")
                loss_epoch = 0
                batches = data.generate_batches()
                num_batch = len(batches)
                # p = progressbar.ProgressBar(max_value=num_batch)
                t1 = time.time()
                for i in range(num_batch):
                    batch = batches[i]

                    node1, node2, node3 = zip(*batch)
                    node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                    text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]
                    # sen1, sen2, sen3 = data.sentiment[node1], data.sentiment[node2], data.sentiment[node3]

                    feed_dict = {
                        model.Text_a: text1,
                        model.Text_b: text2,
                        model.Text_neg: text3,
                        model.Node_a: node1,
                        model.Node_b: node2,
                        model.Node_neg: node3,
                        # model.sentiment_a: sen1,
                        # model.sentiment_b: sen2,
                        # model.sentiment_neg: sen3
                    }

                    # run the graph
                    _, loss_batch = sess.run([train_op, model.loss], feed_dict=feed_dict)
                    sys.stdout.flush()
                    sys.stdout.write("Batch: {}/{}, loss: {}\r".format(i, num_batch, loss_batch))
                    loss_epoch += loss_batch
                t2 = time.time()
                print('epoch: ', epoch + 1, ' loss: ', loss_epoch, 'cost time: ', t2 - t1)

            file = open('embed.txt', 'w')
            batches = data.generate_batches(mode='add')
            num_batch = len(batches)
            embed = [[] for _ in range(data.num_nodes)]
            for i in range(num_batch):
                batch = batches[i]

                node1, node2, node3 = zip(*batch)
                node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]

                feed_dict = {
                    model.Text_a: text1,
                    model.Text_b: text2,
                    model.Text_neg: text3,
                    model.Node_a: node1,
                    model.Node_b: node2,
                    model.Node_neg: node3
                }

                # run the graph
                convA, convB, TA, TB = sess.run([model.convA, model.convB, model.N_A, model.N_B], feed_dict=feed_dict)
                for i in range(config.batch_size):
                    em = list(convA[i]) + list(TA[i])
                    embed[node1[i]].append(em)
                    em = list(convB[i]) + list(TB[i])
                    embed[node2[i]].append(em)
            for i in range(data.num_nodes):
                if embed[i]:
                    # print embed[i]
                    tmp = np.sum(embed[i], axis=0) / len(embed[i])
                    file.write(' '.join(map(str, tmp)) + '\n')
                else:
                    file.write('\n')


def autoencoder_train():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            model = rescane.Modelv4(data.num_vocab, data.num_nodes)
            opt = tf.train.AdamOptimizer(config.lr)
            train_op = opt.minimize(model.loss)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # training
            print('start training.......')
            saver.recover_last_checkpoints("./ckpt/model.ckpt")
            for epoch in range(config.num_epoch):
                if epoch % 10 == 0:
                    saver.save(sess, "./ckpt/model.ckpt")
                loss_epoch = 0
                batches = data.generate_batches()
                num_batch = len(batches)
                # p = progressbar.ProgressBar(max_value=num_batch)
                t1 = time.time()
                for i in range(num_batch):
                    batch = batches[i]

                    node1, node2, node3 = zip(*batch)
                    node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                    text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]
                    sen1, sen2, sen3 = data.sentiment[node1], data.sentiment[node2], data.sentiment[node3]

                    feed_dict = {
                        model.Text_a: text1,
                        model.Text_b: text2,
                        model.Text_neg: text3,
                        model.Node_a: node1,
                        model.Node_b: node2,
                        model.Node_neg: node3,
                        model.sentiment_a: sen1,
                        model.sentiment_b: sen2,
                        model.sentiment_neg: sen3,
                    }

                    # run the graph
                    _, loss_batch = sess.run([train_op, model.loss], feed_dict=feed_dict)
                    sys.stdout.flush()
                    sys.stdout.write("Batch: {}/{}, loss: {}\r".format(i, num_batch, loss_batch))
                    loss_epoch += loss_batch
                t2 = time.time()
                print('epoch: ', epoch + 1, ' loss: ', loss_epoch, 'cost time: ', t2 - t1)
            file = open('embed.txt', 'w')
            batches = data.generate_batches(mode='add')
            num_batch = len(batches)
            embed = [[] for _ in range(data.num_nodes)]
            for i in range(num_batch):
                batch = batches[i]

                node1, node2, node3 = zip(*batch)
                node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]
                sen1, sen2, sen3 = data.sentiment[node1], data.sentiment[node2], data.sentiment[node3]

                feed_dict = {
                    model.Text_a: text1,
                    model.Text_b: text2,
                    model.Text_neg: text3,
                    model.Node_a: node1,
                    model.Node_b: node2,
                    model.Node_neg: node3,
                    model.sentiment_a: sen1,
                    model.sentiment_b: sen2,
                    model.sentiment_neg: sen3
                }

                # run the graph
                convA, convB, TA, TB = sess.run([model.encoder_op_A, model.encoder_op_B, model.N_A, model.N_B],
                                                feed_dict=feed_dict)
                for i in range(config.batch_size):
                    em = list(convA[i]) + list(TA[i])
                    embed[node1[i]].append(em)
                    em = list(convB[i]) + list(TB[i])
                    embed[node2[i]].append(em)
            for i in range(data.num_nodes):
                if embed[i]:
                    # print embed[i]
                    tmp = np.sum(embed[i], axis=0) / len(embed[i])
                    file.write(' '.join(map(str, tmp)) + '\n')
                else:
                    file.write('\n')

def create_auc_src():
    file = open('embed.txt', 'w')
    batches = data.generate_batches(mode='add')
    num_batch = len(batches)
    embed = [[] for _ in range(data.num_nodes)]
    for i in range(num_batch):
        batch = batches[i]

        node1, node2, node3 = zip(*batch)
        node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
        text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]

        feed_dict = {
            model.Text_a: text1,
            model.Text_b: text2,
            model.Text_neg: text3,
            model.Node_a: node1,
            model.Node_b: node2,
            model.Node_neg: node3
        }

        # run the graph
        convA, convB, TA, TB = sess.run([model.convA, model.convB, model.N_A, model.N_B], feed_dict=feed_dict)
        for i in range(config.batch_size):
            em = list(convA[i]) + list(TA[i])
            embed[node1[i]].append(em)
            em = list(convB[i]) + list(TB[i])
            embed[node2[i]].append(em)
    for i in range(data.num_nodes):
        if embed[i]:
            # print embed[i]
            tmp = np.sum(embed[i], axis=0) / len(embed[i])
            file.write(' '.join(map(str, tmp)) + '\n')
        else:
            file.write('\n')


def create_auc_autoencoder():
    file = open('embed.txt', 'w')
    batches = data.generate_batches(mode='add')
    num_batch = len(batches)
    embed = [[] for _ in range(data.num_nodes)]
    for i in range(num_batch):
        batch = batches[i]

        node1, node2, node3 = zip(*batch)
        node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
        text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]
        sen1, sen2, sen3 = data.sentiment[node1], data.sentiment[node2], data.sentiment[node3]

        feed_dict = {
            model.Text_a: text1,
            model.Text_b: text2,
            model.Text_neg: text3,
            model.Node_a: node1,
            model.Node_b: node2,
            model.Node_neg: node3,
            model.sentiment_a: sen1,
            model.sentiment_b: sen2,
            model.sentiment_neg: sen3
        }

        # run the graph
        convA, convB, TA, TB = sess.run([model.encoder_op_A, model.encoder_op_B, model.N_A, model.N_B], feed_dict=feed_dict)
        for i in range(config.batch_size):
            em = list(convA[i]) + list(TA[i])
            embed[node1[i]].append(em)
            em = list(convB[i]) + list(TB[i])
            embed[node2[i]].append(em)
    for i in range(data.num_nodes):
        if embed[i]:
            # print embed[i]
            tmp = np.sum(embed[i], axis=0) / len(embed[i])
            file.write(' '.join(map(str, tmp)) + '\n')
        else:
            file.write('\n')

normal_train()