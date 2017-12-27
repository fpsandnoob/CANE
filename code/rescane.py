import tensorflow as tf
from tensorflow.contrib import layers
import config


class Modelv1:
    def __init__(self, vocab_size, num_nodes):
        # '''hyperparameter'''
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Ta')
            self.Text_b = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tb')
            self.Text_neg = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tneg')
            self.Node_a = tf.placeholder(tf.int32, [config.batch_size], name='n1')
            self.Node_b = tf.placeholder(tf.int32, [config.batch_size], name='n2')
            self.Node_neg = tf.placeholder(tf.int32, [config.batch_size], name='n3')

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, int(config.embed_size / 2)], stddev=0.3))
            self.node_embed = tf.Variable(tf.truncated_normal([num_nodes, int(config.embed_size / 2)], stddev=0.3))
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            self.T_A = tf.expand_dims(self.TA, -1)

            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)

            self.TNEG = tf.nn.embedding_lookup(self.text_embed, self.Text_neg)
            self.T_NEG = tf.expand_dims(self.TNEG, -1)

            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.N_NEG = tf.nn.embedding_lookup(self.node_embed, self.Node_neg)
        self.convA, self.convB, self.convNeg = self.newconv()
        self.loss = self.compute_loss()

    def newconv(self):
        weights = {"w1_1": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 1, 100], stddev=0.3)),
                   "w2_1": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 100, 100], stddev=0.3)),
                   "w1_2": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 1, 100], stddev=0.3)),
                   "w2_2": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 100, 100], stddev=0.3))}
        rand_matrix = tf.Variable(tf.truncated_normal([100, 100], stddev=0.3))

        conv1_1_A = tf.nn.conv2d(self.T_A, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_1_B = tf.nn.conv2d(self.T_B, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_1_NEG = tf.nn.conv2d(self.T_NEG, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')

        conv1_2_A = tf.nn.conv2d(self.T_A, weights['w1_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_2_B = tf.nn.conv2d(self.T_B, weights['w1_2'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_2_NEG = tf.nn.conv2d(self.T_NEG, weights['w1_2'], strides=[1, 1, 1, 1], padding='SAME')

        conv1_A_activate = tf.multiply(conv1_1_A, tf.sigmoid(conv1_2_A))
        conv1_B_activate = tf.multiply(conv1_1_B, tf.sigmoid(conv1_2_B))
        conv1_NEG_activate = tf.multiply(conv1_1_NEG, tf.sigmoid(conv1_2_NEG))

        conv2_1_A = tf.nn.conv2d(conv1_A_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_1_B = tf.nn.conv2d(conv1_B_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_1_NEG = tf.nn.conv2d(conv1_NEG_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='VALID')

        conv2_2_A = tf.nn.conv2d(conv1_A_activate, weights['w2_2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_2_B = tf.nn.conv2d(conv1_B_activate, weights['w2_2'], strides=[1, 1, 1, 1], padding='VALID')
        conv2_2_NEG = tf.nn.conv2d(conv1_NEG_activate, weights['w2_2'], strides=[1, 1, 1, 1], padding='VALID')

        conv2_A_activate = tf.multiply(conv2_1_A, tf.sigmoid(conv2_2_A))
        conv2_B_activate = tf.multiply(conv2_1_B, tf.sigmoid(conv2_2_B))
        conv2_NEG_activate = tf.multiply(conv2_1_NEG, tf.sigmoid(conv2_2_NEG))

        hA = tf.nn.softmax(tf.squeeze(conv2_A_activate))
        hB = tf.nn.softmax(tf.squeeze(conv2_B_activate))
        hNEG = tf.nn.softmax(tf.squeeze(conv2_NEG_activate))

        tmphA = tf.reshape(hA, [config.batch_size * (config.MAX_LEN - 1), int(config.embed_size / 2)])
        ha_mul_rand = tf.reshape(tf.matmul(tmphA, rand_matrix),
                                 [config.batch_size, config.MAX_LEN - 1, int(config.embed_size / 2)])
        r1 = tf.matmul(ha_mul_rand, hB, transpose_a=False, transpose_b=True)
        r3 = tf.matmul(ha_mul_rand, hNEG, transpose_a=False, transpose_b=True)
        att1 = tf.expand_dims(tf.stack(r1), -1)
        att3 = tf.expand_dims(tf.stack(r3), -1)

        att1 = tf.tanh(att1)
        att3 = tf.tanh(att3)

        pooled_A = tf.reduce_mean(att1, 2)
        pooled_B = tf.reduce_mean(att1, 1)
        pooled_NEG = tf.reduce_mean(att3, 1)

        a_flat = tf.squeeze(pooled_A)
        b_flat = tf.squeeze(pooled_B)
        neg_flat = tf.squeeze(pooled_NEG)

        w_A = tf.nn.softmax(a_flat)
        w_B = tf.nn.softmax(b_flat)
        w_NEG = tf.nn.softmax(neg_flat)

        rep_A = tf.expand_dims(w_A, -1)
        rep_B = tf.expand_dims(w_B, -1)
        rep_NEG = tf.expand_dims(w_NEG, -1)

        hA = tf.transpose(hA, perm=[0, 2, 1])
        hB = tf.transpose(hB, perm=[0, 2, 1])
        hNEG = tf.transpose(hNEG, perm=[0, 2, 1])

        rep1 = tf.matmul(hA, rep_A)
        rep2 = tf.matmul(hB, rep_B)
        rep3 = tf.matmul(hNEG, rep_NEG)

        attA = tf.squeeze(rep1)
        attB = tf.squeeze(rep2)
        attNEG = tf.squeeze(rep3)

        return attA, attB, attNEG

    def compute_loss(self):
        p1 = tf.reduce_sum(tf.multiply(self.convA, self.convB), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(tf.multiply(self.convA, self.convNeg), 1)
        p2 = tf.log(tf.sigmoid(-p2) + 0.001)

        p3 = tf.reduce_sum(tf.multiply(self.N_A, self.N_B), 1)
        p3 = tf.log(tf.sigmoid(p3) + 0.001)

        p4 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG), 1)
        p4 = tf.log(tf.sigmoid(-p4) + 0.001)

        p5 = tf.reduce_sum(tf.multiply(self.convB, self.N_A), 1)
        p5 = tf.log(tf.sigmoid(p5) + 0.001)

        p6 = tf.reduce_sum(tf.multiply(self.convNeg, self.N_A), 1)
        p6 = tf.log(tf.sigmoid(-p6) + 0.001)

        p7 = tf.reduce_sum(tf.multiply(self.N_B, self.convA), 1)
        p7 = tf.log(tf.sigmoid(p7) + 0.001)

        p8 = tf.reduce_sum(tf.multiply(self.N_B, self.convNeg), 1)
        p8 = tf.log(tf.sigmoid(-p8) + 0.001)

        rho1 = 0.7
        rho2 = 1.0
        rho3 = 0.1
        temp_loss = rho1 * (p1 + p2) + rho2 * (p3 + p4) + rho3 * (p5 + p6) + rho3 * (p7 + p8)
        loss = -tf.reduce_sum(temp_loss)
        return loss

class Modelv2:
    def __init__(self, vocab_size, num_nodes):
        # '''hyperparameter'''
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Ta')
            self.Text_b = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tb')
            self.Text_neg = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tneg')
            self.Node_a = tf.placeholder(tf.int32, [config.batch_size], name='n1')
            self.Node_b = tf.placeholder(tf.int32, [config.batch_size], name='n2')
            self.Node_neg = tf.placeholder(tf.int32, [config.batch_size], name='n3')
            self.sentiment_a = tf.placeholder(tf.float32, [config.batch_size], name='Sa')
            self.sentiment_b = tf.placeholder(tf.float32, [config.batch_size], name='Sb')
            self.sentiment_neg = tf.placeholder(tf.float32, [config.batch_size], name='Sneg')

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, int(config.embed_size / 2)], stddev=0.3))
            self.node_embed = tf.Variable(tf.truncated_normal([num_nodes, int(config.embed_size / 2)], stddev=0.3))
            self.sentiment_embed = tf.Variable(tf.truncated_normal([1, 100], stddev=0.3))
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            self.T_A = tf.expand_dims(self.TA, -1)

            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)

            self.TNEG = tf.nn.embedding_lookup(self.text_embed, self.Text_neg)
            self.T_NEG = tf.expand_dims(self.TNEG, -1)

            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.N_NEG = tf.nn.embedding_lookup(self.node_embed, self.Node_neg)

            self.sentiment_a = tf.expand_dims(self.sentiment_a, -1)
            self.sentiment_b = tf.expand_dims(self.sentiment_b, -1)
            self.sentiment_neg = tf.expand_dims(self.sentiment_neg, -1)
            self.S_A = tf.multiply(self.sentiment_a, self.sentiment_embed)
            self.S_B = tf.multiply(self.sentiment_b, self.sentiment_embed)
            self.S_NEG = tf.multiply(self.sentiment_neg, self.sentiment_embed)

        self.convA, self.convB, self.convNeg = self.conv()

        with tf.name_scope('Merge Vector') as scope:
            self.merge_A = tf.concat([self.convA, self.S_A, self.N_A], axis=1)
            self.merge_B = tf.concat([self.convB, self.S_B, self.N_B], axis=1)
            self.merge_NEG = tf.concat([self.convNeg, self.S_NEG, self.N_NEG], axis=1)



        self.loss = self.compute_loss()

    def conv(self):
        weights = {"w1_1": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 1, 32], stddev=0.3)),
                   "w2_1": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 32, 64], stddev=0.3)),
                   "w3_1": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 64, 100], stddev=0.3))}
        # "w1_2": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 1, 100], stddev=0.3)),
        # "w2_2": tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 100, 100], stddev=0.3))}

        conv1_1_A = tf.nn.conv2d(self.T_A, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_1_B = tf.nn.conv2d(self.T_B, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1_1_NEG = tf.nn.conv2d(self.T_NEG, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')

        # conv1_2_A = tf.nn.conv2d(self.T_A, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')
        # conv1_2_B = tf.nn.conv2d(self.T_B, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')
        # conv1_2_NEG = tf.nn.conv2d(self.T_NEG, weights['w1_1'], strides=[1, 1, 1, 1], padding='SAME')

        conv1_A_activate = tf.multiply(conv1_1_A, tf.sigmoid(conv1_1_A))
        conv1_B_activate = tf.multiply(conv1_1_B, tf.sigmoid(conv1_1_B))
        conv1_NEG_activate = tf.multiply(conv1_1_NEG, tf.sigmoid(conv1_1_NEG))

        conv2_1_A = tf.nn.conv2d(conv1_A_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv2_1_B = tf.nn.conv2d(conv1_B_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv2_1_NEG = tf.nn.conv2d(conv1_NEG_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME')

        # conv2_2_A = tf.nn.conv2d(conv1_A_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME')
        # conv2_2_B = tf.nn.conv2d(conv1_B_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME')
        # conv2_2_NEG = tf.nn.conv2d(conv1_NEG_activate, weights['w2_1'], strides=[1, 1, 1, 1], padding='SAME')

        conv2_A_activate = tf.multiply(conv2_1_A, tf.sigmoid(conv2_1_A))
        conv2_B_activate = tf.multiply(conv2_1_B, tf.sigmoid(conv2_1_B))
        conv2_NEG_activate = tf.multiply(conv2_1_NEG, tf.sigmoid(conv2_1_NEG))

        conv3_1_A = tf.nn.conv2d(conv2_A_activate, weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_1_B = tf.nn.conv2d(conv2_B_activate, weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME')
        conv3_1_NEG = tf.nn.conv2d(conv2_NEG_activate, weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME')

        # conv3_1_A = tf.nn.conv2d(conv2_A_activate, weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME')
        # conv3_1_B = tf.nn.conv2d(conv2_B_activate, weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME')
        # conv3_1_NEG = tf.nn.conv2d(conv2_NEG_activate, weights['w3_1'], strides=[1, 1, 1, 1], padding='SAME')

        conv3_A_activate = tf.multiply(conv3_1_A, tf.sigmoid(conv3_1_A))
        conv3_B_activate = tf.multiply(conv3_1_B, tf.sigmoid(conv3_1_B))
        conv3_NEG_activate = tf.multiply(conv3_1_NEG, tf.sigmoid(conv3_1_NEG))

        W2 = tf.Variable(tf.truncated_normal([2, int(config.embed_size / 2), 100, 100], stddev=0.3))
        rand_matrix = tf.Variable(tf.truncated_normal([100, 100], stddev=0.3))

        convA = tf.nn.conv2d(conv3_A_activate, W2, strides=[1, 1, 1, 1], padding='VALID')
        convB = tf.nn.conv2d(conv3_B_activate, W2, strides=[1, 1, 1, 1], padding='VALID')
        convNEG = tf.nn.conv2d(conv3_NEG_activate, W2, strides=[1, 1, 1, 1], padding='VALID')

        """
            :activation Relu
        """
        hA = tf.nn.tanh(tf.squeeze(convA))
        hB = tf.nn.tanh(tf.squeeze(convB))
        hNEG = tf.nn.tanh(tf.squeeze(convNEG))

        # hA = tf.nn.softmax(tf.squeeze(conv2_A_activate))
        # hB = tf.nn.softmax(tf.squeeze(conv2_B_activate))
        # hNEG = tf.nn.softmax(tf.squeeze(conv2_NEG_activate))

        tmphA = tf.reshape(hA, [config.batch_size * (config.MAX_LEN - 1), int(config.embed_size / 2)])
        ha_mul_rand = tf.reshape(tf.matmul(tmphA, rand_matrix),
                                 [config.batch_size, config.MAX_LEN - 1, int(config.embed_size / 2)])
        r1 = tf.matmul(ha_mul_rand, hB, transpose_a=False, transpose_b=True)
        r3 = tf.matmul(ha_mul_rand, hNEG, transpose_a=False, transpose_b=True)
        att1 = tf.expand_dims(tf.stack(r1), -1)
        att3 = tf.expand_dims(tf.stack(r3), -1)

        att1 = tf.tanh(att1)
        att3 = tf.tanh(att3)

        pooled_A = tf.reduce_mean(att1, 2)
        pooled_B = tf.reduce_mean(att1, 1)
        pooled_NEG = tf.reduce_mean(att3, 1)

        a_flat = tf.squeeze(pooled_A)
        b_flat = tf.squeeze(pooled_B)
        neg_flat = tf.squeeze(pooled_NEG)

        w_A = tf.nn.softmax(a_flat)
        w_B = tf.nn.softmax(b_flat)
        w_NEG = tf.nn.softmax(neg_flat)

        rep_A = tf.expand_dims(w_A, -1)
        rep_B = tf.expand_dims(w_B, -1)
        rep_NEG = tf.expand_dims(w_NEG, -1)

        hA = tf.transpose(hA, perm=[0, 2, 1])
        hB = tf.transpose(hB, perm=[0, 2, 1])
        hNEG = tf.transpose(hNEG, perm=[0, 2, 1])

        rep1 = tf.matmul(hA, rep_A)
        rep2 = tf.matmul(hB, rep_B)
        rep3 = tf.matmul(hNEG, rep_NEG)

        attA = tf.squeeze(rep1)
        attB = tf.squeeze(rep2)
        attNEG = tf.squeeze(rep3)

        return attA, attB, attNEG

    def autoencoder(self):
        with tf.name_scope('autoencoder') as scope:
            n_input = 300
            n_hidden_1 = 256
            n_hidden_2 = 128
            n_hidden_3 = 100

            with tf.name_scope('init') as scope:
                weights = {
                    'encoder_w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.3)),
                    'encoder_w2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.3)),
                    'encoder_w3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.3)),
                    'decoder_w1': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], stddev=0.3)),
                    'decoder_w2': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.3)),
                    'decoder_w3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.3))
                }
                biases = {
                    'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.3)),
                    'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.3)),
                    'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.3)),
                    'decoder_b1': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.3)),
                    'decoder_b2': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.3)),
                    'decoder_b3': tf.Variable(tf.truncated_normal([n_input], stddev=0.3))
                }

            def encoder(x: tf.Tensor) -> tf.Tensor:
                with tf.name_scope("encoder") as scope:
                    layer_1 = tf.nn.relu6(tf.nn.xw_plus_b(x, weights['encoder_w1'], biases['encoder_b1']))
                    layer_2 = tf.nn.relu6(tf.nn.xw_plus_b(layer_1, weights['encoder_w2'], biases['encoder_b2']))
                    layer_3 = tf.nn.relu6(tf.nn.xw_plus_b(layer_2, weights['encoder_w3'], biases['encoder_b3']))
                    return layer_3

            def decoder(x: tf.Tensor) -> tf.Tensor:
                with tf.name_scope("decoder") as scope:
                    layer_1 = tf.nn.relu6(tf.nn.xw_plus_b(x, weights['decoder_w1'], biases['decoder_b1']))
                    layer_2 = tf.nn.relu6(tf.nn.xw_plus_b(layer_1, weights['decoder_w2'], biases['decoder_b2']))
                    layer_3 = tf.nn.relu6(tf.nn.xw_plus_b(layer_2, weights['decoder_w3'], biases['decoder_b3']))
                    return layer_3





    def compute_loss(self):
        p1 = tf.reduce_sum(tf.multiply(self.convA, self.convB), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(tf.multiply(self.convA, self.convNeg), 1)
        p2 = tf.log(tf.sigmoid(-p2) + 0.001)

        p3 = tf.reduce_sum(tf.multiply(self.N_A, self.N_B), 1)
        p3 = tf.log(tf.sigmoid(p3) + 0.001)

        p4 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG), 1)
        p4 = tf.log(tf.sigmoid(-p4) + 0.001)

        p5 = tf.reduce_sum(tf.multiply(self.convB, self.N_A), 1)
        p5 = tf.log(tf.sigmoid(p5) + 0.001)

        p6 = tf.reduce_sum(tf.multiply(self.convNeg, self.N_A), 1)
        p6 = tf.log(tf.sigmoid(-p6) + 0.001)

        p7 = tf.reduce_sum(tf.multiply(self.N_B, self.convA), 1)
        p7 = tf.log(tf.sigmoid(p7) + 0.001)

        p8 = tf.reduce_sum(tf.multiply(self.N_B, self.convNeg), 1)
        p8 = tf.log(tf.sigmoid(-p8) + 0.001)

        rho1 = 0.7
        rho2 = 1.0
        rho3 = 0.1
        temp_loss = rho1 * (p1 + p2) + rho2 * (p3 + p4) + rho3 * (p5 + p6) + rho3 * (p7 + p8)
        loss = -tf.reduce_sum(temp_loss)
        return loss


if __name__ == '__main__':
    c = Modelv2(1000, 10)
