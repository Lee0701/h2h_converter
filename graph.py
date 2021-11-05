
from data_load import load_vocab, load_data
import tensorflow as tf
import tqdm
from hyperparams import Hyperparams as hp
import codecs

class Graph():
    '''Builds a model graph'''
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen), name="hangul_sent")
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen), name="hanja_sent")

            # Sequence lengths
            self.seqlens = tf.reduce_sum(tf.sign(self.x), -1)

            # Embedding
            self.inputs = tf.one_hot(self.x, len(hangul2idx))

            # Network
            cell_fw = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
            cell_bw = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.inputs, self.seqlens, dtype=tf.float32)
            logits = tf.layers.dense(tf.concat(outputs, -1), len(hanja2idx))
            self.preds = tf.to_int32(tf.arg_max(logits, -1))

            ## metric
            hits = tf.to_int32(tf.equal(self.preds, self.y))
            hits *= tf.sign(self.y)

            self.acc = tf.reduce_sum(hits) / tf.reduce_sum(self.seqlens)

            ## Loss and training
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y)
            self.mean_loss = tf.reduce_mean(loss)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(hp.learning_rate)
            self.train_op = optimizer.minimize(self.mean_loss, global_step=self.global_step)

            # Summary
            tf.summary.scalar("mean_loss", self.mean_loss)
            tf.summary.scalar("acc", self.acc)

            self.merged = tf.summary.merge_all()

