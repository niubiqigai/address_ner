# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:51
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""

import os
import sys
import threading
import time

import tensorflow as tf
# from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
# from tensorflow.contrib.rnn import LSTMCell
# tensorflow==1.3.0 transform to tensorflow==2.3.0
from tensorflow_addons.text import crf_log_likelihood, viterbi_decode
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

from utils.log import logging

from .data import batch_yield, pad_sequences
from .eval import conlleval

logger = logging.getLogger(__name__)


class BiLSTM_CRF(object):
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.graph = tf.Graph()
        self.batch_size = args['batch_size']
        self.epoch_num = args['epoch']
        self.hidden_dim = args['hidden_dim']
        self.embeddings = embeddings
        self.CRF = args['CRF']
        self.update_embedding = args['update_embedding']
        self.dropout_keep_prob = args['dropout']
        self.optimizer = args['optimizer']
        self.lr = args['lr']
        self.clip_grad = args['clip']
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args['shuffle']
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        # self.logger = get_logger(paths['log_path'])
        self.logger = logger
        self.result_path = paths['result_path']
        self.config = config
        self.lock = threading.Lock()

    def build_graph(self):
        """
        构建TensorFlow网络结构
        :return:
        """
        with self.graph.as_default():
            self.add_placeholders()
            self.lookup_layer_op()
            self.biLSTM_layer_op()
            self.softmax_pred_op()
            self.loss_op()
            self.trainstep_op()
            self.init_op()

    def add_placeholders(self):
        """
        使用tf的占位符，初始化5个属性
        :return:
        """
        self.word_ids = tf.compat.v1.placeholder(
            tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.compat.v1.placeholder(
            tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.compat.v1.placeholder(
            tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.compat.v1.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.compat.v1.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, 1 - (self.dropout_pl))

    def biLSTM_layer_op(self):
        with tf.compat.v1.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, 1 - (self.dropout_pl))

        with tf.compat.v1.variable_scope("proj"):
            W = tf.compat.v1.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                dtype=tf.float32)

            b = tf.compat.v1.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.compat.v1.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(input=output)
            output = tf.reshape(output, [-1, 2 * self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(input_tensor=log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(tensor=losses, mask=mask)
            self.loss = tf.reduce_mean(input_tensor=losses)

        tf.compat.v1.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(input=self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.compat.v1.variable_scope("train_step"):
            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.compat.v1.train.MomentumOptimizer(
                    learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.compat.v1.train.GradientDescentOptimizer(
                    learning_rate=self.lr_pl)
            else:
                optim = tf.compat.v1.train.GradientDescentOptimizer(
                    learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(
                g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(
                grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        """变量初始化"""
        self.init_op = tf.compat.v1.global_variables_initializer()

    def add_summary(self, sess):
        """
        打印概览
        :param sess:
        :return:
        """
        self.merged = tf.compat.v1.summary.merge_all()
        self.file_writer = tf.compat.v1.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        """
        训练入口
        :param train:训练集
        :param dev: 验证集
        :return:
        """

        with self.graph.as_default():
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)

        feed_data_list = self.get_feed_dict_for_epoch(train)

        with tf.compat.v1.Session(config=self.config, graph=self.graph) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(
                    sess, feed_data_list, dev, epoch, saver)

    def test(self, test):
        """
        测试入口
        :param test:测试集
        :return:
        """
        with self.graph.as_default():
            saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session(config=self.config, graph=self.graph) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            return self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """
        单独测试1个句子
        :param sess:session
        :param sent: 句子数组
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, feed_data_list, dev, epoch, saver):
        """
        训练1个opoch
        :param sess:session
        :param train:训练集
        :param dev:验证集
        :param tag2label:标签转数字
        :param epoch:当前epoch
        :param saver:tf模型保存对象
        :return:
        """
        feed_data_len = len(feed_data_list)

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        for index, feed_dict in enumerate(feed_data_list, start=1):
            # self.logger.debug(
            #     f' processing: {index} batch / {feed_data_len} batches.')

            step_num = epoch * feed_data_len + index

            _, loss_train, summary, _ = sess.run(
                [self.train_op, self.loss, self.merged, self.global_step], feed_dict=feed_dict)
            if index % 300 == 1 or index == feed_data_len:
                self.logger.debug(
                    f'{start_time} epoch {epoch}, step {index}, loss: {loss_train}, global_step: {step_num}'
                )
            self.file_writer.add_summary(summary, step_num)

        saver.save(sess, self.model_path, global_step=step_num)

        self.logger.debug('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """
        获取tf网络模型需要“喂”的数据
        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def get_feed_dict_for_epoch(self, train_data):
        result = []

        batches = batch_yield(train_data, self.batch_size,
                              self.vocab, self.tag2label, shuffle=self.shuffle)
        for _, (seqs, labels) in enumerate(batches):

            feed_dict, _ = self.get_feed_dict(
                seqs, labels, self.lr, self.dropout_keep_prob)
            result.append(feed_dict)
        return result

    def dev_one_epoch(self, sess, dev):
        """
        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """
        批量预测
        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)
        self.lock.acquire()
        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            self.lock.release()
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(
                    logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            self.lock.release()
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """
        测试结果评估
        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            # {0: 0, 1: 'B-P', 2: 'I-P', 3: 'B-C', 4: 'I-C', 5: 'B-D', 6: 'I-D', 7: 'B-A', 8: 'I-A', 9: 'B-R', 10: 'I-R', 11: 'B-T', 12: 'I-T'}
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        # <class 'list'>: [0, 0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]..
        for label_, (words, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            # <class 'list'>: ['广', 'B-C', 'I-A']... 字，真实标注，预测标注
            sent_res = []
            # if len(label_) != len(words):
            #     print(words)
            #     print(len(label_))
            #     print(tag)
            for i in range(len(words)):
                sent_res.append([words[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch + 1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric = conlleval(model_predict, label_path)
        for line in metric:
            self.logger.info(line)
        return metric


def get_logger(filename):
    """
    配置logger
    :param filename:
    :return:
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
