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

import tensorflow as tf

from utils.log import logging
from . import const
from .data import (get_train_data_2, random_embedding, read_corpus,
                   read_dictionary, tag2label, vocab_build)
from .model import BiLSTM_CRF
from .utils import get_entity

logger = logging.getLogger()

current_path = os.path.dirname(__file__)


def get_config(
        train_source, test_source, train_path,
        test_path, version, output_path=None, run_type='train', model_folder=None
):
    """
    设置参数
    :return: args, embeddings, tag2label, word2id, paths, config,train_data,test_data
            超参数、嵌入层、标签转数字，字符转数字，路径词典，环境配置，训练数据，测试数据，
    """
    # 训练显卡配置
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # 配置超参数
    args = {}
    args = get_hyper_parameter(args)

    # 输出路径设置
    paths = {}
    paths = set_path(paths, run_type, version, output_path=output_path, model_folder=model_folder)
    word2id_path = os.path.join(paths['model_folder'], "word2id.json")

    # 获取训练集、测试集
    train_data = None
    if run_type == 'train':
        get_train_data_2(train_source, train_path)
        get_train_data_2(test_source, test_path)
        train_data = read_corpus(train_path)
        test_data = read_corpus(test_path)
        vocab_build(word2id_path, train_path, 0)
    else:
        get_train_data_2(test_source, test_path)
        test_data = read_corpus(test_path)

    word2id = read_dictionary(word2id_path)
    # 构建嵌入层
    embeddings = random_embedding(word2id, args['embedding_dim'])

    if run_type == 'train':
        return args, embeddings, tag2label, word2id, paths, config, train_data, test_data
    else:
        return args, embeddings, tag2label, word2id, paths, config, test_data


def get_hyper_parameter(args):
    """
    配置超参数
    :param args:
    :return:
    """
    args['batch_size'] = 64
    args['epoch'] = 20
    args['hidden_dim'] = 300
    args['optimizer'] = 'Adam'
    args['CRF'] = True
    args['lr'] = 0.001
    args['clip'] = 5.0
    args['dropout'] = 0.5
    args['update_embedding'] = True
    args['pretrain_embedding'] = 'random'
    args['embedding_dim'] = 300
    args['shuffle'] = True

    return args


def set_path(paths, run_type, version, output_path=None, model_folder=None):
    """
    配置路径
    :param paths: 路径词典
    :param run_type: 运行类型：train/test/demo
    :return: 配置好的path
    """
    if output_path is None:
        output_path = os.path.join('data/address_ner/output', version)
    else:
        output_path = os.path.join(output_path)

    paths['output_path'] = output_path
    os.makedirs(output_path, exist_ok=True)

    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    os.makedirs(summary_path, exist_ok=True)

    paths['model_folder'] = model_folder or os.path.join(output_path, 'model')
    os.makedirs(paths['model_folder'], exist_ok=True)

    model_path = os.path.join(paths['model_folder'], "checkpoints/")
    os.makedirs(model_path, exist_ok=True)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    if run_type == 'train':
        # 用于存储模型文件
        paths['model_path'] = ckpt_prefix
    else:
        # 用于读取模型文件
        paths['model_path'] = model_path

    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    os.makedirs(result_path, exist_ok=True)

    return paths


def train(args, embeddings, tag2label, word2id, paths, config, train_data, test_data):
    model = BiLSTM_CRF(args, embeddings, tag2label,
                       word2id, paths, config=config)
    model.build_graph()

    print("train data: {}".format(len(train_data)))
    model.train(train=train_data, dev=test_data)


def test(args, embeddings, tag2label, word2id, paths, config, test_data):
    ckpt_file = tf.train.latest_checkpoint(paths['model_path'])
    logging.debug(f'ckpt_file:{ckpt_file}')
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label,
                       word2id, paths, config=config)
    model.build_graph()
    logging.debug(f'test data:{len(test_data)}')
    return model.test(test_data)


def demo(args, embeddings, tag2label, word2id, paths, config):
    ckpt_file = tf.train.latest_checkpoint(paths['model_path'])
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label,
                       word2id, paths, config=config)
    model.build_graph()
    # saver = tf.train.Saver()
    with tf.compat.v1.Session(config=config) as sess:
        print('============= demo =============')
        # saver.restore(sess, ckpt_file)
        while (1):
            print('请输入地址文本:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('请输入有效地址！')
                continue
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                province, city, district, area, name = get_entity(
                    tag, demo_sent)
                print('省份: {}\n城市: {}\n地区: {}\n地址：{}\n人名：{}\n'.format(
                    province, city, district, area, name))


if __name__ == '__main__':
    # 准备数据
    train_source = ['data/address_ner/original/train_data.txt']
    test_source = ['data/address_ner/original/test_data.txt']
    train_path = 'data/address_ner/original/train_data_deal.txt'
    test_path = 'data/address_ner/original/test_data_deal.txt'

    # 训练
    (
        args, embeddings, tag2label, word2id,
        paths, config, train_data, test_data
    ) = get_config(
        train_source, test_source,
        train_path, test_path,
        version=const.MODEL_VERSION, run_type='train'
    )
    train(args, embeddings, tag2label, word2id, paths, config, train_data, test_data)

    # # 测试
    # args, embeddings, tag2label, word2id, paths, config, test_data = get_config(train_source, test_source,train_path,test_path, version=const.MODEL_VERSION, run_type='test')
    # test(args, embeddings, tag2label, word2id, paths, config, test_data)

    # demo(args, embeddings, tag2label, word2id, paths, config)
