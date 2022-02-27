# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:51
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import json
import os

import tensorflow as tf

from . import const
from .data import random_embedding, read_dictionary, tag2label
from .model import BiLSTM_CRF

# Session configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# hyperparameters
args = {}
args['train_data'] = const.ADDRESS_NER_DATA_PATH
args['test_data'] = const.ADDRESS_NER_DATA_PATH
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
args['mode'] = 'demo'

model_path = const.MODEL_PATH
word2id = read_dictionary(const.WORD_TO_ID_PATH)
embeddings = random_embedding(word2id, 300)

paths = {}
ckpt_file = tf.train.latest_checkpoint(const.CHECK_POINT_PATH)
paths['model_path'] = ckpt_file
paths['summary_path'] = None
# paths['log_path'] = 'test.log'
paths['result_path'] = None
model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
model.build_graph()
with model.graph.as_default():
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session(config=config, graph=model.graph)
    saver.restore(sess, ckpt_file)

with open(os.path.join(const.SHORT_TO_LONG_CITY_PATH), encoding='utf8') as f:
    short2long_city = json.load(fp=f)
