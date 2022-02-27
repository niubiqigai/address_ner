# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:50
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import os

ADDRESS_NER_DATA_PATH = 'data/address_ner/'
MODEL_VERSION = 'v1_0'  # 控制版本路径

MODEL_PATH = f'data/address_ner/output/{MODEL_VERSION}/model'
WORD_TO_ID_PATH = os.path.join(MODEL_PATH, 'word2id.json')
CHECK_POINT_PATH = os.path.join(MODEL_PATH, 'checkpoints/')
SHORT_TO_LONG_CITY_PATH = os.path.join(ADDRESS_NER_DATA_PATH, 'postcode/short2long_city.json')
