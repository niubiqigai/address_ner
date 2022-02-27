# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:15
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import os
import environs
from utils.log import logging
from settings.routers import APP

logger = logging.getLogger(__name__)

env = environs.Env()
env.read_env(override=True, verbose=True)

DEBUG = env.bool('DEBUG', False)
FLASK_HOST = env.str('FLASK_HOST', '0.0.0.0')
FLASK_PORT = env.int('FLASK_PORT', 5033)

CUDA_VISIBLE_DEVICES = env.str('CUDA_VISIBLE_DEVICES', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

TF_CPP_MIN_LOG_LEVEL = env.str('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = CUDA_VISIBLE_DEVICES


def get_default_install_app():
    """
    APP.ADDRESS_NER: 邮寄地址智能粘贴
    """
    return [APP.ADDRESS_NER, ]


INSTALL_APP = env.list('INSTALL_APP', get_default_install_app())
