# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:35
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
from enum import Enum


class APP(str, Enum):
    ADDRESS_NER = 'address_ner'


APP_BLUEPRINT = {
    # app_tag:('module_name','blueprint_name','url_prefix')
    APP.ADDRESS_NER: ('route.address_ner', 'address_ner_blueprint', '/base/address')
}
