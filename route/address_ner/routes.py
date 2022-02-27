from flask import Blueprint, request
from src.address_ner.product.v1_0 import predict as address_predict
from utils.common import get_response
from utils.log import logging

logger = logging.getLogger()
address_ner_blueprint = Blueprint('address_ner', __name__)


@address_ner_blueprint.route('/predict', methods=['get'])
def predict():
    """
    邮寄地址智能粘贴
    :param: GET方法，字段 text: 原始文本内容，格式字符串
    :return: 调用接口返回结果，包括："data", "resp_code", "resp_msg"
            "data": 预测结果，格式字典，包括字段:
                    name: 收件人名称
                    province: 省份
                    city: 城市
                    county: 所在辖区
                    area: 邮寄详细地址
                    phone: 联系人电话
                    postcode: 邮政编码
    """
    text = request.args.get('text', '', type=str).strip()
    logger.info(f"传入参数: text={text}")
    data = address_predict(text)
    logger.info(f"返回结果: {data}")
    return get_response(200, data, msg='ok')
