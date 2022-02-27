# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:51
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import logging
import argparse
import os
import json

current_path = os.path.abspath(os.path.dirname(__file__))
ftsp_area_code = json.load(
    open(os.path.join(current_path, '../../../../data/address_ner/postcode/ftsp_area_code.json'), 'r', encoding='utf8'))


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    province = get_entity_by_tag(tag_seq, char_seq, 'B-P', 'I-P')
    city = get_entity_by_tag(tag_seq, char_seq, 'B-C', 'I-C')
    district = get_entity_by_tag(tag_seq, char_seq, 'B-D', 'I-D')
    area = get_entity_by_tag(tag_seq, char_seq, 'B-A', 'I-A')
    name = get_entity_by_tag(tag_seq, char_seq, 'B-R', 'I-R')
    phone = get_entity_by_tag(tag_seq, char_seq, 'B-T', 'I-T')
    return province, city, district, area, name, phone


def get_entity_by_tag(tag_seq, char_seq, start_tag, end_tag):
    entity = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        # 如果省市区长度大于8，去掉重新取
        if start_tag in ['B-P', 'B-C', 'B-D'] and tag not in [start_tag, end_tag] and len(entity) > 8:
            entity = []
        if tag == start_tag:
            # 如果检测到开始标签，判断是否已有检测到的实体，如果已匹配到第一个实体，则break跳出循环，将第一个作为最终预测结果。
            if len(entity) == 0:
                # 如果尚未到识别到实体，判断开始标签start_tag的下一个标签是不是end_tag，是则作为实体预测结果的开始标签字符。
                # 即排除只有单个开始标签的字符作为实体，剔除对第二个预测实体的干扰。
                if i + 1 < len(tag_seq) and tag_seq[i + 1] == end_tag:
                    entity.append((char, i))
            else:
                break
        # 如果识别到结束标签，同时具有开始标签，即加入实体的结束标签字符。
        if tag == end_tag and len(entity) > 0:
            entity.append((char, i))
    res = ''.join([char for char, i in entity])
    return res


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def process_ftsp_code():
    """
    根据云服务提供的ftsp_district_data.json行政地区编码文件，转换为易用的词典
    :return:
    """
    area_dict = {}
    ftsp_area_code = json.load(open('../../data/address_ner/postcode/ftsp_district_data.json', 'r', encoding='utf8'))
    province_list = ftsp_area_code['86']
    for code, area_list in ftsp_area_code.items():
        # code，对应value是其下级组成的list，需要找到code对应的中文name
        city = ''
        for pro_dict in province_list:
            pro_code = pro_dict['code']
            pro_name = pro_dict['name']
            # 台湾、香港、澳门 跳过
            if pro_code in ['710000', '810000', '820000']:
                continue
            # 先判断key是否为省份
            if code == pro_code:
                city = pro_name
                continue
            else:
                city_list = ftsp_area_code[pro_code]
                for city_dict in city_list:
                    city_code = city_dict['code']
                    city_name = city_dict['name']
                    # key是区域代码，找到对应的城市代码关联出城市中文
                    if city_code == code:
                        city = city_name
                        continue

        # 将key对应的城市名写入dict，value为上一级单位{name:[code]}
        for area in area_list:
            name = area['name']
            code = area['code']
            if name not in area_dict:
                area_dict[name] = {city: code}
            else:
                area_dict[name][city] = code
    json.dump(area_dict, open('../../data/address_ner/postcode/ftsp_area_code.json', 'w', encoding='utf8'),
              ensure_ascii=False)


def process_ftsp_area_code():
    """
    由包含省市区全称和简称的省、市、区json文件，生成区县反推城市的ftsp_area_code.json文件
    :return:
    """
    county_list = json.load(open('../../data/address_ner/postcode/areas.json', 'r', encoding='utf-8'))
    city_list = json.load(open('../../data/address_ner/postcode/cities.json', 'r', encoding='utf-8'))
    province_dict = json.load(open('../../data/address_ner/postcode/province.json', 'r', encoding='utf-8'))
    dict_ = {}
    # 城市 -> 省份 "广州市": {"广东省": "4401"}
    # for city in city_list:
    #     for provCode, prov in province_dict.items():
    #         if city['provinceCode'] == provCode:
    #             for i in range(len(city['name'])):
    #                 if not dict_.get(city['name'][i]):
    #                     dict_[city['name'][i]] = {}
    #                 dict_[city['name'][i]][prov[0]] = city['code']
    # 区县 -> 城市 "天河区": {"广州市": "440106"}
    for county in county_list:
        for city in city_list:
            if county['cityCode'] == city['code']:
                for i in range(len(county['name'])):
                    if not dict_.get(county['name'][i]):
                        dict_[county['name'][i]] = {}
                    dict_[county['name'][i]][city['name'][0]] = county['code']

    # json.dump(dict_, open('../../data/address_ner/postcode/ftsp_area_code.json', 'w', encoding='utf8'),
    #           ensure_ascii=False)


def get_area_code(province, city, county):
    """
    获取行政区域编码
    :param province:省份
    :param city: 城市
    :param county: 区县
    :return: 6位数字的编码
    """
    res = ''
    area_code_dict = ftsp_area_code.get(county, {})
    if len(area_code_dict) == 1:
        res = list(area_code_dict.values())[0]
    elif len(area_code_dict) > 1:  # 当区县对应两个城市时，查询该区县下对应城市的编码
        res = area_code_dict.get(city, '')
    if len(res) == 0:  # 当没有查到区县编码时，查询城市对应省份编码
        area_code_dict = ftsp_area_code.get(city, {})
        if len(area_code_dict) == 1:
            res = list(area_code_dict.values())[0]
        elif len(area_code_dict) > 1:  # 当城市对应两个省份时，查询该城市下对应省份的编码
            res = area_code_dict.get(province, '')
    # if len(res) == 0:
    #     res = ftsp_area_code.get(city, '')
    # if len(res) == 0:
    #     res = ftsp_area_code.get(province, '')
    return res


def fix_province_city_county(province, city, county):
    """
    省市区字段修复
    :param province: 省份
    :param city: 城市
    :param county: 区县
    :return: 修复后的省市区
    """
    # 城市修复
    area_code_dict = ftsp_area_code.get(county, {})
    if len(area_code_dict) == 1:
        city_ = list(area_code_dict.keys())[0]
        if len(city) < len(city_):
            city = city_
    # 省份修复
    city_code_dict = ftsp_area_code.get(city, {})
    if len(city_code_dict) == 1:
        province_ = list(city_code_dict.keys())[0]
        if len(province) < len(province_):
            province = province_

    return province, city, county


if __name__ == '__main__':
    # process_ftsp_code()
    process_ftsp_area_code()

    pass
