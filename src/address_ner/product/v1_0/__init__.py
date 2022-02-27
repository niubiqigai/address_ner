# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:47
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import os
import re

from utils.log import logging

from .data import text_clean
from .main import get_config, test, train
from .utils import fix_province_city_county, get_area_code, get_entity

logger = logging.getLogger()


def train_model(temp_folder, train_data_files, **kwargs):
    """
    train_data_files_path：训练集文件路径
    return：模型所在文件夹路径

    """
    training_data_file = os.path.join(temp_folder, 'training_data.txt')
    testing_data_file = os.path.join(temp_folder, 'testing_data.txt')
    output_path = os.path.join(temp_folder, 'output')

    open(training_data_file, 'w').close()
    open(testing_data_file, 'w').close()
    os.makedirs(output_path, exist_ok=True)

    logger.debug(train_data_files)

    test_source = []

    (
        arg, embeddings, tag2label, word2id,
        paths, config, train_data, test_data
    ) = get_config(
        train_data_files, test_source,
        training_data_file, testing_data_file, 'temp',
        output_path=output_path, run_type='train'
    )
    logger.debug(paths)
    train(arg, embeddings, tag2label, word2id,
          paths, config, train_data, test_data)

    return {
        'model_path': paths['model_folder'],
        'other_data_path': paths['output_path']
    }


def test_model(temp_folder, model_folder, test_data_files, **kwargs):
    training_data_file = os.path.join(temp_folder, 'training_data.txt')
    testing_data_file = os.path.join(temp_folder, 'testing_data.txt')

    open(training_data_file, 'w')
    open(testing_data_file, 'w')

    (
        args, embeddings, tag2label,
        word2id, paths, config, test_data
    ) = get_config(
        [], test_data_files, training_data_file,
        testing_data_file, 'temp',
        output_path=temp_folder, run_type='test',
        model_folder=model_folder
    )
    logger.debug(paths)

    metric = test(args, embeddings, tag2label,
                  word2id, paths, config, test_data)

    accuracy = float(metric[0].split(',')[0].split(':')[1])

    return {
        'accuracy': accuracy
    }


def predict(text):
    from . import init_model

    # reg_sj = re.compile('1[0-9]{10}')
    # reg_bsj = re.compile('1\d{2}[ |-]\d{4}[ |-]\d{4}')
    # reg_zj = re.compile(
    #     '((?:010|021|022|023|852|853|0310|0311|0312|0313|0314|0315|0316|0317|0318|0319|0335|0570|0571|0572|0573|0574|0575|0576|0577|0578|0579|0580|024|0410|0411|0412|0413|0414|0415|0416|0417|0418|0419|0421|0427|0429|027|0710|0711|0712|0713|0714|0715|0716|0717|0718|0719|0722|0724|0728|025|0510|0511|0512|0513|0514|0515|0516|0517|0517|0518|0519|0523|0470|0471|0472|0473|0474|0475|0476|0477|0478|0479|0482|0483|0790|0791|0792|0793|0794|0795|0796|0797|0798|0799|0701|0350|0351|0352|0353|0354|0355|0356|0357|0358|0359|0930|0931|0932|0933|0934|0935|0936|0937|0938|0941|0943|0530|0531|0532|0533|0534|0535|0536|0537|0538|0539|0450|0451|0452|0453|0454|0455|0456|0457|0458|0459|0591|0592|0593|0594|0595|0595|0596|0597|0598|0599|020|0751|0752|0753|0754|0755|0756|0757|0758|0759|0760|0762|0763|0765|0766|0768|0769|0660|0661|0662|0663|028|0810|0811|0812|0813|0814|0816|0817|0818|0819|0825|0826|0827|0830|0831|0832|0833|0834|0835|0836|0837|0838|0839|0840|0730|0731|0732|0733|0734|0735|0736|0737|0738|0739|0743|0744|0745|0746|0370|0371|0372|0373|0374|0375|0376|0377|0378|0379|0391|0392|0393|0394|0395|0396|0398|0870|0871|0872|0873|0874|0875|0876|0877|0878|0879|0691|0692|0881|0883|0886|0887|0888|0550|0551|0552|0553|0554|0555|0556|0557|0558|0559|0561|0562|0563|0564|0565|0566|0951|0952|0953|0954|0431|0432|0433|0434|0435|0436|0437|0438|0439|0440|0770|0771|0772|0773|0774|0775|0776|0777|0778|0779|0851|0852|0853|0854|0855|0856|0857|0858|0859|029|0910|0911|0912|0913|0914|0915|0916|0917|0919|0971|0972|0973|0974|0975|0976|0977|0890|0898|0899|0891|0892|0893)[ -][0-9]{7,8})')
    reg_pc = re.compile('(?<![0-9])([0-9]{6})(?![0-9])')

    # # 电话使用正则获取
    # phone, phone_b, tel = None, None, None
    #
    # # 匹配手机号
    # phones = reg_sj.findall(text)
    # if len(phones) > 0:
    #     idx = text.index(phones[0])
    #     text = text[:idx] + ',' + text[idx + + len(phones[0]):]
    #     phone = phones[0]
    #
    # # 匹配含有空格或-的手机号
    # phones_b = reg_bsj.findall(text)
    # if len(phones_b) > 0:
    #     idx = text.index(phones_b[0])
    #     text = text[:idx] + ',' + text[idx + + len(phones_b[0]):]
    #     # 替换空格或-
    #     phone_b = re.sub('[- ]', '', phones_b[0])
    #
    # # 匹配固定电话
    # phones = reg_zj.findall(text)
    # if len(phones) > 0:
    #     idx = text.index(phones[0])
    #     text = text[:idx] + ',' + text[idx + len(phones[0]):]
    #     tel = phones[0]
    # if phone:
    #     telphone = phone
    # elif phone_b:
    #     telphone = phone_b
    # elif tel:
    #     telphone = tel
    # else:
    #     telphone = ''

    # 预处理
    text_ = text_clean(text)
    if len(text_) == 0:
        return {
            'name': '',
            'province': '',
            'city': '',
            'district': '',
            'area': '',
            'phone': '',
            'postcode': ''
        }

    text_ = list(text_.strip())
    demo_data = [(text_, ['O'] * len(text_))]
    tag = init_model.model.demo_one(init_model.sess, demo_data)
    if tag.count('B-A') == 2:
        # 如果截取到多个area,将第二个B-A的替换为I-A
        second_b_a_tag_index = tag[tag.index(
            'B-A') + 1:].index('B-A') + tag.index('B-A') + 1
        tag[second_b_a_tag_index] = 'I-A'

    province, city, county, area, name, phone = get_entity(tag, text_)
    # 如果城市出现简称，即城市不是由区县反推而来（邮寄信息本身含城市简称，包含两种情况：1.没写区县2.同一个区县映射到不同城市），将简称转为全称
    if city in init_model.short2long_city.keys():
        city = init_model.short2long_city[city]

    # postcode = get_postcode(province, city, district)
    # 2020年9月22日：需要行政编码，不是邮政编码
    postcode = get_area_code(province, city, county)
    province, city, county = fix_province_city_county(province, city, county)

    # 邮编使用正则获取
    if len(postcode) == 0:
        postcode = reg_pc.findall(text)
        if len(postcode) > 0:
            idx = text.index(postcode[0])
            text = text[:idx] + ',' + text[idx + len(postcode[0]):]
            postcode = postcode[0]
        else:
            postcode = ''

    return {
        'name': name,
        'province': province,
        'city': city,
        'county': county,
        'area': area.replace(',', ' ').strip(),
        'phone': phone.replace(',', '').strip(),
        'postcode': postcode
    }
