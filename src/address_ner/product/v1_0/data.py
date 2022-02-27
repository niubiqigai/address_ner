# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:50
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""

import json
import random
import re

import numpy as np
import pandas as pd

"""
字母标签与数字标签的转换
O:无关字符
"B-P": 省份开头字符（如广东省的“广”）
"I-P": 省份除开头外的其余字符（如广东省的“东”“省”）
"B-C": 城市开头字符
"I-C": 城市除开头外的其余字符
"B-D": 地区开头字符
"I-D": 地区除开头外的其余字符
"B-A": 详细地址开头字符
"I-A": 详细地址除开头外的其余字符
"B-R": 人名开头字符（如马冬梅的“马”） 
"I-R": 人名除开头外的其余字符（如马冬梅的“冬”“梅”）
"B-T": 电话号码开头字符
"I-T": 电话号码除开头外的其余字符
"""
# tags, BIO
tag2label = {"O": 0,
             "B-P": 1, "I-P": 2,
             "B-C": 3, "I-C": 4,
             "B-D": 5, "I-D": 6,
             "B-A": 7, "I-A": 8,
             "B-R": 9, "I-R": 10,
             "B-T": 11, "I-T": 12
             }


def read_corpus(corpus_path):
    """
    根据路径读取语料
    :param corpus_path:语料路径
    :return: 返回数组格式的语料
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for idx, line in enumerate(lines):
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """
    根据语料路径、读取语料，构建字符2id的词典
    :param vocab_path:保存路径
    :param corpus_path:语料路径
    :param min_count:字符出现的最小次数（小于该值丢弃）
    :return:无返回，文件保存于vocab_path
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            # 对数字、英文做归一处理
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id) + 1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    with open(vocab_path, 'w', encoding='utf-8') as fw:
        json.dump(word2id, fw, ensure_ascii=False, indent=4)
        fw.write('\n')


def sentence2id(sent, word2id):
    """
    部分特殊字符进项转换，如英文统一转成<ENG>
    :param sent:句子
    :param word2id:字与id的映射词典
    :return:转换后的句子
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
    根据词典路径读取词典
    :param vocab_path: 词典路径
    :return:词典
    """
    word2id = json.load(open(vocab_path, 'r', encoding='utf-8'))
    # print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """
    生成随机嵌入层，纬度值最小-0.25，最大0.25
    :param vocab:字容量大小
    :param embedding_dim:嵌入层维度
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    句子补全（补0）
    :param sequences:句子数组
    :param pad_mark:补全的符号
    :return:补全后的句子数组、句子长度
    """
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """
    根据batch_size获取每个批次的数据集
    :param data:全量数据
    :param batch_size:每个批次大小
    :param vocab:字符词典
    :param tag2label:标签词典
    :param shuffle:是否随机打乱
    :return: 每个批次数量的数据集
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def get_train_data_2(path_list, save_path):
    """
    根据以行标注的语料，转换为列式标注的语料
    :param path_list: 语料路径数组
    :param save_path: 保存路径
    :return: 无返回，存于文本中
    """
    result = []
    for path in path_list:
        data = open(path, 'r', encoding='utf-8').readlines()
        for line in data:
            # 广州市C\天河区D\中新镇恒大山水城15-23A\ 电话 O\13076812865T\ O\谢家棚R\
            line = text_clean(line)
            for item in line.split('\\'):
                if len(item) == 0:
                    continue
                tag = item[-1]
                if tag == 'O':
                    for idx in range(len(item) - 1):
                        cha = item[idx]
                        result.append(cha + ' ' + 'O')
                else:
                    result.append(item[0] + ' B-' + tag)
                    for idx in range(1, len(item) - 1):
                        cha = item[idx]
                        result.append(cha + ' I-' + tag)
            result.append('\n')

    with open(save_path, 'w', encoding='utf-8') as fr:
        for item in result:
            if item != '\n':
                item = item + '\n'
            fr.write(item)


def text_clean(text):
    """
    文本清洗，训练集、测试集、接口访问都需要经过这个步骤
    :param text: 文本
    :return: 清洗后的文本
    """
    text = str(text).replace('\n', ' ').strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub('[,.，。 /:：]', ',', text)
    text = re.sub(',+', ',', text)
    return text


def generate_corpus():
    """
    使用省市区+详细地址信息+随机电话号码+人名，生成训练集
    :return:
    """
    county_list = json.load(open('../../data/address_ner/postcode/areas.json', 'r', encoding='utf-8'))
    city_list = json.load(open('../../data/address_ner/postcode/cities.json', 'r', encoding='utf-8'))
    province_dict = json.load(open('../../data/address_ner/postcode/province.json', 'r', encoding='utf-8'))
    area_list = open('../../data/address_ner/original/custom_corpus/area.txt', 'r', encoding='utf-8').readlines()
    name_list = open('../../data/address_ner/original/custom_corpus/name.txt', 'r', encoding='utf-8').readlines()

    city_dict = {}
    for city_dict_ in city_list:
        city_dict[city_dict_['code']] = city_dict_['name']  # {'1101':['北京市，北京']}

    result = []
    test_result = []
    train_result = []
    for idx, county_dict in enumerate(county_list):
        # county = county_dict['name']  # 区全称,区简称
        counties = county_dict['name']  # [区全称, 区简称]
        cities = city_dict[county_dict['cityCode']]  # [区所属城市全称，区所属城市简称]
        # [区所属省份全称，区所属省份简称]
        provinces = province_dict[county_dict['provinceCode']]
        rank = 0
        for county in counties:
            for city in cities:
                for province in provinces:
                    # print(province, city, county)
                    area = random.choice(area_list).strip()
                    name = name_list[idx].strip()
                    phone = create_phone()
                    line = ''
                    if idx % 2 == 0:
                        line = '{}R\{}T\ O\{}P\{}C\{}D\{}A\\'.format(
                            name, phone, province, city, county, area)
                        # result.append(line)
                    elif idx % 2 == 1:
                        line = '{}P\{}C\{}D\{}A\ O\{}R\{}T\\'.format(
                            province, city, county, area, name, phone)
                    # 处理特殊城市
                    if 'P\县C\\' in line:
                        line = line.replace('P\县C\\', 'P\\')
                    elif '省直辖县级行政区划C\\' in line:
                        line = line.replace("省直辖县级行政区划C\\", '')
                    elif '自治区直辖县级行政区划C\\' in line:
                        line = line.replace('自治区直辖县级行政区划C\\', '')
                    # 处理特殊省份-直辖市
                    # zxs_list = ['北京P\北京市C\\', '北京P\北京C\\', '北京市P\北京市C\\', '北京市P\北京C\\',
                    #             '天津P\天津市C\\', '天津P\天津C\\', '天津市P\天津市C\\', '天津市P\天津C\\',
                    #             '重庆P\重庆市C\\', '重庆P\重庆C\\', '重庆市P\重庆市C\\', '重庆市P\重庆C\\',
                    #             '上海P\上海市C\\', '上海P\上海C\\', '上海市P\上海市C\\', '上海市P\上海C\\']
                    zxs_list = ['北京', '天津', '上海', '重庆',
                                '北京市', '天津市', '上海市', '重庆市']

                    for zxs in zxs_list:
                        if zxs in line:
                            if 'C\\' in line:
                                line = line.replace("%sP\\" % zxs, '')
                            else:
                                line = line.replace(
                                    "%sP\\" % zxs, "%sC\\" % zxs)

                    # 根据直辖市出现的次数决定删除P\还是替换为C\
                    for zxs in zxs_list:
                        if zxs in line:
                            line = line.replace("%sP\\" % zxs, '')

                    result.append(line)
                    rank += 1
                    if rank == 1:
                        test_result.append(line)
                    else:
                        train_result.append(line)
                    if rank == len(counties) * len(cities) * len(provinces):
                        rank = 0

    with open('../../data/address_ner/original/train_data.txt', 'w', encoding='utf8') as train_f:
        for line in train_result:
            train_f.write(line + '\n')
    with open('../../data/address_ner/original/test_data.txt', 'w', encoding='utf8') as test_f:
        for line in test_result:
            test_f.write(line + '\n')

    print('corpus generate ok~~~')


def create_phone():
    """
    创建随机电话号码
    :return:
    """
    # 第二位数字
    second = [3, 4, 5, 7, 8][random.randint(0, 4)]

    # 第三位数字
    third = {
        3: random.randint(0, 9),
        4: [5, 7, 9][random.randint(0, 2)],
        5: [i for i in range(10) if i != 4][random.randint(0, 8)],
        7: [i for i in range(10) if i not in [4, 9]][random.randint(0, 7)],
        8: random.randint(0, 9),
    }[second]

    # 最后八位数字
    suffix = random.randint(9999999, 100000000)

    # 拼接手机号
    return "1{}{}{}".format(second, third, suffix)


if __name__ == '__main__':
    # get_train_data_2(['../../data/address_ner/original/train_80.txt'], '../../data/address_ner/train_80.txt')
    # get_train_data_2(['../../data/address_ner/original/test_20.txt'], '../../data/address_ner/test_20.txt')
    # vocab_build('../../data/address_ner/word2id.json','../../data/address_ner/train_80.txt', 0)
    # 使用省市区+详细地址信息+随机电话号码+人名，生成训练集
    generate_corpus()
