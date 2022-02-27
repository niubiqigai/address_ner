# -*- coding: utf-8 -*-
"""
    @Author zbh
    @email pypy17z@126.com
    @Version 1.0.0
      创建日期:2022/2/27 17:50
      修改记录
      修改后版本:     修改人：   修改日期:    修改内容:
"""
import numpy as np

from .utils import get_entity_by_tag


def conlleval_origin(label_predict, label_path):
    """
    测试结果评估
    :param label_predict:预测的标签
    :param label_path:标签路径
    :return: 元组格式的统计结果
    """
    result = []

    np_tag = []
    np_tag_ = []

    with open(label_path, "w", encoding='utf-8') as fw:
        line = []
        for sent_result in label_predict:
            # 字符，真实标签，预测标签
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                line.append("{} {} {}\n".format(char, tag, tag_))
                np_tag.append(tag)
                np_tag_.append(tag_)
            line.append("\n")
        fw.writelines(line)

    # 准确率、精确率、召回率统计
    tag_len = len(np_tag)
    np_tag = np.asarray(np_tag)
    np_tag_ = np.asarray(np_tag_)
    all_acc = np.sum((np_tag == np_tag_) != 0) / tag_len

    np_recall = np_tag[np_tag != '0']
    np_recall_ = np_tag_[np_tag != '0']
    all_recall = np.sum((np_recall == np_recall_) != 0) / len(np_recall)

    np_precision_ = np_tag_[np_tag_ != '0']
    np_precision = np_tag[np_tag_ != '0']
    all_presision = np.sum((np_precision == np_precision_) != 0) / len(np_precision)

    f1 = 0 if (all_recall + all_presision) == 0 else 2 * all_recall * all_presision / (all_recall + all_presision)
    res = 'accuracy:{:.4},recall:{:.4},precision:{:.4},F1:{:.4f}'.format(all_acc, all_recall, all_presision, f1)
    result.append(res)

    # 每个tag的准确率、召回率、精确率
    for item in ['R', 'P', 'C', 'D', 'A']:
        recall_cnt = 0
        pre_correct_cnt = 0
        for idx, tag in enumerate(np_tag):
            if item in tag:
                recall_cnt += 1
                if tag == np_tag_[idx]:
                    pre_correct_cnt += 1
        precision = 0 if recall_cnt == 0 else round(pre_correct_cnt / recall_cnt, 4)

        recall_cnt = 0
        recall_correct_cnt = 0
        for idx, tag in enumerate(np_tag_):
            if item in tag:
                recall_cnt += 1
                if tag == np_tag[idx]:
                    recall_correct_cnt += 1
        recall = 0 if recall_cnt == 0 else round(recall_correct_cnt / recall_cnt, 4)

        f1 = 0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)

        result.append('{},recall:{:.4f},precision:{:.4f},F1:{:.4f}'.format(item, recall, precision, f1))

    return result


def conlleval(label_predict, label_path):
    """
    测试结果评估
    :param label_predict:预测的标签
    :param label_path:标签路径
    :return: 元组格式的统计结果
    """
    result = []

    np_tag = []
    np_tag_ = []

    with open(label_path, "w", encoding='utf-8') as fw:
        line = []
        for sent_result in label_predict:
            # 字符，真实标签，预测标签
            for char, tag, tag_ in sent_result:
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)

    for sent_result in label_predict:
        s_char_seq = []
        s_np_tag = []
        s_np_tag_ = []
        for char, tag, tag_ in sent_result:
            s_char_seq.append(char)
            s_np_tag.append(tag)
            s_np_tag_.append(tag_)
        # 真实实体
        province = get_entity_by_tag(s_np_tag, s_char_seq, 'B-P', 'I-P')
        city = get_entity_by_tag(s_np_tag, s_char_seq, 'B-C', 'I-C')
        district = get_entity_by_tag(s_np_tag, s_char_seq, 'B-D', 'I-D')
        area = get_entity_by_tag(s_np_tag, s_char_seq, 'B-A', 'I-A')
        name = get_entity_by_tag(s_np_tag, s_char_seq, 'B-R', 'I-R')
        phone = get_entity_by_tag(s_np_tag, s_char_seq, 'B-T', 'I-T')
        np_tag.extend([province, city, district, area, name, phone])
        # 预测实体
        province_pred = get_entity_by_tag(s_np_tag_, s_char_seq, 'B-P', 'I-P')
        city_pred = get_entity_by_tag(s_np_tag_, s_char_seq, 'B-C', 'I-C')
        district_pred = get_entity_by_tag(s_np_tag_, s_char_seq, 'B-D', 'I-D')
        area_pred = get_entity_by_tag(s_np_tag_, s_char_seq, 'B-A', 'I-A')
        name_pred = get_entity_by_tag(s_np_tag_, s_char_seq, 'B-R', 'I-R')
        phone_pred = get_entity_by_tag(s_np_tag_, s_char_seq, 'B-T', 'I-T')
        np_tag_.extend([province_pred, city_pred, district_pred, area_pred, name_pred, phone_pred])

    # 准确率、精确率、召回率统计
    np_tag = np.asarray(np_tag)
    np_tag_ = np.asarray(np_tag_)
    all_acc = np.sum((np_tag == np_tag_) != 0) / len(np_tag)
    # 召回率在本项目中的定义：真实实体有值，预测有值且相等
    recall_correct_cnt = []
    for idx, tag in enumerate(np_tag):
        if tag:
            if tag == np_tag_[idx]:
                recall_correct_cnt.append(1)
            else:
                recall_correct_cnt.append(0)
    all_recall = 0 if not recall_correct_cnt else sum(recall_correct_cnt) / len(recall_correct_cnt)
    # 精确率在本项目中的定义：预测实体有值，真实实体有值且相等
    pre_correct_cnt = []
    for idx, tag in enumerate(np_tag_):
        if tag:
            if tag == np_tag[idx]:
                pre_correct_cnt.append(1)
            else:
                pre_correct_cnt.append(0)
    all_presision = 0 if not pre_correct_cnt else sum(pre_correct_cnt) / len(pre_correct_cnt)

    f1 = 0 if (all_recall + all_presision) == 0 else 2 * all_recall * all_presision / (all_recall + all_presision)
    res = 'accuracy:{:.4},recall:{:.4},precision:{:.4},F1:{:.4f}'.format(float(all_acc), float(all_recall), float(all_presision), f1)
    result.append(res)

    # 每个tag的准确率、召回率、精确率
    for i, item in enumerate(['P', 'C', 'D', 'A', 'R', 'T']):
        sub_np_tag = np_tag[i::6]
        sub_np_tag_ = np_tag_[i::6]
        recall_correct_cnt = []
        for idx, tag in enumerate(sub_np_tag):
            if tag:
                if tag == sub_np_tag_[idx]:
                    recall_correct_cnt.append(1)
                else:
                    recall_correct_cnt.append(0)
        recall = 0 if not recall_correct_cnt else sum(recall_correct_cnt) / len(recall_correct_cnt)

        pre_correct_cnt = []
        for idx, tag in enumerate(sub_np_tag_):
            if tag:
                if tag == sub_np_tag[idx]:
                    pre_correct_cnt.append(1)
                else:
                    pre_correct_cnt.append(0)

        precision = 0 if not pre_correct_cnt else sum(pre_correct_cnt) / len(pre_correct_cnt)

        f1 = 0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)
        res = '{},recall:{:.4f},precision:{:.4f},F1:{:.4f}'.format(item, float(recall), float(precision), f1)
        result.append(res)

    return result
