# -*- coding:utf-8 -*-

"""
__file__

    utils.py

__description__

    用到的函数

__author__

    Nil Lei <https://github.com/nillei>

"""
import pandas as pd
import os
import thulac


def generate_stop(text):
    # 导入停用词

    f = open(text)
    lines = f.readlines()
    f.close()
    re_list = []
    for raw_line in lines:
        line = raw_line.rstrip('\r\n')
        re_list.append(line.decode('gbk'))
    return re_list


def label_filter(input_data):
    # label数据清洗，并输出清洗后的结果到filter文件中

    f = open(input_data)
    lines = f.readlines()
    f.close()

    label_list = []
    label_text = []
    label_pic = []
    label_num = []

    for line in lines:
        data = line.split('\t')
        label_num.append(data[0])
        label_pic.append(data[2])
        text = ' '.join(data[3:])
        label_text.append(text.strip('\n'))
        label_list.append(data[1])

    label_df = pd.DataFrame({'ID': label_num, 'label': label_list, 'pic': label_pic, 'text': label_text})
    name = input_data[:-4] + '_filter' + '.txt'
    label_df.to_csv(name, header=None, sep='\t', index=None)

    return label_df


def text2word(input_data, sep_txt_name, org_txt_name):
    # 分词

    data = input_data
    # thu1.cut_f分词
    if os.path.exists(sep_txt_name):
        pass
    else:
        text_all = []
        for i in range(data.shape[0]):
            data1 = data.iloc[i, 1]
            text_all.append(data1)

        f = open(org_txt_name, 'w')
        for data in text_all:
            f.write(data.decode('utf-8'))
            f.write('\n')
        f.close()
        thu1 = thulac.thulac(seg_only=True)
        thu1.cut_f(org_txt_name, sep_txt_name)

    f1 = open(sep_txt_name)
    text_all_sep = f1.readlines()

    return text_all_sep

