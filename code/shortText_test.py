# -*- coding:utf-8 -*-

"""
__file__

    shortText_test.py

__description__

    本文件用来对短文本进行分类

__author__

    Nil Lei <https://github.com/nillei>

"""

import thulac
import pandas as pd
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import utils


def info_cut(input_data):
    # 对info的filter数据进行处理，生成短文本文件

    text_data = input_data.iloc[:, 1].str.split(' ', expand=True)
    text_num_series = text_data.count(axis=1)
    text_data['joint'] = text_data.apply(lambda x: x.str.cat(sep=' '), axis=1)
    joint1 = text_data.apply(lambda x: x.str.cat(sep=' '), axis=0)
    text_data_joint = joint1['joint'].decode('utf-8')
    text_data_joint_list = text_data_joint.split(' ')

    f = open('test.txt', 'w')
    for data in text_data_joint_list:
        f.write(data)
        f.write('\n')
    f.close()

    return text_data_joint_list, text_num_series


def generate_label(label_list, text_list, text_num_series):
    # 生成短文本分类特征


    # 分0,1,2三种情况进行处理

    label_list_index = []

    for label1 in label_list:
        if label1 in text_list:
            label_list_index.append(text_list.index(label1))
        else:
            print label1, 'not in'

    


def text2vec(input_data):
    # 短文本转成向量

    data = input_data
    data_num = data[0].values

    # thu1.cut_f分词
    text_all = []
    for i in range(data.shape[0]):
        data1 = data.iloc[i, 1]
        text_all.append(data1)
    f = open('text_all.txt', 'w')
    for data in text_all:
        f.write(data.decode('utf-8'))
        f.write('\n')
    f.close()
    thu1 = thulac.thulac(seg_only=True)
    thu1.cut_f("text_all.txt", "text_all_sep.txt")
    f1 = open('output.txt')
    text_all_sep = f1.readlines()

    # 保存分词的文件
    output = open('fenci_0.pkl', 'wb')
    pickle.dump(text_all_sep, output)
    output.close()

    '''
    # 读取分词的文件
    pkl_file = open('fenci.pkl', 'rb')
    text_all = pickle.load(pkl_file)
    '''

    # 输出词向量
    vectorizer = CountVectorizer(max_df=2000, min_df=4, stop_words=list_stop)
    X = vectorizer.fit_transform(text_all_sep)

    # TF-IDF
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()
    weight_df = pd.DataFrame(weight, index=data_num)

    return weight_df

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')

    # 导入数据位置
    in_data_path = '../data/'
    # 输出数据位置
    out_data_path = '../output/'
    # 导入停用词
    list_stop = utils.generate_stop(in_data_path + 'stop_words.txt')

    # 读取训练集，并将长文本切片
    info = pd.read_table(in_data_path + 'News_info_train_filter.txt', header=None)
    text_list, count_series = info_cut(info)

    # 读取标签集，并filter
    label_df = utils.label_filter(in_data_path + 'News_pic_label_train_example100_filter.txt')
    list_label = label_df[label_df.text != 'NULL'].text.values
    list_label_1 = []
    for label in list_label:
        list_label_1 += label.split(' ')


