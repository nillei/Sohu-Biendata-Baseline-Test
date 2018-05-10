# -*- coding:utf-8 -*-

"""
__file__

    longText_test.py

__description__

    本文件用来生成长文本的文本分类模型

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
import utils
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import tools


def text2word(input_data):
    # 分词

    data = input_data
    # thu1.cut_f分词

    '''
    # 本分词方法速度较慢，建议使用thu1.cut_f
    text_all = []
    for i in range(data.shape[0]):
        data1 = data.iloc[i,1]
        text1 = thu1.cut(data1, text=True)
        text_all.append(text1)
    '''

    if os.path.exists(out_data_path + 'text_all_sep1.txt'):
        pass
    else:
        text_all = []
        for i in range(data.shape[0]):
            data1 = data.iloc[i, 1]
            text_all.append(data1)

        f = open(out_data_path + 'text_all1.txt', 'w')
        for data in text_all:
            f.write(data.decode('utf-8'))
            f.write('\n')
        f.close()
        thu1 = thulac.thulac(seg_only=True)
        thu1.cut_f(out_data_path + "text_all1.txt", out_data_path + "text_all_sep1.txt")

    f1 = open('text_all_sep1.txt')
    text_all_sep = f1.readlines()

    return text_all_sep


def feature_select(sep_txt_name, input_x, input_y, num=3000):
    # 利用卡方对特征进行选择

    data = input_x
    data_num = data[0].values

    Y = np.array(input_y[1].values)
    Y[Y == 2]=1

    # 对分词结果进行分割
    f1 = open(sep_txt_name)
    text_all_sep = f1.readlines()
    text_all_sep_train = []
    for n in data_num:
        text_all_sep_train.append(text_all_sep[int(n[1:])-1])

    vectorizer = CountVectorizer(stop_words=list_stop)
    X = vectorizer.fit_transform(text_all_sep_train)
    word = vectorizer.get_feature_names()
    X[X > 0] = 1
    chi2_model = SelectKBest(chi2, k=num)
    X1 = chi2_model.fit_transform(X, Y)

    chi_score = pd.DataFrame({'word': word, 'score': chi2_model.scores_})
    chi_score1 = chi_score.sort_values(by='score', ascending=False)
    chi_score1.to_csv(out_data_path + 'chi_score.txt', index=None)
    chi_score_select = chi_score1.head(num)

    word_list = []
    list0 = chi_score_select['word'].values
    for i in range(num):
        word_list.append((list0[i], i))
    word_dict = dict(word_list)

    return word_dict


def word2vec(input_data, word_dict):

    data = input_data
    data_num = data[0].values
    # 对分词结果进行分割
    f1 = open(out_data_path + 'text_all_sep.txt')
    text_all_sep = f1.readlines()
    text_all_sep_train = []
    for n in data_num:
        text_all_sep_train.append(text_all_sep[int(n[1:])-1])

    vectorizer = CountVectorizer(stop_words=list_stop, vocabulary=word_dict)
    X = vectorizer.fit_transform(text_all_sep_train)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    word = vectorizer.get_feature_names()
    '''
    for wor in word:
        print wor
    '''
    # weight = tfidf.toarray()
    # 利用SparseDataFrame将稀疏矩阵转为DF
    weight_df = pd.SparseDataFrame(tfidf, index=data_num)
    return weight_df


if __name__ == "__main__":

    reload(sys)
    sys.setdefaultencoding('utf-8')

    # 导入数据位置
    in_data_path = '../data/'
    # 输出数据位置
    out_data_path = '../output/'

    # label数据清洗
    # data_test_x = utils.label_filter(in_data_path+'News_pic_label_train.txt')

    # 导入停用词
    list_stop = utils.generate_stop(in_data_path + 'stop_words.txt')

    # 读取训练集
    data_x = pd.read_table(in_data_path + 'News_info_train_filter.txt', header=None)

    # 读取标注集
    data_y = pd.read_table(in_data_path + 'News_pic_label_train_filter.txt', header=None)

    # 读取测评集
    # data_test_x = pd.read_table(in_data_path + 'News_info_validate_filter.txt', header=None)

    # 分割训练集和测试集
    data_x1 = data_x.sample(frac=0.7)

    data_x_train = data_x1.sort_values(by=0, ascending=True)
    data_x_test = data_x[~data_x[0].isin(list(data_x_train[0]))]

    data_y_train = data_y[data_x[0].isin(list(data_x_train[0]))]
    data_y_test = data_y[~data_x[0].isin(list(data_x_train[0]))]

    # 卡方统计特征选择
    feature_word_dict = feature_select(out_data_path + 'text_all_sep.txt', data_x_train, data_y_train, num=10000)

    # 生成训练集的词向量
    weight_train = word2vec(data_x_train, feature_word_dict)
    X_train = weight_train.fillna(0)
    # 生成训练集label
    Y_train = np.array(data_y_train[1].values)
    Y_train[Y_train == 2] = 1
    # 生成测试集的词向量
    weight_test = word2vec(data_x_test, feature_word_dict)
    X_test = weight_test.fillna(0)
    # 生成测试集label
    Y_test = np.array(data_y_test[1].values)
    Y_test[Y_test == 2] = 1

    # 进行逻辑回归
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    proba_test = lr.predict_proba(X_test)[:, 1]

    # 二值化
    submit_list = []
    for val in proba_test:
        if val >= 0.5:
            submit_list.append(2)
        else:
            submit_list.append(0)

    # 输出结果
    df = pd.DataFrame({"instanceID": data_x_test[0], "proba": submit_list})
    df['text'] = 'NULL'
    df['pic'] = 'NULL'
    df1 = df.sort_values(by=['instanceID'])
    df1.to_csv(out_data_path+"my_label.txt", index=False, header=False, sep='\t')

    # 输出比率
    df = pd.DataFrame({"instanceID": data_x_test[0], "proba": proba_test})
    df['text'] = 'NULL'
    df['pic'] = 'NULL'
    df1 = df.sort_values(by=['instanceID'])
    df1.to_csv(out_data_path+"my_proba.txt", index=False, header=False, sep='\t')

    # 生成测评文件
    data_y_test1 = data_y_test.fillna('NULL')
    data_y_test1.to_csv(out_data_path + "news_pic_label_test.txt", index=False, header=False, sep='\t')

    data_x_test.to_csv(out_data_path + "news_pic_test.txt", index=False, header=False, sep='\t')

    # 数据评分
    ret, res = tools.eval_file(out_data_path + "news_pic_test.txt", out_data_path + "news_pic_label_test.txt", out_data_path+"my_label.txt")
    if ret != 0:
        sys.exit(ret)
    else:
        print 'f_measure:\t%.6f\nrecall:\t\t%.6f\nprecision:\t%.6f' % \
              (res[0], res[1], res[2])
