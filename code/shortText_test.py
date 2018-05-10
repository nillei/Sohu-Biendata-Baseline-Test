# -*- coding:utf-8 -*-

"""
__file__

    shortText_test.py

__description__

    本文件用来对短文本进行分类

__author__

    Nil Lei <https://github.com/nillei>

"""

import pandas as pd
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import utils
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def info_cut(input_data, sep_txt_name, org_txt_name):
    # 对info的filter数据进行处理，生成短文本文件

    text_data = input_data.iloc[:, 1].str.split(' ', expand=True)
    text_num_series = text_data.count(axis=1)
    text_data['joint'] = text_data.apply(lambda x: x.str.cat(sep=' '), axis=1)
    joint1 = text_data.apply(lambda x: x.str.cat(sep=' '), axis=0)
    text_data_joint = joint1['joint'].decode('utf-8')
    text_data_joint_list = text_data_joint.split(' ')

    f = open(org_txt_name, 'w')
    for data in text_data_joint_list:
        f.write(data)
        f.write('\n')
    f.close()

    num_list = []
    for ind in text_num_series.index:
        for i in range(text_num_series[ind]):
            num_list.append(ind+1)

    c = {0: num_list,
        1: text_data_joint_list}
    text_data_joint_df = pd.DataFrame(c)

    text_all_sep = utils.text2word(text_data_joint_df, sep_txt_name, org_txt_name)

    return text_data_joint_df, text_all_sep


def feature_select(sep_txt_name, input_y, num=3000):
    # 利用卡方对特征进行选择

    Y = input_y

    # 对分词结果进行分割
    f1 = open(sep_txt_name)
    text_all_sep = f1.readlines()

    vectorizer = CountVectorizer(stop_words=list_stop)
    X = vectorizer.fit_transform(text_all_sep)
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


def generate_short_label(train_y_df, train_text_df):
    # 生成短文本分类特征

    # 读取标签集，并生成list
    label_df = train_y_df
    label_df.columns = ['ID','label','pic','text']
    list_label = label_df[~label_df.text.isnull()].text.values
    text_label_list = []
    for label in list_label:
        text_label_list += label.split(' ')

    text_list = list(train_text_df[1])

    label_list = list(label_df['label'])
    ID_list = []
    for lab in list(label_df['ID']):
        ID_list.append(int(lab[1:]))
    c = {"ID": ID_list,
         "label": label_list}
    ID_label_df = pd.DataFrame(c)
    ID_label_df['ID'].astype('str')
    ID_label_df['label'].astype('str')

    res = pd.merge(train_text_df, ID_label_df, left_on=0, right_on='ID', how='left')

    # 分0,1,2三种情况进行处理
    for (text,j) in zip(text_list,range(len(text_list))):
        if text in text_label_list:
            res.iloc[j, 3] = 2
    short_label = list(res.label)

    return short_label


def word2vec(input_data_name, word_dict):

    f1 = open(input_data_name)
    text_all_sep = f1.readlines()

    vectorizer = CountVectorizer(stop_words=list_stop, vocabulary=word_dict)
    X = vectorizer.fit_transform(text_all_sep)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    word = vectorizer.get_feature_names()
    '''
    for wor in word:
        print wor
    '''
    # weight = tfidf.toarray()
    # 利用SparseDataFrame将稀疏矩阵转为DF
    weight_df = pd.SparseDataFrame(tfidf)
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

    # 读取训练集，并将长文本切片
    data_x = pd.read_table(in_data_path + 'News_info_train_example100_filter.txt', header=None)
    # 读取标注集
    data_y = pd.read_table(in_data_path + 'News_pic_label_train_example100_filter.txt', header=None)
    # 读取测评集
    # data_test_x = pd.read_table(in_data_path + 'News_info_validate_filter.txt', header=None)

    # 分割训练集和测试集
    data_x1 = data_x.sample(frac=0.7)

    data_x_train = data_x1.sort_values(by=0, ascending=True)
    data_x_test = data_x[~data_x[0].isin(list(data_x_train[0]))]
    data_y_train = data_y[data_x[0].isin(list(data_x_train[0]))]
    data_y_test = data_y[~data_x[0].isin(list(data_x_train[0]))]

    # 对文本进行切分,生成短文本及其编号
    train_text_df, train_sep = info_cut(data_x_train, 'train_sep.txt', 'train_original.txt')
    test_text_df, test_sep = info_cut(data_x_test, 'test_sep.txt', 'test_original.txt')

    # 生成训练集短文本短label
    short_label_train = generate_short_label(data_y_train, train_text_df)
    Y_train = np.array(short_label_train)
    Y_train[Y_train == 1] = 0

    # 生成测试集短文本短label
    short_label_test = generate_short_label(data_y_test, test_text_df)
    Y_test = np.array(short_label_test)
    Y_test[Y_test == 1] = 0

    # 卡方统计特征选择
    feature_word_dict = feature_select('train_sep.txt', Y_train, num=30)

    # 生成训练集的词向量
    weight_train = word2vec('train_sep.txt', feature_word_dict)
    X_train = weight_train.fillna(0)

    # 生成测试集的词向量
    weight_test = word2vec('test_sep.txt', feature_word_dict)
    X_test = weight_test.fillna(0)

    # 进行逻辑回归
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    proba_test = lr.predict_proba(X_test)[:, 1]

    # 生成短文本DF
    c = {"text": test_sep,
         "proba": proba_test}
    final_df = pd.DataFrame(c)

    final_df.to_csv(out_data_path + "short_text_results.txt", index=False, header=False, sep='\t')
    
