from flask import Flask, request, jsonify
from flask_cors import CORS
from Bert_sentiment import predict, BertClassifier
import torch
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from tqdm import tqdm
import re
from collections import Counter
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras

database = 'dataset/Weibo.csv'

pd_all = pd.read_csv(database)
comment_count = 6000
frequency_count = 4000
test_size = 0.2
model_list = [MultinomialNB(), SVC(), DecisionTreeClassifier()
    , RandomForestClassifier(), KNeighborsClassifier()]

bert_model2 = BertClassifier(num_labels=2)
bert_model2.load_state_dict(torch.load('model2.pth'))
print('二分类BERT加载完成')
bert_model3 = BertClassifier(num_labels=3)
bert_model3.load_state_dict(torch.load('model3.pth'))
print('三分类BERT加载完成')
bert_model6 = BertClassifier(num_labels=6)
bert_model6.load_state_dict(torch.load('model6.pth'))
print('六分类BERT加载完成')
bert = {}
bert[2] = bert_model2
bert[3] = bert_model3
bert[6] = bert_model6

def init():
    global pd_all
    global model_list
    stop_words = ['的', '是', '了', '和', '有', '在', '不', '与', '及', '同', '另', '但', '或', '又', '等', '而', '之',
                  '与', '被', '从', '往', '得', '以', '为', '是', '于', '就', '而', '但', '和', '又', '只', '同', '所',
                  '每']
    print("-----元素为空计数-------")
    print(pd_all.isna().sum())
    pd_all = pd_all.dropna()
    print("----------------------")
    print('评论数目（总体）：%d' % pd_all.shape[0])
    print('评论数目（正向）：%d' % pd_all[pd_all.label == 1].shape[0])
    print('评论数目（负向）：%d' % pd_all[pd_all.label == 0].shape[0])

    def get_balance_corpus(corpus_size, corpus_pos, corpus_neg):
        sample_size = corpus_size // 2
        pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0] < sample_size), \
                                       corpus_neg.sample(sample_size, replace=corpus_neg.shape[0] < sample_size)])

        print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
        print('评论数目（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 1].shape[0])
        print('评论数目（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 0].shape[0])

        return pd_corpus_balance

    print("-------样本计数---------")
    pd_positive = pd_all[pd_all.label == 1]
    pd_negative = pd_all[pd_all.label == 0]
    ChnSentiCorp_htl_ba_6000 = get_balance_corpus(comment_count, pd_positive, pd_negative)
    print("----------------------")
    print("---------分词----------")
    print(f"stop_words: {stop_words}")
    Y = []  # 原始标签
    X = []  # 原始分词字符串
    yy = []  # 总分词数据串
    for item in tqdm(ChnSentiCorp_htl_ba_6000.itertuples(index=False), "Running"):
        label, text = item[0], item[1]
        try:
            text = re.sub(r"[^a-zA-Z\u4e00-\u9fa5]", "", text)  # 将非英文和中文的其他字符全部删除
        except:
            print(text)
        words = list(jieba.cut(text))  # 使用结巴分词对文本分词
        X.append(words)
        yy += words
        Y.append(label)
    yy = [w for w in yy if w not in stop_words]
    word_freq = Counter(yy)
    print(f'总词汇数计数：{len(word_freq)}')
    global top_words
    # 获取前x个使用频率最高的词
    top_words = [word for word, freq in word_freq.most_common(frequency_count)]
    print("----------------------")
    print("-------创建特征矩阵------")
    feature_matrix = []

    # 将每个文本数据转换为特征向量
    for text in tqdm(X, 'Running'):
        words = text
        # 统计每个词在特征向量中的出现次数
        feature_vector = [words.count(word) for word in top_words]
        feature_matrix.append(feature_vector)
    print("----------------------")
    print("-----划分测试、训练集-----")
    X_train, X_test, Y_train, Y_test = train_test_split(feature_matrix, Y, test_size=test_size)
    print("----------------------")
    print("-----在模型上进行测试-----")
    for model in model_list:
        model.fit(X_train, Y_train)
        print(f'{str(model)} 模型在测试集上的正确率为：{accuracy_score(Y_test, model.predict(X_test))}')
    print("----使用自然神经网络测试---")

    def nn_training(model_list, x1, x2, x3):
        model = keras.Sequential([
            keras.layers.Dense(x1, activation='relu', input_shape=(4000,)),  # 假设有1000个特征
            keras.layers.Dense(x2, activation='relu'),
            keras.layers.Dense(x3, activation='relu'),
            keras.layers.Dense(1)  # 输出层，用于回归问题
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 使用分类交叉熵损失函数和准确率作为指标
        model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0)  # X_train和y_train是训练数据集
        loss, accuracy = model.evaluate(X_test, Y_test)  # X_test和y_test是测试数据集
        print(f"三层神经网络中间层，其中每层神经元数量为{x1}, {x2}, {x3}。测试集上的准确率:{accuracy}")
        model_list.append(model)

    # nn_training(model_list, 128, 256, 256)
    # nn_training(model_list, 256, 256, 128)
    # nn_training(model_list, 512, 256, 128)
    # nn_training(model_list, 512, 512, 128)

init()

app = Flask(__name__)
CORS(app)  # 这会为所有路由启用CORS




def getScore(str, model):
    str = re.sub(r"[^a-zA-Z\u4e00-\u9fa5]", "", str)
    str_list = list(jieba.cut(str))
    f_v = [[str_list.count(word) for word in top_words]]
    if model.predict(f_v)==1:
        return '正向'
    else:
        return '负向'


@app.route('/sentiment', methods=['POST'])
def sentiment():
    try:
        data = request.get_json()
        text = data['text']
        num_labels = {'二分类': 2, '三分类': 3, '六分类':6}[data['classificationType']]
        str1=''
        if num_labels==2:
            for mod in model_list:
                str1+=f'\n{str(mod)} 模型的预测结果为：{getScore(text, mod)}'
        str1+=f'\nBERT的预测结果为：{predict(text, bert[num_labels])}'
        return jsonify({'result': str1})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
@app.route('/test', methods=['GET'])
def test():
    return 'Server is working'

if __name__ == '__main__':
    # init()
    app.run(host='0.0.0.0',port=5000)
