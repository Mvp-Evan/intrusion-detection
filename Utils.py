import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def parse_data_one_hot(samples):
    dataList = []
    for data_cale in samples:
        dataList.append(data_cale[0])

    # 构建数据中被标记的索引
    token_index = {}
    for sample in dataList:
        # 利用split方法进行分词
        for word in sample.split():
            if word not in token_index:
                # 为唯一单词指定唯一索引
                token_index[word] = len(token_index) + 1

    max_length = 20

    # 结果保存在result中
    results = np.zeros((len(dataList), max_length, 20))
    for i, sample in enumerate(dataList):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    return results


def show_proportion(data_csv):
    num_nonfraud = np.sum(data_csv['Class'] == 0)
    num_fraud = np.sum(data_csv['Class'] == 1)
    plt.bar(['Black-data', 'White-data'], [num_nonfraud, num_fraud], color='dodgerblue')
    plt.show()


class Utils:
    def __init__(self, model_path):
        self.__data_path = '/Users/jianxinyu/Downloads/data.csv'
        self.__model_path = model_path
        self.__X_train = None
        self.__X_test = None
        self.__X_white = None

    # 预处理数据，转one-hot编码
    def set_data(self):
        plt.style.use('seaborn')

        # 读取数据
        d = pd.read_csv(self.__data_path)

        # 查看样本比例
        show_proportion(d)

        data = d.drop(['Amount'], axis=1)

        # 提取负样本，并且按照8:2切成训练集和测试集
        mask = (data['Class'] == 0)
        self.__X_train, self.__X_test = train_test_split(data[mask], test_size=0.2, random_state=920)
        self.__X_train = self.__X_train.drop(['Class'], axis=1).values
        self.__X_test = self.__X_test.drop(['Class'], axis=1).values
        self.__X_train = parse_data_one_hot(self.__X_train)
        self.__X_test = parse_data_one_hot(self.__X_test)

        # 提取所有正样本，作为测试集的一部分
        self.__X_white = data[~mask].drop(['Class'], axis=1).values
        self.__X_white = parse_data_one_hot(self.__X_white)

    # 压缩数据矩阵为数据向量
    def to_line(self):
        self.__X_train = self.__X_train.reshape((-1, 50))
        self.__X_test = self.__X_test.reshape((-1, 50))
        self.__X_white = self.__X_white.reshape((-1, 50))

    # 计算mse，mae打分
    def get_score(self):
        # 读取模型
        autoencoder = load_model(self.__model_path)

        # 利用训练好的autoencoder重建测试集
        pred_test = autoencoder.predict(self.__X_test)
        pred_fraud = autoencoder.predict(self.__X_white)

        # 计算还原误差MSE和MAE
        mse_test = np.mean(np.power(self.__X_test - pred_test, 2), axis=1)
        mse_fraud = np.mean(np.power(self.__X_white - pred_fraud, 2), axis=1)
        mae_test = np.mean(np.abs(self.__X_test - pred_test), axis=1)
        mae_fraud = np.mean(np.abs(self.__X_white - pred_fraud), axis=1)
        mse_df = pd.DataFrame()
        mse_df['Class'] = [0] * len(mse_test) + [1] * len(mse_fraud)
        mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
        mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
        mse_df = mse_df.sample(frac=1).reset_index(drop=True)
        # 分别画出测试集中正样本和负样本的还原误差MAE和MSE
        markers = ['o', '^']
        markers = ['o', '^']
        colors = ['dodgerblue', 'coral']
        labels = ['Black', 'White']

        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        for flag in [1, 0]:
            temp = mse_df[mse_df['Class'] == flag]
            plt.scatter(temp.index,
                        temp['MAE'],
                        alpha=0.7,
                        marker=markers[flag],
                        c=colors[flag],
                        label=labels[flag])
        plt.title('Reconstruction MAE')
        plt.ylabel('Reconstruction MAE')
        plt.xlabel('Index')
        plt.subplot(122)
        for flag in [1, 0]:
            temp = mse_df[mse_df['Class'] == flag]
            plt.scatter(temp.index,
                        temp['MSE'],
                        alpha=0.7,
                        marker=markers[flag],
                        c=colors[flag],
                        label=labels[flag])
        plt.legend(loc=[1, 0], fontsize=12)
        plt.title('Reconstruction MSE')
        plt.ylabel('Reconstruction MSE')
        plt.xlabel('Index')
        plt.show()
        # 画出Precision-Recall曲线
        plt.figure(figsize=(14, 6))
        for i, metric in enumerate(['MAE', 'MSE']):
            plt.subplot(1, 2, i + 1)
            precision, recall, _ = precision_recall_curve(mse_df['Class'], mse_df[metric])
            pr_auc = auc(recall, precision)
            plt.title('Precision-Recall curve based on %s\nAUC = %0.2f' % (metric, pr_auc))
            plt.plot(recall[:-2], precision[:-2], c='coral', lw=4)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        plt.show()

        # 画出ROC曲线
        plt.figure(figsize=(14, 6))
        for i, metric in enumerate(['MAE', 'MSE']):
            plt.subplot(1, 2, i + 1)
            fpr, tpr, _ = roc_curve(mse_df['Class'], mse_df[metric])
            roc_auc = auc(fpr, tpr)
            plt.title('Receiver Operating Characteristic based on %s\nAUC = %0.2f' % (metric, roc_auc))
            plt.plot(fpr, tpr, c='coral', lw=4)
            plt.plot([0, 1], [0, 1], c='dodgerblue', ls='--')
            plt.ylabel('TPR')
            plt.xlabel('FPR')
        plt.show()
        # 画出MSE、MAE散点图
        markers = ['o', '^']
        colors = ['dodgerblue', 'coral']
        labels = ['Black', 'White']

        plt.figure(figsize=(10, 5))
        for flag in [1, 0]:
            temp = mse_df[mse_df['Class'] == flag]
            plt.scatter(temp['MAE'],
                        temp['MSE'],
                        alpha=0.7,
                        marker=markers[flag],
                        c=colors[flag],
                        label=labels[flag])
        plt.legend(loc=[1, 0])
        plt.ylabel('Reconstruction RMSE')
        plt.xlabel('Reconstruction MAE')
        plt.show()

    def get_train_set(self):
        return self.__X_train

    def get_test_set(self):
        return self.__X_test

    def get_white_set(self):
        return self.__X_white
