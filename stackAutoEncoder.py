import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve

model_file = 'AE_model.h5'


def show_proportion(data_csv):
    num_nonfraud = np.sum(d['Class'] == 0)
    num_fraud = np.sum(d['Class'] == 1)
    plt.bar(['Black-data', 'White-data'], [num_nonfraud, num_fraud], color='dodgerblue')
    plt.show()


def parse_data(samples):
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

    max_length = 50

    # 结果保存在result中
    results = np.zeros((len(dataList), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(dataList):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    return results


def train_model(X_train, X_test):
    # 设置AutoEncoder的参数
    # 隐藏层节点数分别为16，8，8，16
    # epoch为50，batch size为32
    input_dim = X_train.shape[1]
    encoding_dim = 26
    num_epoch = 50
    batch_size = 52

    input_layer = Input(shape=(input_dim,))
    print(input_layer)

    # 五层结构:X -> 16 -> 8 -> 16 -> X'
    encoder = Dense(encoding_dim, activation="tanh",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['mae'])

    print("model ok")

    # 模型保存为SofaSofa_model.h5，并开始训练模型
    checkpointer = ModelCheckpoint(filepath=model_file,
                                   verbose=0,
                                   save_best_only=True)

    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_test, X_test),
                              verbose=1,
                              callbacks=[checkpointer]).history

    print("model training ok")

    # 画出损失函数曲线
    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.plot(history['loss'], c='dodgerblue', lw=3)
    plt.plot(history['val_loss'], c='coral', lw=3)
    plt.title('model loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(122)
    plt.plot(history['mae'], c='dodgerblue', lw=3)
    plt.plot(history['val_mae'], c='coral', lw=3)
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def get_score(X_test, X_fraud):
    # 读取模型
    autoencoder = load_model(model_file)

    # 利用训练好的autoencoder重建测试集
    pred_test = autoencoder.predict(X_test)
    pred_fraud = autoencoder.predict(X_fraud)

    # 计算还原误差MSE和MAE
    mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
    mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)
    mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
    mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)
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


if __name__ == '__main__':
    plt.style.use('seaborn')

    # 读取数据
    d = pd.read_csv('/Users/jianxinyu/Downloads/data.csv')

    # 查看样本比例
    #show_proportion(d)

    # 对Amount进行标准化
    data = d.drop(['Amount'], axis=1)
    # data['Data'] = StandardScaler().fit_transform(data[['Data']])

    # 提取负样本，并且按照8:2切成训练集和测试集
    mask = (data['Class'] == 0)
    X_train, X_test = train_test_split(data[mask], test_size=0.2, random_state=920)
    X_train = X_train.drop(['Class'], axis=1).values
    X_test = X_test.drop(['Class'], axis=1).values
    X_train = parse_data(X_train)
    X_test = parse_data(X_test)
    X_train = X_train.reshape((-1, 50))
    X_test = X_test.reshape((-1, 50))

    # 提取所有正样本，作为测试集的一部分
    X_fraud = data[~mask].drop(['Class'], axis=1).values
    X_fraud = parse_data(X_fraud)
    X_fraud = X_fraud.reshape((-1, 50))

    print("X_train:" + str(X_train.shape))
    print("X_test:" + str(X_test.shape))
    print("X_fraud:" + str(X_fraud.shape))

    train_model(X_train, X_test)

    get_score(X_test, X_fraud)
