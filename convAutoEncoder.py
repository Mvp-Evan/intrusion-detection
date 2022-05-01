import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import Utils

model_file = 'CAE_model.h5'


def train_model(X_train, X_test):
    # 数据升维度
    X_train = tf.expand_dims(X_train, axis=-1)
    X_test = tf.expand_dims(X_test, axis=-1)

    # 设置AutoEncoder的参数
    # 隐藏层节点数分别为16，8，8，16
    # epoch为50，batch size为32
    input_dim = X_train.shape[1]
    encoding_dim = 26
    num_epoch = 50
    batch_size = 52

    print("X_train:" + str(X_train.shape))
    print("X_test:" + str(X_test.shape))
    print("X_fraud:" + str(X_white.shape))
    input_layer = Input(shape=(input_dim, 20, 1))
    print(input_layer)

    # Encoder
    conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
    conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
    h = MaxPooling2D((2, 2), padding='same')(conv1_3)

    # Decoder
    conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(conv2_1)
    conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(conv2_2)
    conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)
    up3 = UpSampling2D((2, 2))(conv2_3)
    r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)

    autoencoder = Model(inputs=input_layer, outputs=r)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    print("model ok")

    # 模型保存为SofaSofa_model.h5，并开始训练模型
    checkpointer = ModelCheckpoint(filepath=model_file,
                                   verbose=0,
                                   save_best_only=True)

    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epoch,
                              steps_per_epoch=20,
                              shuffle=True,
                              validation_data=(X_test, X_test),
                              validation_steps=50,
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


if __name__ == '__main__':
    utils = Utils.Utils(model_file)

    utils.set_data()

    X_train = utils.get_train_set()
    X_test = utils.get_test_set()
    X_white = utils.get_white_set()

    train_model(X_train, X_test)

    utils.get_score()
