# coding=UTF-8

from PyEMD import EEMD, EMD, visualisation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 这里是引用了交叉验证
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.models import load_model
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def PCA_proc(x):
    # feature normalization (feature scaling)
    X_scaler = StandardScaler()
    x = X_scaler.fit_transform(x)

    # PCA
    pca = PCA(n_components=0.9)  # 保证降维后的数据保持90%的信息
    x = pca.fit_transform(x)

    # PCA效果
    print(pca.explained_variance_ratio_.sum())

    return x


def Get_data():
    # env_data = np.loadtxt("201910.txt", dtype=np.float16, delimiter=",")
    env_data = []
    f = open("201910.txt", "r")
    for line in f.readlines():
        _ = line.split(",")
        env_data.append(_[:5] + [_[i] for i in range(5, len(_[5:])) if i % 2 == 1])
    f.close()

    env_data = np.array(env_data, dtype=np.float)

    time = env_data[:, :5]  # 第几分钟的数据
    time.astype(np.int32)
    x = PCA_proc(env_data[:, 6:])  # feature
    y = env_data[:, 5]  # label

    return time, x, y


def EEMD_proc():
    time, x, y = Get_data()
    pre_period = 10
    y = y[:-pre_period]  # sample
    y_ = y[pre_period:]  # label

    tMin, tMax = 0, y.shape[0]
    T = np.linspace(tMin, tMax, tMax)

    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(100)

    print("Getting EIMFS...")
    E_IMFs = eemd.eemd(y, T)
    print(E_IMFs.shape)
    # E_IMFs = np.array(E_IMFs)
    # print(E_IMFs.shape)
    return E_IMFs, y.T

def LSTM_proc():
    E_IMFs, y = EEMD_proc()
    E_IMFs = E_IMFs.T  # 4w+*17
    print(E_IMFs.shape, y.shape)
    X_train = E_IMFs[:int(-44640 / 31), :]
    X_test = E_IMFs[int(-44640 / 31):, :]
    y_train = y[:int(-44640 / 31)]
    y_test = y[int(-44640 / 31):]

    X_train = X_train.reshape((X_train.shape[0] / 10, 10, int(X_train.shape[1])))
    X_test = X_test.reshape((X_test.shape[0] / 10, 10, int(X_test.shape[1])))

    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,
                                                                                              y_train.shape,
                                                                                              X_test.shape,
                                                                                              y_test.shape))

    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(X_train, y_train, epochs=1000, batch_size=256, validation_data=(X_test, y_test), verbose=1,
                        shuffle=False)
    model.save('model')

    # model = load_model('model')

    # make a prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    yhat = model.predict(X_test)
    print(yhat.shape)
    print(y.shape)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
    print(X_test.shape)

    y1 = yhat
    y2 = y_test
    plt.plot(range(y_test.shape[0]), y1, label='Frist line')
    plt.plot(range(y_test.shape[0]), y2, label='second line')
    plt.xlabel('Plot Number')
    plt.ylabel('Important var')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()
    '''
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, X_test[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    inv_y = scaler.inverse_transform(X_test)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    r2 = r2_score(inv_y, inv_yhat)
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('RMSE: %.3f R2: %.3f MAE: %.3f' % (rmse, r2, mae))
    '''


if __name__ == '__main__':
    EEMD_proc()
