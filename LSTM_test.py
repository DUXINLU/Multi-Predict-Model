# coding=UTF-8
from PyEMD import EEMD, EMD, visualisation
import numpy as np
import codecs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 这里是引用了交叉验证
from sklearn.linear_model import LinearRegression  # 线性回归
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model


def Show_as_fig():
    time, x, y = Get_data()
    x = PCA_proc(x)

    print(x.dtype, y.dtype)
    # np.savetxt("y.txt", y, delimiter=',', fmt="%s")
    plt.plot(range(len(time[:, 4])), x[:, 9])
    plt.show()


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


def PCA_proc(x):
    # feature normalization (feature scaling)
    X_scaler = StandardScaler()
    x = X_scaler.fit_transform(x)
    # PCA
    pca = PCA(n_components=0.9)  # 保证降维后的数据保持90%的信息
    x = pca.fit_transform(x)
    # PCA效果
    print("PCA:" + str(pca.explained_variance_ratio_.sum()))

    return x


def Get_info():
    info = []  # info字段 行数、名称、单位
    f = codecs.open('field definition.txt', 'r', 'utf-8')
    for line in f.readlines():
        _ = line.split("\t")
        _[1] = _[1].split(",")[0]
        _[2] = _[2][:-1]
        info.append(_)
    f.close()

    return info


def DHMXA_porc():
    pass


def EEMD_proc():
    time, x, y = Get_data()

    x = x[:1000, :]
    y = y[10:1010]

    tMin, tMax = 0, x.shape[0]
    T = np.linspace(tMin, tMax, tMax)

    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(100)

    E_IMFs = []
    print("Getting EIMFS...")

    # E_IMFs = np.loadtxt("E_IMFs")

    for i in range(x.shape[1]):
        _ = eemd.eemd(x[:, i], T)
        E_IMFs.append(_)
        print(_.shape)

    E_IMFs = np.concatenate(E_IMFs)
    print(E_IMFs.shape)

    return E_IMFs, y


def MLR_proc():
    _ = 5
    predict_peroid = 10
    time, x, y = Get_data()
    x = x[:-predict_peroid, :]
    y = y[predict_peroid:]
    X_train = x[:int(-x.shape[0] / _), :]
    X_test = x[int(-x.shape[0] / _):, :]
    y_train = y[:int(-x.shape[0] / _)]
    y_test = y[int(-x.shape[0] / _):]

    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)
    # print(model)
    # 训练后模型截距
    # print(linreg.intercept_)
    # 训练后模型权重（特征个数无变化）
    # print(linreg.coef_)

    # 输出RMSE输出、图表显示
    y_pred = linreg.predict(X_test)
    print(y_pred)  # 10个变量的预测结果

    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_test))  # 这个是你测试级的数量
    # calculate RMSE by hand
    print("RMSE", sum_erro)
    # 做ROC曲线
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    # plt.savefig("fig/MLR_result.jpg")
    plt.show()


def LSTM_proc():
    _ = 5  # test/all

    # using EEMD
    # E_IMFs, y = EEMD_proc()
    # E_IMFs = E_IMFs[0].T
    # X_train = E_IMFs[:int(-E_IMFs.shape[0] / _), :]
    # X_test = E_IMFs[int(-E_IMFs.shape[0] / _):, :]
    # y_train = y[:int(-E_IMFs.shape[0] / _)]
    # y_test = y[int(-E_IMFs.shape[0] / _):]

    # not using EEMD
    time, x, y = Get_data()
    x = x[:-10, :]
    y = y[10:]
    X_train = x[:int(-x.shape[0] / _), :]
    X_test = x[int(-x.shape[0] / _):, :]
    y_train = y[:int(-x.shape[0] / _)]
    y_test = y[int(-x.shape[0] / _):]

    print('X_train{}\ny_train{}\nX_test{}\ny_test{}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    data_gen = TimeseriesGenerator(X_train, y_train, length=10, batch_size=y_train.shape[0])
    X = []
    y = []
    for i in zip(*data_gen[0]):
        x_, y_ = i
        X.append(x_)
        y.append(y_)
    X = np.array(X)
    y = np.array(y)

    # plt.plot(range(y.shape[0]), y, label='y')
    # plt.show()

    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    history = model.fit(X, y, epochs=300, batch_size=128, validation_data=(X, y), verbose=1,
                        shuffle=False)
    model.save('model')

    # model = load_model('model')

    data_gen = TimeseriesGenerator(X_test, y_test, length=10, batch_size=y_test.shape[0])

    X = []
    y = []
    for i in zip(*data_gen[0]):
        x_, y_ = i
        X.append(x_)
        y.append(y_)
    X = np.array(X)
    y = np.array(y)

    # make a prediction
    y_predict = model.predict(X)
    plt.plot(range(y.shape[0]), y, label='Test')
    plt.plot(range(y.shape[0]), y_predict, label='Predict')
    plt.xlabel('Time')
    plt.ylabel('.')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    MLR_proc()
