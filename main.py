# coding=UTF-8
from PyEMD import EEMD, EMD, visualisation
import numpy as np
import codecs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 这里是引用了交叉验证
from sklearn.linear_model import LinearRegression  # 线性回归
import tensorflow


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

    tMin, tMax = 0, x.shape[0]
    T = np.linspace(tMin, tMax, tMax)

    eemd = EEMD()
    eemd.trials = 50
    eemd.noise_seed(100)

    E_IMFs = []
    print("Getting EIMFS...")

    for i in range(x.shape[1]):
        E_IMFs.append(eemd.eemd(i, T))

    return E_IMFs, y


def MLR_proc():
    time, x, y = Get_data()
    # print(x[:, 0].shape, x[:, 4:7].shape)
    # 选0、4、5、6维用做线性回归的输入
    # x = np.hstack((x[:, 0].reshape((x.shape[0], 1)), x[:, 4:7]))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)
    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,
                                                                                              y_train.shape,
                                                                                              X_test.shape,
                                                                                              y_test.shape))
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
    pass


def Show_as_fig():
    time, x, y = Get_data()
    x = PCA_proc(x)

    print(x.dtype, y.dtype)
    # np.savetxt("y.txt", y, delimiter=',', fmt="%s")
    plt.plot(range(len(time[:, 4])), x[:, 9])
    plt.show()


if __name__ == "__main__":
    plt.show()
