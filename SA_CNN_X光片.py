import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend
#from keras.utils import np_utils
import time
import math
import random

np.random.seed(2)
# 读取数据
raw_data = pd.read_csv("train_data1.csv")
raw_data_2=pd.read_csv("test_data1.csv")
raw_data_3=pd.read_csv("val_data1.csv")
m=(raw_data.shape[1])
print(m)
n=int((m-1)**0.5)
train_data = raw_data.iloc[:, 0:m-1].to_numpy(dtype=np.float32)
print(n)
test_data = raw_data_2.iloc[:, 0:m-1].to_numpy(dtype=np.float32)
val_data = raw_data_3.iloc[:, 0:m-1].to_numpy(dtype=np.float32)

# 标签处理
train_label = raw_data.iloc[:, m-1].to_numpy(dtype=np.float32)
test_label = raw_data_2.iloc[:, m-1].to_numpy(dtype=np.float32)
val_label = raw_data_3.iloc[:, m-1].to_numpy(dtype=np.float32)

# 数据预处理
train_data =0.1+0.8*(train_data/255)
Train_data=tf.reshape(train_data,[train_data.shape[0],n,n,1])
test_data = 0.1+0.8*(test_data/255)
Test_data=tf.reshape(test_data,[test_data.shape[0],n,n,1])
val_data = 0.1+0.8*(val_data/255)
Val_data=tf.reshape(val_data,[val_data.shape[0],n,n,1])

def CNN(X):
    T_Start = time.time()
    p = 0
    s = 1
    os1 = int((n - X[1] + 2 * p) / s) + 1
    os2 = int((os1 - X[2] + 2 * p) / s) + 1
    os3 = int((os2 - X[4] + 2 * p) / s) + 1
    if os1 < X[2]:
        scores1 = scores2 = scores3 = np.array([1, 0])
    else:
        if os2 < X[4]:
            scores1 = scores2 = scores3 = np.array([2, -1])
        else:
            if os3 < X[5]:
                scores1 = scores2 = scores3 = np.array([3, -2])
            else:
                # for test in range(3):

                model = Sequential([
                    Conv2D(X[0], (X[1], X[1]), activation='relu', strides = 1, input_shape=(n, n, 1)),
                    MaxPooling2D(pool_size=(X[2], X[2]),strides = 1),
                    # Conv2D(32,(3,3),activation='relu'),
                    # MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(X[3], (X[4], X[4]), strides = 1, activation='relu'),
                    MaxPooling2D(pool_size=(X[5], X[5]),strides = 1),
                    Dropout(0.1),
                    Flatten(),
                    Dense(X[6], activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),  # 添加l2正则项
                    Dense(1, activation='sigmoid')
                ])

                #model.summary()
                model.compile(optimizer='adam',
                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                              metrics=['accuracy'])
                model.fit(Train_data, train_label, validation_data=(Test_data, test_label), epochs=20, batch_size=10,
                          verbose=0)

                scores1 = model.evaluate(Train_data, train_label)
                scores2 = model.evaluate(Test_data, test_label)
                scores3 = model.evaluate(Val_data, val_label)
    print("X:\n", X)
    print("scores1: ", scores1)
    print("scores2: ", scores2)
    print("scores3: ", scores3)
    print('')
    # backend.clear_session()
    T_End = time.time()
    print("It cost %f sec" % (T_End - T_Start))
    return scores1, scores2, scores3

def fun(X):
    scores1, scores2, scores3 = CNN(X)
    F = float(1 - scores3[1])
    return F

# 自定义 SA() 函数
def SA():
    # 初始化 参数
    D = 7
    T_max = 100  # 初始温度
    T_min = 0.001  # 终止温度
    alpha = 0.95  # 温度冷却系数
    Iter_temp = 5  # 单一温度迭代次数
    R = 2  # 邻近范围
    T = T_max
    # 初始化 决策变数
    X = np.array([random.randint(1, 64), random.randint(1, 5), random.randint(1, 10), random.randint(1, 64),random.randint(1, 5), random.randint(1, 10), random.randint(2, 512)])
    X_U = [64, 5, 10, 64, 5, 10, 512]
    X_L = [1, 1, 1, 1, 1, 1, 2]
    F_X = fun(X)
    X_best = X
    F_best = F_X
    X_best_save_SA = []  # 用于存储 X_best
    F_best_save_SA = []  # 用于存储 F_best

    # 移步、取代、迭代
    Iter_total = 0
    while T >= T_min:
        for i in range(Iter_temp):
            Iter_total += 1
            X_try = X.copy()
            for j in range(D):
                X_try[j] = np.round(X[j] - R + 2 * R * np.random.random((1)))  # 移步
            for j in range(D):
                if X_try[j] > X_U[j]:  # 修正
                    X_try[j] = X_U[j]  # 超过最大时 设为最大
                if X_try[j] < X_L[j]:
                    X_try[j] = X_L[j]  # 低于最小时 设为最小
            # print("X_try: ", X_try)
            F_X_try = fun(X_try)  # 计算 f(X_try)
            # print("F_X_try: ", F_X_try)
            AcceptProb = np.exp(-(F_X_try - F_X) / T)  # 可接受概率
            Prob = np.random.random()
            if Prob < AcceptProb:  # 判断是否接受
                X = X_try  # 取代
                F_X = F_X_try  # 取代
            if F_X < F_best:  # 判断 是否 优于F_best
                X_best = X  # 更新 X_best
                F_best = F_X  # 更新 F_best
            X_best_save_SA.append(X_best)
            F_best_save_SA.append(F_best)
            print("Iteration: ", Iter_total)
            print("Local X:\n", X_best)
            print("Local F:\n", F_best)
        T = T * alpha  # 降温
    print("X best of SA:\n", X_best)
    print("F best of SA:\n", F_best)
    #print(Iter_total)
    return X_best_save_SA, F_best_save_SA

X_best_save_SA, F_best_save_SA = SA()
