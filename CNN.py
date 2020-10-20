import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend
import time
import random

#np.random.seed(2)
# 读取数据
raw_data = pd.read_csv("train_data1.csv")
raw_data_2=pd.read_csv("test_data1.csv")
raw_data_3=pd.read_csv("val_data1.csv")
m=(raw_data.shape[1])
#print(m)
n=int((m-1)**0.5)
train_data = raw_data.iloc[:, 0:m-1].to_numpy(dtype=np.float32)
#print(n)
test_data = raw_data_2.iloc[:, 0:m-1].to_numpy(dtype=np.float32)
val_data = raw_data_3.iloc[:, 0:m-1].to_numpy(dtype=np.float32)

# 标签处理
train_label = raw_data.iloc[:, m-1].to_numpy(dtype=np.float32)
test_label = raw_data_2.iloc[:, m-1].to_numpy(dtype=np.float32)
val_label = raw_data_3.iloc[:, m-1].to_numpy(dtype=np.float32)

# 数据预处理
train_data = 0.1 + 0.8 * (train_data/255)
Train_data=tf.reshape(train_data,[train_data.shape[0],n,n,1])
test_data = 0.1 + 0.8 * (test_data/255)
Test_data=tf.reshape(test_data,[test_data.shape[0],n,n,1])
val_data = 0.1 + 0.8 * (val_data/255)
Val_data=tf.reshape(val_data,[val_data.shape[0],n,n,1])

def CNN(X):
    T_Start = time.time()
    scores1_mean = 0
    scores2_mean = 0
    scores3_mean = 0
    Scores_Total = np.zeros([30,3])
    for test in range(30):
        model = Sequential([
                    Conv2D(int(X[0]), (int(X[1]), int(X[1])), activation='relu', strides = 2, input_shape=(n, n, 1)),
                    MaxPooling2D(pool_size=(int(X[2]), int(X[2])) , strides = 2),
                    # Conv2D(32,(3,3),activation='relu'),
                    # MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(int(X[3]), (int(X[4]), int(X[4])), strides = 2,activation='relu'),
                    MaxPooling2D(pool_size=(int(X[5]), int(X[5])), strides = 2),
                    Dropout(0.1),
                    Flatten(),
                    Dense(int(X[6]), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),  # 添加l2正则项
                    Dense(1,activation='sigmoid')
                ])

        # model.summary()
        model.compile(optimizer='adam',
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                         metrics=['accuracy'])
        history = model.fit(Train_data, train_label, validation_data=(Test_data, test_label), epochs=500, batch_size=10,
                        verbose=0)

        scores1 = model.evaluate(Train_data,train_label,verbose=0)
        scores2 = model.evaluate(Test_data,test_label,verbose=0)
        scores3 = model.evaluate(Val_data,val_label,verbose=0)
        Scores_Total[test, 0] = scores1[1]
        Scores_Total[test, 1] = scores2[1]
        Scores_Total[test, 2] = scores3[1]
        print("test:\n", test + 1)
        print("scores1: ",scores1)
        print("scores2: ",scores2)
        print("scores3: ",scores3)
        print('')
        backend.clear_session()

    Scores_Total = Scores_Total[Scores_Total[:, -1].argsort()]

    for i in range(10,20):
        scores1_mean = scores1_mean + Scores_Total[i, 0]
        scores2_mean = scores2_mean + Scores_Total[i, 1]
        scores3_mean = scores3_mean + Scores_Total[i, 2]
    scores1_mean = scores1_mean / 10
    scores2_mean = scores2_mean / 10
    scores3_mean = scores3_mean / 10
    print('scores1_mean: ', scores1_mean)
    print('scores2_mean: ', scores2_mean)
    print('scores3_mean: ', scores3_mean)
    T_End = time.time()
    print("It cost %f sec" % (T_End - T_Start))
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    y_pred = model.predict(Val_data, batch_size=32, verbose=1).ravel()
    y_pred = (y_pred >= 0.5).astype(int)
    submission = pd.read_csv("sample_submission.csv")
    submission['target'] = y_pred
    submission.to_csv('nlp_prediction.csv', index=False)
    return

X = np.array([15, 5, 3, 35, 7, 2, 98])

CNN(X)