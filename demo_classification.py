# !usr/bin/env/python3
# -*- encoding:utf-8 -*-
# @date     : 2020/02/20
# @author   : CHEN WEI
# @filename : demo_classification.py
# @software : pycharm

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical


def find_rows(data, date_start, date_end, time_start):
    """
    行数查询器，根据起始或者结束日期确定样本所在行

    :param data: pd.DataFrame. Raw data read from csv.
    :param date_start, date_end: string. e.g. '2006-01-04'
    :param time_start: string. e.g. '09:35:00'
    :return: tuple.
    :examples
    >>> row_min, row_max = find_rows(raw_data, '2006-01-04', '2008-12-31', '09:35:00')
    """
    datetime_start = date_start + ' ' + time_start
    datetime_end = date_end + ' ' + time_start
    row_min = data.index.tolist().index(datetime_start)
    row_max = data.index.tolist().index(datetime_end)
    return row_min, row_max


def get_rows_interval(start, end, interval):
    """
    给定一个区间，划分成若干子区间。这个函数用在划分验证集的batch上，使得各个batch近似均衡。

    :param start: int. start side of the interval.
    :param end: int. end side of the interval.
    :param interval: int. number of sub_intervals.
    :return: list.
    :examples:
    >>> get_rows_interval(58140, 232860, 32)
    >>> get_rows_interval(233100, 349500, 4)
    >>> get_rows_interval(349740, 869580, 32)
    """
    interval_len = (end - start + 240) // 240 // interval
    rows_list = np.arange(start, end, interval_len * 240)
    if not (end - start + 240) % (interval_len * 240):
        return rows_list
    else:
        for i in range(1, (end - start + 240) // 240 % interval + 1):
            # 这里暂时有一个bug，可能不整除，但是对于我们的例子，step=240，一定是整除的
            rows_list[i:] += 240
        while rows_list[-1] >= end:
            rows_list = rows_list[:-1]
        return rows_list


def generator(data, lookback, delay, date_start, date_end, time_start,
              shuffle=False, batch_size=128, step=240, interval=32):
    """
    定义数据生成器
    两种训练方式：model.fit((X_train,y_train))和model.fit_generator(generator)
    fit需要将X_train和y_train读入内存，易造成内存泄露，Keras官方推荐使用fit_generator

    :return: samples, targets
    :examples
    >>> data = raw_data
    >>> lookback = param['time_lookback']
    >>> delay = param['time_delay']
    >>> date_start = param['date_train_start']
    >>> date_end = param['date_train_end']
    >>> time_start = param['time_pred']
    >>> generator(data, lookback, date_start, date_end, time_start)
    """
    # 调用行数查询器
    row_min, row_max = find_rows(data, date_start, date_end, time_start)

    # 生成验证集与测试集的batch块
    rows_gap = get_rows_interval(row_min, row_max, interval)
    i = 0

    # 逐个批次生成样本
    while 1:
        # 确定该批次样本所在行
        if shuffle:
            # 训练集需要随机打乱
            rows = np.random.choice(np.arange(row_min, row_max, step), batch_size)
        else:
            if i == (len(rows_gap) - 1):
                rows = np.arange(rows_gap[i], row_max + 1, step)
                i = 0
            else:
                rows = np.arange(rows_gap[i], rows_gap[i + 1], step)
                i += 1

        # 初始化标签和特征
        samples = np.zeros((len(rows), lookback, data.shape[-1]))
        targets = np.zeros((len(rows),))

        # 逐个样本生成标签和特征
        for j, row in enumerate(rows):
            # 提取特征
            indices = np.arange(rows[j] - lookback + 1, rows[j] + 1, step=1)
            samples[j] = data.iloc[indices]

            # 提取分类标签
            targets[j] = data.iloc[rows[j] + delay]['close'] / data.iloc[indices[-1]]['close'] - 1
            targets[j] = 1 if targets[j] > 0 else 0

            # 标准化
            mean = samples[j].mean(axis=0)
            std = samples[j].std(axis=0)
            samples[j] = (samples[j] - mean) / std

        yield samples, targets


def plot_history_acc(history):
    """
    绘制神经网络训练和验证正确率
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc')
    plt.legend()


def plot_history_loss(history, start=0):
    """
    绘制神经网络训练和验证损失函数
    """
    loss = history.history['loss'][start:]
    val_loss = history.history['val_loss'][start:]
    epochs = range(start + 1, start + len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def run_single_bkt(raw_data, param, prediction):
    """
    简单回测函数

    """
    row_min, row_max = find_rows(data=raw_data,
                                 date_start=param['date_test_start'],
                                 date_end=param['date_test_end'],
                                 time_start=param['time_pred'])

    # 提取测试集真实收益
    backtests = pd.DataFrame({'y_test':
                                  raw_data['close'].shift(-param['time_delay'])[
                                      np.arange(row_min, row_max + 1, MIN_PER_DAY)] \
                                  / raw_data['close'][np.arange(row_min, row_max + 1, MIN_PER_DAY)] - 1})

    # 提取测试集预测值
    backtests['prediction'] = prediction
    backtests['signal'] = np.sign(prediction - 0.5)
    backtests['signal_base'] = 1

    # 计算多头和空头收益
    backtests['return'] = np.multiply(backtests['y_test'], backtests['signal']) - param['fee']
    backtests['return_base'] = np.multiply(backtests['y_test'], backtests['signal_base']) - param['fee']
    backtests['value'] = (1 + backtests['return']).cumprod()
    backtests['value_base'] = (1 + backtests['return_base']).cumprod()

    return backtests


if __name__ == '__main__':
    # %% 参数
    MIN_PER_DAY = 240
    param = dict()

    # --- 回测参数
    param['fee'] = 0.0005

    # --- 其余参数
    param['path_data'] = os.path.join(os.getcwd(), '000300.XSHG.csv')
    param['date_train_start'] = '2006-01-04'  # 训练集开始日期 '2006-01-04'
    param['date_train_end'] = '2008-12-31'  # 训练集结束日期 '2008-12-31'
    param['date_val_start'] = '2009-01-05'  # 验证集开始日期 '2009-01-05'
    param['date_val_end'] = '2010-12-31'  # 验证集结束日期 '2010-12-31'
    param['date_test_start'] = '2011-01-04'  # 测试集开始日期 '2011-01-04'
    param['date_test_end'] = '2020-01-23'  # 测试集结束日期
    param['time_pred'] = '10:30:00'  # 站在哪个时刻做预测
    param['time_lookback'] = MIN_PER_DAY * 1 + 60  # 特征回溯时间，即使用过去多少分钟数据作为特征
    param['time_delay'] = 180  # 标签延迟时间，即预测未来多少分钟的收益
    param['name_features'] = ['high', 'close', 'open', 'low', 'volume', 'money']  # 特征列名
    param['batch_size'] = 64  # 神经网络batch size
    param['val_batch'] = 4
    param['test_batch'] = 32

    # %% 读取数据
    raw_data = pd.read_csv(param['path_data'], index_col=0)[param['name_features']]
    raw_data['pnl'] = raw_data['close'] / raw_data['close'].shift(1) - 1

    # %% 设置随机数种子
    np.random.seed(42)

    # %% 确定steps，每个epoch训练多少次
    # 一般取len(X_train)//batch_size
    row_min, row_max = find_rows(data=raw_data,
                                     date_start=param['date_train_start'],
                                     date_end=param['date_train_end'],
                                     time_start=param['time_pred'])
    train_steps = (row_max - row_min + 1) // MIN_PER_DAY // param['batch_size']


    # %% 生成训练、验证、测试数据
    train_gen = generator(data=raw_data,
                          lookback=param['time_lookback'],
                          delay=param['time_delay'],
                          date_start=param['date_train_start'],
                          date_end=param['date_train_end'],
                          time_start=param['time_pred'],
                          shuffle=True,
                          batch_size=param['batch_size'],
                          step=MIN_PER_DAY
                          )

    val_gen = generator(data=raw_data,
                        lookback=param['time_lookback'],
                        delay=param['time_delay'],
                        date_start=param['date_val_start'],
                        date_end=param['date_val_end'],
                        time_start=param['time_pred'],
                        shuffle=False,
                        step=MIN_PER_DAY,
                        interval=param['val_batch'])

    test_gen = generator(data=raw_data,
                         lookback=param['time_lookback'],
                         delay=param['time_delay'],
                         date_start=param['date_test_start'],
                         date_end=param['date_test_end'],
                         time_start=param['time_pred'],
                         shuffle=False,
                         step=MIN_PER_DAY,
                         interval=param['test_batch'])

    # %% 一维卷积神经网络及LSTM叠加
    tf.random.set_seed(123)
    model = Sequential()
    # 一维卷积层
    model.add(layers.Conv1D(64, 5, activation='tanh', input_shape=(None, raw_data.shape[-1])))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(128, 5, activation='tanh'))

    # LSTM层
    model.add(layers.LSTM(32, activation='tanh', return_sequences=True, dropout=0.1))
    model.add(layers.LSTM(64, activation='tanh', return_sequences=True, dropout=0.1))
    model.add(layers.LSTM(128, activation='tanh', dropout=0.1))

    # 输出层
    model.add(layers.Dense(128, activation='tanh'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=30,
                                  validation_data=val_gen,
                                  validation_steps=param['val_batch'])

    plot_history_loss(history)  # 训练和验证损失
    # plt.savefig('011701_loss.png')
    plot_history_acc(history)  # 训练和验证损失
    # plt.savefig('011701_acc.png')

    # %% 预测测试集
    prediction = model.predict_generator(test_gen, steps=param['test_batch'])

    # %%查看预测信号比例
    signal = np.sign(prediction-0.5)
    print(np.sum(signal > 0) / len(signal))
    print(np.sum(signal < 0) / len(signal))

    # 回测
    backtests = run_single_bkt(raw_data, param, prediction)
    backtests.loc[:, ['value', 'value_base']].plot(rot=30)

    # 保存回测数据
    with pd.ExcelWriter('2020021201.xlsx', datetime_format='YYYY-MM-DD') as writer:
        backtests.to_excel(writer, index=True)
    # %% 保存模型
    model.save('2020021201.h5')
