# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

test_size = 0.2  # 数据集中被用成测试集的比例

N = 2  # 为了增加特征，我们将该天前N天的数据作为一个特征

model_seed = 100


def get_mape(y_true, y_pred):
    '''
    用来计算"平均绝对百分比误差"来评判是否正确
    Args:
        y_true: 正确的label
        y_pred: 预测的label

    Returns:

    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def _load_data():
    '''
    从VTI中获取数据
    '''
    stk_path = "./VTI.csv"
    df = pd.read_csv(stk_path, sep=",")

    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # 日期正则化

    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]  # 将标头变成小写

    df['month'] = df['date'].dt.month  # 增加一个month属性

    df.sort_values(by='date', inplace=True, ascending=True)  # 正则化之后可以利用时间排序

    return df


def feature_engineer(df):
    '''
    生成新特征：
        最高最低价差、开盘收盘价差
        前N天的信息拼入当天，作为当天的特征
    '''
    df['range_hl'] = df['high'] - df['low']
    df['range_oc'] = df['open'] - df['close']

    lag_cols = ['adj_close', 'range_hl', 'range_oc', 'volume']
    shift_range = [x + 1 for x in range(N)]

    for col in lag_cols:
        for i in shift_range:
            # 格式化字符串，生成前N天的数据放到该天的数据集中
            new_col = '{}_{}'.format(col, i)
            df[new_col] = df[col].shift(i)
    # 放弃前N天！
    return df[N:]


def scale_row(line, feat_mean, feat_std):
    '''

    Args:
        line: 需要标准化的数组
        feat_mean: 平均值
        feat_std: 标准差

    Returns:
        已经标准化的数组
    '''
    # 如果标准差为0，即数据不变化，就要把标准差设为一个很小的数，避免0成为了除数。
    if feat_std == 0:
        feat_std = 0.0001
    else:
        feat_std

    row_scaled = (line - feat_mean) / feat_std  # Z-score 标准化方法

    return row_scaled


def get_mov_avg_std(df, col, N):
    '''
    Args:
        df: 传入的数据集
        col: 想要计算标准差和均值的行名称
        N: 获取前几天的数据进行均值和标准差
    Returns:
        包含标准差和均值的数据集
    '''
    # 通过rolling(N).mean()获得前 N->0 个数据的平均值等
    # 但是第一行会产生数据NaN，因为0是NaN
    mean_list = df[col].rolling(window=N, min_periods=1).mean()
    std_list = df[col].rolling(window=N, min_periods=1).std()
    # 属性装入
    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list
    return df_out


def simp_adaboost_reg(Y_train, X_train, X_test, mean, std, Y_test, M=20, weak_clf=DecisionTreeRegressor(max_depth=1)):
    '''
    :param Y_train: 训练集的标签
    :param X_train: 训练集
    :param Y_test: 测试集的标签
    :param X_test: 测试集
    :param M: 基学习器的个数
    :param weak_clf: 基学习器
    :return:
    '''
    n_train, n_test = len(X_train), len(X_test)  # 训练数量和测试数量

    w = np.ones(n_train) / n_train  # 这个是每个样本的初始化权重

    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]  # 训练的预测值和测试预测值

    for i in range(M):
        # 训练一个基学习器
        weak_clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = weak_clf.predict(X_train)  # 获得n_train个预测值
        pred_test_i = weak_clf.predict(X_test)

        # 获取相对误差
        tempList = [abs(pred_train_i[i] - Y_train[i]) for i in range(0, len(pred_train_i))]
        down = np.max(tempList)
        miss = []
        for i in range(len(pred_train_i)):
            miss.append(abs(pred_train_i[i] - Y_train[i]) / down)

        # 误差率
        err_m = np.dot(w, miss)
        # 权重系数
        alpha_m = err_m / (1 - err_m)

        # New weights
        Z = np.dot(w, alpha_m)
        w = np.multiply(w, alpha_m / Z)
    weak_clf.fit(X_train, Y_train, sample_weight=w)
    final = weak_clf.predict(X_test)
    #还原数据
    for i in range(len(final)):
        final[i] = final[i] * std[i] + mean[i]
    # 可视化操作
    plt.plot(final, label='prediction')
    plt.plot(Y_test, label='real')
    plt.grid()
    plt.title('Simp-AdaBoost prediction')
    plt.legend()
    plt.savefig('./AdaBoost.jpg')
    plt.show()
    Ada_RMSE = math.sqrt(mean_squared_error(Y_test, final))
    Ada_MAPE = get_mape(Y_test, final)
    print("AdaBoost RMSE on dev set = %0.3f" % Ada_RMSE)
    print("Adaoost MAPE on dev set = %0.3f%%" % Ada_MAPE)

    return Ada_RMSE, Ada_MAPE


if __name__ == '__main__':
    dataDf = _load_data()
    # print(data_df)

    df = feature_engineer(dataDf)

    colsList = [
        "adj_close",
        "range_hl",
        "range_oc",
        "volume"
    ]
    for col in colsList:
        df = get_mov_avg_std(df, col, N)

    numTest = int(test_size * len(df))

    numTrain = len(df) - numTest
    # 训练集
    train = df[:numTrain]
    # 测试集
    test = df[numTrain:]

    colsToScale = [
        "adj_close"
    ]
    for i in range(1, N + 1):
        colsToScale.append("adj_close_" + str(i))
        colsToScale.append("range_hl_" + str(i))
        colsToScale.append("range_oc_" + str(i))
        colsToScale.append("volume_" + str(i))

    # 注意：标准化不能带测试集
    scaler = StandardScaler()
    trainScaled = scaler.fit_transform(train[colsToScale])  # 标准差归一化
    # 注意这里数据变成了numpy array，为了之后的panda操作，要将其换成pandas dataframe
    trainScaled = pd.DataFrame(trainScaled, columns=colsToScale)

    trainScaled[['date', 'month']] = train.reset_index()[['date', 'month']]

    testScaled = test[['date']]  # 保证测试的时间

    # 对测试集进行处理
    for col in tqdm(colsList):
        featList = [col + '_' + str(shift) for shift in range(1, N + 1)]
        temp = test.apply(lambda row: scale_row(row[featList], row[col + '_mean'], row[col + '_std']),
                          axis=1)  # 对测试集每一行进行标准化，调用函数scale_row
        testScaled = pd.concat([testScaled, temp], axis=1)

    # 建立样本
    features = []
    for i in range(1, N + 1):
        features.append("adj_close_" + str(i))
        features.append("range_hl_" + str(i))
        features.append("range_oc_" + str(i))
        features.append("volume_" + str(i))

    target = "adj_close"

    xSample = test[features]
    ySample = test[target]
    # 标准化后的训练参数
    xTrainScaled = trainScaled[features]
    yTrainScaled = trainScaled[target]
    xSampleScaled = testScaled[features]

    # --------------------------------开始训练--------------------------------

    ada_rmse, ada_mape = simp_adaboost_reg(yTrainScaled, xTrainScaled, xSampleScaled, test['adj_close_mean'].values,
                                         test['adj_close_std'].values, test['adj_close'].values)

    # 使用GridSearchCV进行参数微调

    parameters = {'n_estimators': [90],
                  'max_depth': [7],
                  'learning_rate': [0.3],
                  'min_child_weight': range(5, 21, 1),
                  }

    model = XGBRegressor(seed=model_seed,
                         n_estimators=100,
                         max_depth=3,
                         eval_metric='rmse',
                         learning_rate=0.1,
                         min_child_weight=1,
                         subsample=1,
                         colsample_bytree=1,
                         colsample_bylevel=1,
                         gamma=0)

    gs = GridSearchCV(estimator=model, param_grid=parameters, cv=5, refit=True, scoring='neg_mean_squared_error')

    gs.fit(xTrainScaled, yTrainScaled)

    preYScaled = gs.predict(xSampleScaled)
    test['pre_y_scaled'] = preYScaled
    test['pre_y'] = test['pre_y_scaled'] * test['adj_close_std'] + test['adj_close_mean']

    # 可视化分析

    plt.figure()
    ax = test.plot(x='date', y='adj_close', style='b-', grid=True)
    ax = test.plot(x='date', y='pre_y', style='r-', grid=True, ax=ax)
    plt.title('XGBoost prediction')
    plt.savefig('./XGBoost.jpg')
    plt.show()

    # 进行RMSE评估
    RMSE = math.sqrt(mean_squared_error(ySample, test['pre_y']))
    print("XGBoost RMSE on dev set = %0.3f" % RMSE)
    MAPE = get_mape(ySample, test['pre_y'])
    print("XGBoost MAPE on dev set = %0.3f%%" % MAPE)

# 官方AdaBoost 回归
    from sklearn.ensemble import AdaBoostRegressor
    clf = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1), n_estimators=20)
    clf.fit(xTrainScaled, yTrainScaled)
    pre = clf.predict(xSampleScaled)
    test['ada_prey_scaled'] = pre
    test['ada_prey'] = test['ada_prey_scaled'] * test['adj_close_std'] + test['adj_close_mean']
    of_ada_rmse = math.sqrt(mean_squared_error(ySample, test['ada_prey']))
    of_ada_mape = get_mape(ySample, test['ada_prey'])
    print("Official AdaBoost RMSE on dev set = %0.3f" % of_ada_rmse)
    print("Official AdaBoost MAPE on dev set = %0.3f" % of_ada_mape)


    plt.figure()
    ax = test.plot(x='date', y='adj_close', style='g-', grid=True)
    ax = test.plot(x='date', y='ada_prey', style='y-', grid=True, ax=ax)
    plt.title('Official AdaBoost prediction')
    plt.savefig('./OfficialAdaBoost.jpg')
    plt.show()

# 评价可视化
    name = ['RMSE', 'MAPE']
    Len = list(range(2))
    total_with, n = 0.4, 2
    width = total_with / n
    plt.bar(Len, [RMSE, MAPE], width=width, label='XGBoost', fc='y')
    for i in range(len(Len)):
        Len[i] = Len[i] + width
    plt.bar(Len, [ada_rmse, ada_mape], width=width, tick_label=name, label='Simp-AdaBoost', fc='b')
    for i in range(len(Len)):
        Len[i] = Len[i] + width
    plt.bar(Len, [of_ada_rmse, of_ada_mape], width=width, label='OfficalAdaBoost', fc='r')
    plt.legend()
    plt.title('Comparison chart of two regression evaluations')
    plt.savefig('./Comparison.jpg')
    plt.show()
