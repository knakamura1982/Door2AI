import csv
import math
import numpy as np


# ダイアモンドの事例データファイルを読み込む関数
#   - filename: データファイルのファイルパス
def read_diamond_data(filename):

    # CSVファイルを開く
    f = open(filename, 'r')
    reader = csv.reader(f)

    # ヘッダ行を空読み
    next(reader)

    # データを読み込む
    lab = []
    feat = []
    for row in reader:
        if row[4] == 'IF':
            lab.append(0) # clarity が 'IF' ならクラス番号を 0 とする
        elif row[4] == 'SI2':
            lab.append(1) # clarity が 'SI2' ならクラス番号を 1 とする
        else:
            continue  # clarity が 'IF' と 'SI2' の何れでもないものは無視
        x = np.zeros(2, dtype=np.float32)
        x[0] = float(row[1]) # caratの値を1次元目に
        x[1] = float(row[7]) / 100 # priceの値を2次元目に
        feat.append(x)

    # CSVファイルを閉じる
    f.close()

    # 結果を返す
    #   - lab: ラベル配列（list）
    #   - feat: 特徴量配列（list）
    return lab, feat


# 身長・体重の事例データファイルを読み込む関数
#   - filename: データファイルのファイルパス
def read_height_weight_data(filename):

    # CSVファイルを開く
    f = open(filename, 'r')
    reader = csv.reader(f)

    # ヘッダ行を空読み
    next(reader)

    # データを読み込む
    lab = []
    feat = []
    for row in reader:
        if row[0] == 'Female':
            lab.append(0) # Gender が 'Female' ならクラス番号を 0 とする
        elif row[0] == 'Male':
            lab.append(1) # Gender が 'Male' ならクラス番号を 1 とする
        else:
            continue  # Gender が 'Female' と 'Male' の何れでもないものは無視
        x = np.zeros(2, dtype=np.float32)
        x[0] = float(row[1]) # heightの値を1次元目に
        x[1] = float(row[2]) # weightの値を2次元目に
        feat.append(x)

    # CSVファイルを閉じる
    f.close()

    # 結果を返す
    #   - lab: ラベル配列（list）
    #   - feat: 特徴量配列（list）
    return lab, feat


# 人工的に合成した事例データファイルを読み込む関数
#   - filename: データファイルのファイルパス
def read_artificial_data(filename):

    # CSVファイルを開く
    f = open(filename, 'r')
    reader = csv.reader(f)

    # ヘッダ行を空読み
    next(reader)

    # データを読み込む
    lab = []
    feat = []
    for row in reader:
        if row[0] == 'class 1':
            lab.append(0) # Class Label が 'class 1' ならクラス番号を 0 とする
        elif row[0] == 'class 2':
            lab.append(1) # Class Label が 'class 2' ならクラス番号を 1 とする
        else:
            continue  # Class Label が 'class 1' と 'class 2' の何れでもないものは無視
        x = np.zeros(2, dtype=np.float32)
        x[0] = float(row[1]) # Data 1 の値を1次元目に
        x[1] = float(row[2]) # Data 2 の値を2次元目に
        feat.append(x)

    # CSVファイルを閉じる
    f.close()

    # 結果を返す
    #   - lab: ラベル配列（list）
    #   - feat: 特徴量配列（list）
    return lab, feat


# データセットの偏りを補正する
#   - x: 入力データリスト（list）
#   - y: 出力データリスト（list）
def unbias(x, y):
    n = -1
    ids = []
    uni_labs = np.unique(np.asarray(y))
    for l in uni_labs:
        i = np.where(np.asarray(y) == l)[0]
        if n < 0 or len(i) < n:
            n = len(i)
        ids.append(i)
    new_x = []
    new_y = []
    for i in range(len(ids)):
        perm = np.random.permutation(len(ids[i]))
        for j in range(n):
            k = ids[i][perm[j]]
            new_y.append(y[k])
            new_x.append(x[k])
    return new_x, new_y


# データセットを訓練データと検証用データに分割
#   - x: 入力データリスト（list）
#   - y: 出力データリスト（list）
#   - ratio: 全体の何割を検証用データにするか（float）
def split(x, y, ratio):

    n_total = len(x)
    n_valid = math.floor(n_total * ratio)
    n_train = n_total - n_valid

    perm = np.random.permutation(n_total)

    x_train = []
    x_valid = []
    y_train = []
    y_valid = []
    for i in range(0, n_valid):
        x_valid.append(x[perm[i]])
        y_valid.append(y[perm[i]])
    for i in range(n_valid, n_total):
        x_train.append(x[perm[i]])
        y_train.append(y[perm[i]])

    return x_train, x_valid, y_train, y_valid
