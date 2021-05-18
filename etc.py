import csv
import time
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# ダイヤモンドデータを表現するクラス
# 属性は重量(carat)，値段(price)，品質(clarity)のみとし，更にclarityは'IF'か'SI2'の何れかとする．
# carat == -1, price == -1 は各々の値が未知であることを表す．
# clarity == 'unknown' は品質が未知であることを表す．
class Diamond:
    def __init__(self, carat=-1, price=-1, clarity='unknown'):
        self.carat = carat
        self.price = price
        self.clarity = clarity


# ダイヤモンドデータの読み込み関数
#   - filename: データファイルの名前（'yyy/xxx.csv'）
#   - with_header: 1行目に項目名を表す行があるか否か（with_headerがTrueなら有，Falseなら無）
def read_diamond_data(filename, with_header=True):
    f = open(filename, 'r')
    reader = csv.reader(f)
    if with_header:
        next(reader) # 1行目が項目名を表す行なら，それを読み捨てる
    diamonds = []
    for row in reader:
        d = Diamond()
        if row[4] == 'IF' or row[4] == 'SI2':
            d.clarity = row[4] # clarity
        else:
            continue # clarity が 'IF' と 'SI2' の何れでもないものは無視
        x = np.zeros(2, dtype=np.float32)
        d.carat = float(row[1]) # carat
        d.price = float(row[7]) # price
        diamonds.append(d)
    f.close()
    return diamonds


# 事例データ集合を学習データとテストデータに分割する（分割基準はランダム）
#   - exapmles: 全事例データ集合
#   - ratio: 全体の何割をテストデータとして用いるか
def split(examples, ratio=0.1):
    n_total = len(examples)
    n_test = math.floor(n_total * ratio)
    perm = np.random.permutation(n_total)
    test_set = []
    train_set = []
    for i in range(0, n_test):
        test_set.append(examples[perm[i]])
    for i in range(n_test, n_total):
        train_set.append(examples[perm[i]])
    return train_set, test_set


# tミリ秒間，処理をストップする
def sleep_millisec(t):
    current_time = time.perf_counter()
    distination_time = current_time + t / 1000
    while current_time < distination_time:
        current_time = time.perf_counter()


def to_nparray(examples, scale=1.0):
    X = []
    y = []
    for e in examples:
        l = 0 if e.clarity == 'IF' else 1
        X.append([e.carat, e.price / scale])
        y.append(l)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    return X, y


# ダイヤモンドの重量と値段から clarity を推定する（k近傍法，高速化のためモジュールを使う）
#   - x1: 重量（carat）
#   - x2: 値段（price）
#   - examples: 事例データ
#   - k: k近傍のk
class kNNClarityClassifier2():

    def __init__(self, k):
        self.knn = KNeighborsClassifier(n_neighbors=k)

    def set_examples(self, examples):
        X, y = to_nparray(examples, scale=1000)
        self.knn.fit(X, y)

    def classify(self, x1, x2):
        if self.knn.predict(np.asarray([x1, x2/1000], dtype=np.float32).reshape((1, 2)))[0] == 0:
            return 'IF'
        else:
            return 'SI2'


# clarity 推定精度を計算する
#   - model: 認識モデル
#   - test_set: テストデータ集合
def evaluate_model(model, test_set):

    N = len(test_set) # テストデータの個数

    n_IF = 0
    c_IF = 0
    n_SI2 = 0
    c_SI2 = 0
    for i in range(N):
        t = test_set[i] # i 番目のテストデータ
        y = model.classify(t.carat, t.price / 1000)
        if t.clarity == 'IF':
            n_IF += 1
            if y == 'IF':
                c_IF += 1
        else:
            n_SI2 += 1
            if y != 'IF':
                c_SI2 += 1
    print('--------------------------------')
    print('|            |     推定結果    |')
    print('|            |-----------------|')
    print('|            |   IF   |  SI2   |')
    print('|------------|--------|--------|')
    print('|      | IF  | {0:>5.1f}% | {1:>5.1f}% |'.format(100 * c_IF / n_IF, 100 * (n_IF - c_IF) / n_IF))
    print('| 正解 |-----|--------|--------|')
    print('|      | SI2 | {0:>5.1f}% | {1:>5.1f}% |'.format(100 * c_SI2 / n_SI2, 100 * (n_SI2 - c_SI2) / n_SI2))
    print('--------------------------------')
    print()
