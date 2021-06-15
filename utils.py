# coding: UTF-8

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ディレクトリパスを補正する（末尾に '/' がない場合，付与する）
def correct_dir_path(path, win_mode=False):
    if win_mode:
        if path[-1] != '/' and path[-1] != '\\':
            path += '\\'
    else:
        if path[-1] != '/':
            path += '/'
    return path


# .npy 形式のファイルを（ステータス表示付きで）ロードする
def load_npy(filename, print_status=True):
    data = np.load(filename)
    if print_status == True:
        print('load {0}'.format(filename), file=sys.stderr)
        print('  - shape: {0}'.format(data.shape), file=sys.stderr)
        print('  - dtype: {0}'.format(data.dtype), file=sys.stderr)
        print('  - max value: {0}'.format(data.max()), file=sys.stderr)
        print('  - min value: {0}'.format(data.min()), file=sys.stderr)
        print('', file=sys.stderr)
    return data


# 単一の画像を表示
#   - data: 表示対象データ
#   - title: グラフタイトル
#   - mode: 表示モード（0: グレースケール画像モード，それ以外: カラー画像モード）
def show_image(data, title='no title', mode=1):
    if mode == 0:
        img = np.asarray(data[0] * 255, dtype=np.uint8)
    else:
        img = np.asarray(data.transpose(1, 2, 0) * 255, dtype=np.uint8)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap=cm.gray, interpolation='nearest')
    plt.pause(2)
    plt.close()


# 途中経過の表示
#   - filename: 保存先画像のファイルパス
#   - data: 保存対象データ
#   - n_data_max: 保存するデータの最大数
#   - n_data_per_row: 保存先画像における一行あたりのデータ数
#   - mode: 保存モード（0: グレースケール画像モード，それ以外: カラー画像モード）
def save_progress(filename, data, n_data_max=100, n_data_per_row=10, mode=1):

    # 保存するデータの総数
    n_data_total = min(data.shape[0], n_data_max)

    # 保存先画像においてデータを何行に分けて表示するか
    n_rows = n_data_total // n_data_per_row
    if n_data_total % n_data_per_row != 0:
        n_rows += 1

    # 保存先画像の作成
    plt.figure(figsize = (n_data_per_row, n_rows))
    for i in range(0, n_data_total):
        plt.subplot(n_rows, n_data_per_row, i+1)
        plt.axis('off')
        if mode == 0:
            img = np.asarray(data[i][0] * 255, dtype=np.uint8)
        else:
            img = np.asarray(data[i].transpose(1, 2, 0) * 255, dtype=np.uint8)
        plt.imshow(img, cmap=cm.gray, interpolation='nearest')

    # 保存
    plt.savefig(filename)
    plt.close()


# 途中経過の表示
#   - data: 保存対象データ
#   - n_data_max: 保存するデータの最大数
#   - n_data_per_row: 保存先画像における一行あたりのデータ数
#   - mode: 保存モード（0: グレースケール画像モード，それ以外: カラー画像モード）
def show_progress(data, n_data_max=100, n_data_per_row=10, mode=1):

    # 保存するデータの総数
    n_data_total = min(data.shape[0], n_data_max)

    # 保存先画像においてデータを何行に分けて表示するか
    n_rows = n_data_total // n_data_per_row
    if n_data_total % n_data_per_row != 0:
        n_rows += 1

    # 保存先画像の作成
    plt.figure(figsize = (n_data_per_row, n_rows))
    for i in range(0, n_data_total):
        plt.subplot(n_rows, n_data_per_row, i+1)
        plt.axis('off')
        if mode == 0:
            img = np.asarray(data[i][0] * 255, dtype=np.uint8)
        else:
            img = np.asarray(data[i].transpose(1, 2, 0) * 255, dtype=np.uint8)
            img = img[:, :, [2, 1, 0]]
        plt.imshow(img, cmap=cm.gray, interpolation='nearest')

    # 保存
    plt.show()
    plt.close()
