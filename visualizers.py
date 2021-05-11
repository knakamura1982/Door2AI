import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import torch
from imgproc import get_zero_crossing
from imgproc import cvt_gray2gradation
from imgproc import apply_mask


# 二クラス識別器の可視化器
class BCVisualizer():

    # コンストラクタ
    #   - size: 可視化画像のサイズ
    #   - hrange: 横軸の最小値・最大値
    #   - vrange: 縦軸の最小値・最大値
    #   - title: 可視化グラフのタイトル
    #   - hlabel: 横軸の軸ラベル名
    #   - vlabel: 縦軸の軸ラベル名
    #   - clabels: クラスラベル名（凡例の表示に使用）
    #   - bins: 縦軸・横軸の目盛りを何段階用意するか
    def __init__(self, size=512, hrange=(-1.0, 1.0), vrange=(-1.0, 1.0), title='title', hlabel='feature 1', vlabel='feature 2', clabels=('class 0', 'class 1'), bins=5):
        self.size = size
        self.bins = bins
        self.hrange = hrange
        self.vrange = vrange
        self.title = title
        self.hlabel = hlabel
        self.vlabel = vlabel
        self.clabels = clabels
        for i in range(0, size):
            y = np.ones((size, 1), dtype=np.float32) * i
            x = np.asarray([np.arange(size)], dtype=np.float32).transpose(1, 0)
            y = vrange[0] + y * (vrange[1] - vrange[0]) / (size - 1)
            x = hrange[0] + x * (hrange[1] - hrange[0]) / (size - 1)
            c = np.concatenate([x, y], axis=1)
            self.data = c if i == 0 else np.concatenate([self.data, c], axis=0)

    # 可視化の実行
    #   - model: 二クラス識別器クラスのインスタンス
    #   - device: デバイスオブジェクト（cpu か cuda か）
    #   - c0color: クラス 0 の表示色（RGBの順）
    #   - c1color: クラス 1 の表示色（RGBの順）
    #   - samples: 可視化画像に重畳する実データ
    #              samples[0] がラベルデータ(一次元の numpy.ndarray, int32)，
    #              samples[1] が特徴量データ(二次元の numpy.ndarray, float32) となるように指定
    #   - その他の引数: コンストラクタを参照．変更したい場合のみ指定する
    def show(self, model, device, c0color=(255, 0, 0), c1color=(0, 0, 255), samples=None, title=None, hlabel=None, vlabel=None, clabels=None, bins=None):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # パラメータ値に変更がある場合は更新
        if title is not None: self.title = title
        if hlabel is not None: self.hlabel = hlabel
        if vlabel is not None: self.vlabel = vlabel
        if clabels is not None: self.clabels = clabels
        if bins is not None: self.bins = bins

        # グラフタイトルの設定
        ax.set_title(self.title)

        # 背景画像の作成
        model.eval()
        result = model(torch.tensor(self.data, device=device)) + 0.5
        score = result.to('cpu').detach().numpy().copy()
        score = score[:,1].reshape([self.size, self.size])
        c0color_light = 128 + np.asarray(c0color, dtype=np.uint8) // 2
        c1color_light = 128 + np.asarray(c1color, dtype=np.uint8) // 2
        g = cvt_gray2gradation(score, c0color_light, c1color_light)
        z = get_zero_crossing(score - 0.5)
        img = apply_mask(g, z)
        del result

        # 背景画像の設定
        ax.imshow(img)

        # 実データの描画
        if samples is not None:
            lab, feat = samples
            colors = [np.asarray(c0color).reshape([1, 3]) / 255, np.asarray(c1color).reshape([1, 3]) / 255]
            ptx = np.floor((self.size - 1) * (feat[:,0] - self.hrange[0]) / (self.hrange[1] - self.hrange[0])).astype(np.int32)
            pty = np.floor((self.size - 1) * (feat[:,1] - self.vrange[0]) / (self.vrange[1] - self.vrange[0])).astype(np.int32)
            ax.scatter(ptx[lab==0], pty[lab==0], c=colors[0], label=self.clabels[0])
            ax.scatter(ptx[lab==1], pty[lab==1], c=colors[1], label=self.clabels[1])
            ax.legend(loc='upper left')

        # 縦軸・横軸の目盛りの作成
        ax.set_xticks(np.linspace(0, self.size-1, self.bins))
        ax.set_yticks(np.linspace(0, self.size-1, self.bins))
        hlabels = []
        vlabels = []
        for i in range(0, self.bins):
            hlabels.append(format(self.hrange[0] + i * (self.hrange[1] - self.hrange[0]) / (self.bins - 1), '.2f'))
            vlabels.append(format(self.vrange[0] + i * (self.vrange[1] - self.vrange[0]) / (self.bins - 1), '.2f'))
        ax.set_xticklabels(hlabels)
        ax.set_yticklabels(vlabels)
        ax.grid(True)

        # 軸ラベルを設定
        ax.set_xlabel(self.hlabel)
        ax.set_ylabel(self.vlabel)

        # 縦軸の上下を反転
        ax.invert_yaxis()

        # 表示（2秒後にウィンドウを閉じる）
        plt.pause(2)
        plt.close()
