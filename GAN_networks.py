# coding: UTF-8

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from unit_layers import Conv, DownConv, UpConv, Pool, GlobalPool, PixShuffle, ResBlockP, FC, Flatten, Reshape


# ジェネレータ
class Generator(nn.Module):

    L1_CHANNELS = 512
    L2_CHANNELS = 256
    L3_CHANNELS = 128
    L4_CHANNELS = 64
    L5_CHANNELS = 32

    # コンストラクタ
    #   - width: 出力画像の横幅
    #   - height: 出力画像の縦幅
    #   - channels: 出力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    #   - nz: 入力乱数の次元数
    def __init__(self, width, height, channels, nz):
        super(Generator, self).__init__()
        # 定数の定義
        self.w1 = width // (2**4)
        self.h1 = height // (2**4)
        self.n1 = self.w1 * self.h1 * self.L1_CHANNELS
        # 層の定義
        self.fc1 = FC(in_units=nz, out_units=self.n1, do_bn=False, activation=F.leaky_relu)
        self.rs1 = Reshape((self.L1_CHANNELS, self.h1, self.w1))
        self.up2 = PixShuffle(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.up3 = PixShuffle(in_channels=self.L2_CHANNELS, out_channels=self.L3_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.up4 = PixShuffle(in_channels=self.L3_CHANNELS, out_channels=self.L4_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.up5 = PixShuffle(in_channels=self.L4_CHANNELS, out_channels=self.L5_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.res2 = ResBlockP(in_channels=self.L2_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.res3 = ResBlockP(in_channels=self.L3_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.res4 = ResBlockP(in_channels=self.L4_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.res5 = ResBlockP(in_channels=self.L5_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.conv6 = Conv(in_channels=self.L5_CHANNELS, out_channels=channels, kernel_size=1, do_bn=False, activation=torch.tanh)

    # 順伝播
    #   - z: 入力乱数（ミニバッチ）
    def forward(self, z):
        # コンストラクタで定義した層を順に適用していく
        h = self.rs1(self.fc1(z))
        h = self.res2(self.up2(h))
        h = self.res3(self.up3(h))
        h = self.res4(self.up4(h))
        h = self.res5(self.up5(h))
        return self.conv6(h)


# ディスクリミネータ
class Discriminator(nn.Module):

    L1_CHANNELS = 16
    L2_CHANNELS = 32
    L3_CHANNELS = 64
    L4_CHANNELS = 128
    L5_CHANNELS = 128
    L6_UNITS = 256

    # コンストラクタ
    #   - width: 入力画像の横幅
    #   - height: 入力画像の縦幅
    #   - channels: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    def __init__(self, width, height, channels):
        super(Discriminator, self).__init__()
        # 定数の定義
        self.w5 = width // (2**4)
        self.h5 = height // (2**4)
        self.n5 = self.w5 * self.h5 * self.L5_CHANNELS
        # 層の定義
        self.conv1 = Conv(in_channels=channels, out_channels=self.L1_CHANNELS, do_bn=True, activation=F.leaky_relu)
        self.down2 = DownConv(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, do_bn=True, dropout_ratio=0.5, activation=F.leaky_relu)
        self.down3 = DownConv(in_channels=self.L2_CHANNELS, out_channels=self.L3_CHANNELS, do_bn=True, dropout_ratio=0.5, activation=F.leaky_relu)
        self.down4 = DownConv(in_channels=self.L3_CHANNELS, out_channels=self.L4_CHANNELS, do_bn=True, dropout_ratio=0.5, activation=F.leaky_relu)
        self.down5 = DownConv(in_channels=self.L4_CHANNELS, out_channels=self.L5_CHANNELS, do_bn=True, dropout_ratio=0.5, activation=F.leaky_relu)
        self.fl5 = Flatten()
        self.fc6 = FC(in_units=self.n5, out_units=self.L6_UNITS, do_bn=True, activation=F.leaky_relu)
        self.fc7 = FC(in_units=self.L6_UNITS, out_units=1, do_bn=False, activation=None)

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x, use_sigmoid=True):
        # コンストラクタで定義した層を順に適用していく
        h = self.conv1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.fl5(self.down5(h))
        h = self.fc6(h)
        h = self.fc7(h)
        if use_sigmoid:
            return torch.sigmoid(h)
        else:
            return h


# コードディスクリミネータ
class CodeDiscriminator(nn.Module):

    # コンストラクタ
    #   - nz: 入力乱数の次元数
    #   - nh: 隠れ層のユニット数
    def __init__(self, nz, nh):
        super(CodeDiscriminator, self).__init__()
        # 層の定義
        self.fc1 = FC(in_units=nz, out_units=nh, do_bn=True, dropout_ratio=0.0, activation=F.leaky_relu)
        self.fc2 = FC(in_units=nh, out_units=nh, do_bn=True, dropout_ratio=0.0, activation=F.leaky_relu)
        self.fc3 = FC(in_units=nh, out_units=nh, do_bn=True, dropout_ratio=0.0, activation=F.leaky_relu)
        self.fc4 = FC(in_units=nh, out_units=1, do_bn=False, activation=None)

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x, use_sigmoid=True):
        # コンストラクタで定義した層を順に適用していく
        h = self.fc1(x)
        h = self.fc2(h)
        h = self.fc3(h)
        h = self.fc4(h)
        if use_sigmoid:
            return torch.sigmoid(h)
        else:
            return h


# エンコーダ
class Encoder(nn.Module):

    L1_CHANNELS = 16
    L2_CHANNELS = 32
    L3_CHANNELS = 64
    L4_CHANNELS = 128
    L5_UNITS = 256

    # コンストラクタ
    #   - width: 入力画像の横幅
    #   - height: 入力画像の縦幅
    #   - channels: 入力画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    #   - nz: 出力特徴量の次元数
    #   - out_scale: 出力特徴量のスケールファクタ（最後に出力特徴量を out_scale 倍する）
    def __init__(self, width, height, channels, nz, out_scale=1):
        super(Encoder, self).__init__()
        # 定数の定義
        self.w4 = width // (2**4)
        self.h4 = height // (2**4)
        self.n4 = self.w4 * self.h4 * self.L4_CHANNELS
        self.out_scale = out_scale
        # 層の定義
        self.conv1 = Conv(in_channels=channels, out_channels=self.L1_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.down1 = DownConv(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.down2 = DownConv(in_channels=self.L2_CHANNELS, out_channels=self.L3_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.down3 = DownConv(in_channels=self.L3_CHANNELS, out_channels=self.L4_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.down4 = DownConv(in_channels=self.L4_CHANNELS, out_channels=self.L4_CHANNELS, do_bn=False, activation=F.leaky_relu)
        self.fl4 = Flatten()
        self.fc5 = FC(in_units=self.n4, out_units=self.L5_UNITS, do_bn=False, activation=F.relu)
        self.fc6 = FC(in_units=self.L5_UNITS, out_units=nz, do_bn=False, activation=torch.tanh)
        self.fc6_lnvar = FC(in_units=self.L5_UNITS, out_units=nz, do_bn=False, activation=None) # VAE用

    # 順伝播（最終層直前まで）
    #   - x: 入力画像（ミニバッチ）
    def __forward_base_layers(self, x):
        # コンストラクタで定義した層を順に適用していく
        h = self.down1(self.conv1(x))
        h = self.down2(h)
        h = self.down3(h)
        h = self.fl4(self.down4(h))
        return self.fc5(h)

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        h = self.__forward_base_layers(x)
        return self.out_scale * self.fc6(h)

    # 順伝播（VAE用）
    def encode(self, x):
        h = self.__forward_base_layers(x)
        return self.out_scale * self.fc6(h), self.fc6_lnvar(h)


# 認識器
class Classifier(nn.Module):

    L1_CHANNELS = 16
    L2_CHANNELS = 32
    L3_CHANNELS = 64
    L4_CHANNELS = 128
    L5_CHANNELS = 128

    # コンストラクタ
    #   - width: 画像の横幅
    #   - height: 画像の縦幅
    #   - channels: 画像のチャンネル数（グレースケール画像なら1，カラー画像なら3）
    #   - nz: 全結合層のユニット数
    #   - nc: 認識対象クラスの総数
    def __init__(self, width, height, channels, nz, nc):
        super(Classifier, self).__init__()
        # 定数の定義
        self.w5 = width // (2**4)
        self.h5 = height // (2**4)
        self.n5 = self.w5 * self.h5 * self.L5_CHANNELS
        # 層の定義
        self.conv1 = Conv(in_channels=channels, out_channels=self.L1_CHANNELS, do_bn=True, activation=F.relu)
        self.down2 = DownConv(in_channels=self.L1_CHANNELS, out_channels=self.L2_CHANNELS, do_bn=True, activation=F.relu)
        self.down3 = DownConv(in_channels=self.L2_CHANNELS, out_channels=self.L3_CHANNELS, do_bn=True, activation=F.relu)
        self.down4 = DownConv(in_channels=self.L3_CHANNELS, out_channels=self.L4_CHANNELS, do_bn=True, activation=F.relu)
        self.down5 = DownConv(in_channels=self.L4_CHANNELS, out_channels=self.L5_CHANNELS, do_bn=True, activation=F.relu)
        self.res2 = ResBlockP(in_channels=self.L2_CHANNELS, do_bn=True, activation=F.relu)
        self.res3 = ResBlockP(in_channels=self.L3_CHANNELS, do_bn=True, activation=F.relu)
        self.res4 = ResBlockP(in_channels=self.L4_CHANNELS, do_bn=True, activation=F.relu)
        self.fl5 = Flatten()
        self.fc6 = FC(in_units=self.n5, out_units=nz, do_bn=True, activation=F.relu)
        self.fc7 = FC(in_units=nz, out_units=nc, do_bn=False, activation=None)

    # 順伝播
    #   - x: 入力画像（ミニバッチ）
    def forward(self, x):
        h = self.conv1(x)
        h = self.res2(self.down2(h))
        h = self.res3(self.down3(h))
        h = self.res4(self.down4(h))
        h = self.fl5(self.down5(h))
        h = self.fc6(h)
        return self.fc7(h)

    # 認識
    #   - x: 入力画像（ミニバッチ）
    def classify(self, x, score='none'):
        h = self.forward(x)
        y = torch.argmax(h, dim=1)
        if score == 'sigmoid_full': # 全クラスのスコア（クラス毎に独立に0〜1）を返す
            return y, torch.sigmoid(h)
        elif score == 'sigmoid_only_max': # 各クラスのスコア（クラス毎に独立に0〜1）を求めた後，最大値のみを返す
            return y, torch.max(torch.sigmoid(h), dim=1)[0]
        elif score == 'sigmoid_one_hot': # 各クラスのスコア（クラス毎に独立に0〜1）を求めた後，最大値のみを one-hot 形式で返す
            return y, F.one_hot(torch.max(torch.sigmoid(h), dim=1)[0], num_classes=h.size()[1])
        elif score == 'softmax_full': # 全クラスのスコア（総和が1）を返す
            return y, F.softmax(h, dim=1)
        elif score == 'softmax_only_max': # 各クラスのスコア（総和が1）を求めた後，最大値のみを返す
            return y, torch.max(F.softmax(h), dim=1)[0]
        elif score == 'softmax_one_hot': # 各クラスのスコア（総和が1）を求めた後，最大値のみを one-hot 形式で返す
            return y, F.one_hot(torch.max(F.softmax(h), dim=1)[0], num_classes=h.size()[1])
        else:
            return y
