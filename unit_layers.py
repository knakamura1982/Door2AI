import torch
import torch.nn as nn
import torch.nn.functional as F


# 畳込み + バッチ正規化 + 活性化関数を行う層
# 畳込みにおけるパディングサイズとストライド幅は，出力マップが入力マップと同じサイズになるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（奇数のみ可）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う，Falseなら行わない）
#   - dropout_ratio: 0以外ならその割合でドロップアウト処理を実行
#   - activation: 活性化関数
class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(Conv, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        if do_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def __call__(self, x):
        h = self.conv(x) # 畳込み
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = nn.Dropout2d(p=self.dropout_ratio)(h) # ドロップアウト
        return h


# プーリングを行う層
#   - method: プーリング手法（'max'または'avg'のいずれか）
#   - scale: カーネルサイズおよびストライド幅
class Pool(nn.Module):

    def __init__(self, method='max', scale=2):
        super(Pool, self).__init__()
        if method == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=scale, stride=scale)
        elif method == 'max':
            self.pool = nn.MaxPool2d(kernel_size=scale, stride=scale)

    def __call__(self, x):
        return self.pool(x)


# Global Average/Max Poolingを行う層
class GlobalPool(nn.Module):

    def __init__(self, method='avg'):
        super(GlobalPool, self).__init__()
        self.method = method

    def __call__(self, x):
        size = x.size()
        if self.method == 'avg':
            pool = nn.AvgPool2d(kernel_size=(size[2], size[3]))
        elif self.method == 'max':
            pool = nn.MaxPool2d(kernel_size=(size[2], size[3]))
        h = pool(x)
        h = h.view(size[0], size[1])
        return h


# 畳込み + バッチ正規化 + 活性化関数を行う層
# 畳込みにおけるパディングサイズとストライド幅は，出力マップのサイズが入力マップのサイズの半分になるように自動決定する
# 入力マップのサイズは偶数であるものと仮定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（偶数のみ可）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う，Falseなら行わない）
#   - dropout_ratio: 0以外ならその割合でドロップアウト処理を実行
#   - activation: 活性化関数
class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(DownConv, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1)
        if do_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def __call__(self, x):
        h = self.conv(x) # 畳込み
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = nn.Dropout2d(p=self.dropout_ratio)(h) # ドロップアウト
        return h


# 逆畳込み + バッチ正規化 + 活性化関数を行う層
# 逆畳込みにおけるパディングサイズとストライド幅は，出力マップのサイズが入力マップのサイズの 2 倍になるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（偶数のみ可）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う，Falseなら行わない）
#   - dropout_ratio: 0以外ならその割合でドロップアウト処理を実行
#   - activation: 活性化関数
class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(UpConv, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1)
        if do_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def __call__(self, x):
        h = self.deconv(x) # 逆畳み込み
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = nn.Dropout2d(p=self.dropout_ratio)(h) # ドロップアウト
        return h


# 畳込み + バッチ正規化 + 活性化関数 + Pixel Shuffle を行う層
# 畳込みにおけるパディングサイズとストライド幅は，出力マップのサイズが入力マップのサイズの 2 倍になるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数
#   - kernel_size: カーネルサイズ（奇数のみ可）
#   - scale: Pixel Shuffle におけるスケールファクタ
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う，Falseなら行わない）
#   - dropout_ratio: 0以外ならその割合でドロップアウト処理を実行
#   - activation: 活性化関数
class PixShuffle(nn.Module):

    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(PixShuffle, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*scale*scale, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.ps = nn.PixelShuffle(scale)
        if do_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels*scale*scale)

    def __call__(self, x):
        h = self.conv(x) # 畳み込み
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        h = self.ps(h) # pixel shuffle
        if self.dropout_ratio != 0:
            h = nn.Dropout2d(p=self.dropout_ratio)(h) # ドロップアウト
        return h


# 全結合 + バッチ正規化 + 活性化関数を行う層
#   - in_units: 入力ユニット数
#   - out_units: 出力ユニット数
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う，Falseなら行わない）
#   - dropout_ratio: 0以外ならその割合でドロップアウト処理を実行
#   - activation: 活性化関数
class FC(nn.Module):

    def __init__(self, in_units, out_units, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(FC, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        self.fc = nn.Linear(in_features=in_units, out_features=out_units)
        if do_bn:
            self.bn = nn.BatchNorm1d(num_features=out_units)

    def __call__(self, x):
        h = self.fc(x) # 全結合
        if self.do_bn:
            h = self.bn(h) # バッチ正規化
        if not self.activation is None:
            h = self.activation(h) # 活性化関数
        if self.dropout_ratio != 0:
            h = nn.Dropout(p=self.dropout_ratio)(h) # ドロップアウト
        return h


# 平坦化を行う層
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def __call__(self, x):
        h = x.view(x.size()[0], -1)
        return h


# 特徴マップの再配置（平坦化の逆）を行う層
class Reshape(nn.Module):

    def __init__(self, size=None):
        super(Reshape, self).__init__()
        self.size = size

    def __call__(self, x):
        size = []
        size.append(x.size()[0])
        for i in range(len(self.size)):
            size.append(self.size[i])
        h = x.reshape(size)
        return h


# Plain型ResBlock層
# 畳込みにおけるパディングサイズとストライド幅は，出力マップが入力マップと同じサイズになるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数（0以下の時は自動的に in_channels と同じ値を設定）
#   - kernel_size: カーネルサイズ（奇数のみ可）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う，Falseなら行わない）
#   - dropout_ratio: 0以外ならその割合でドロップアウト処理を実行
#   - activation: 活性化関数
class ResBlockP(nn.Module):

    def __init__(self, in_channels, out_channels=0, kernel_size=3, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(ResBlockP, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        if out_channels <= 0:
            out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        if self.do_bn:
            self.bn1 = nn.BatchNorm2d(num_features=in_channels)
            self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        if in_channels != out_channels:
            self.use_resconv = True
            self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            if self.do_bn:
                self.resbn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.use_resconv = False

    def __call__(self, x):
        if self.do_bn:
            h = self.activation(self.bn1(self.conv1(x))) # 畳込み＆バッチ正規化＆活性化関数
            h = self.bn2(self.conv2(h)) # 畳込み＆バッチ正規化
        else:
            h = self.activation(self.conv1(x)) # 畳込み＆活性化関数
            h = self.conv2(h) # 畳込み
        h = self.activation(h + x) # 加算＆活性化関数
        if self.use_resconv:
            if self.do_bn:
                h = self.activation(self.resbn(self.resconv(h))) # チャンネル数を変更するための畳込み＆バッチ正規化
            else:
                h = self.activation(self.resconv(h)) # チャンネル数を変更するための畳込み
        if self.dropout_ratio != 0:
            h = nn.Dropout2d(p=self.dropout_ratio)(h) # ドロップアウト
        return h


# Bottle-Neck型ResBlock層
# 畳込みにおけるパディングサイズとストライド幅は，出力マップが入力マップと同じサイズになるように自動決定する
#   - in_channels: 入力マップのチャンネル数
#   - out_channels: 出力マップのチャンネル数（0以下の時は自動的に in_channels と同じ値を設定）
#   - mid_channels: 中間層のマップのチャンネル数（0以下の時は自動的に in_channels // 4 を設定）
#   - kernel_size: カーネルサイズ（奇数のみ可）
#   - do_bn: バッチ正規化を行うか否か（Trueなら行う，Falseなら行わない）
#   - dropout_ratio: 0以外ならその割合でドロップアウト処理を実行
#   - activation: 活性化関数
class ResBlockBN(nn.Module):

    def __init__(self, in_channels, out_channels=0, mid_channels=0, kernel_size=3, do_bn=True, dropout_ratio=0, activation=F.relu):
        super(ResBlockBN, self).__init__()
        self.do_bn = do_bn
        self.dropout_ratio = dropout_ratio
        self.activation = activation
        if out_channels <= 0:
            out_channels = in_channels
        if mid_channels <= 0:
            mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        if self.do_bn:
            self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
            self.bn2 = nn.BatchNorm2d(num_features=mid_channels)
            self.bn3 = nn.BatchNorm2d(num_features=in_channels)
        if in_channels != out_channels:
            self.use_resconv = True
            self.resconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            if self.do_bn:
                self.resbn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.use_resconv = False

    def __call__(self, x):
        if self.do_bn:
            h = self.activation(self.bn1(self.conv1(x))) # 畳込み＆バッチ正規化＆活性化関数
            h = self.activation(self.bn2(self.conv2(h))) # 畳込み＆バッチ正規化＆活性化関数
            h = self.bn3(self.conv3(h)) # 畳込み＆バッチ正規化
        else:
            h = self.activation(self.conv1(x)) # 畳込み＆活性化関数
            h = self.activation(self.conv2(h)) # 畳込み＆活性化関数
            h = self.conv3(h) # 畳込み
        h = self.activation(h + x) # 加算＆活性化関数
        if self.use_resconv:
            if self.do_bn:
                h = self.activation(self.resbn(self.resconv(h))) # チャンネル数を変更するための畳込み＆バッチ正規化
            else:
                h = self.activation(self.resconv(h)) # チャンネル数を変更するための畳込み
        if self.dropout_ratio != 0:
            h = nn.Dropout2d(p=self.dropout_ratio)(h) # ドロップアウト
        return h
