import numpy as np


# 二次元配列 img に対し zero crossing を求める
# zero crossing なピクセルを 0，そうでないピクセルを 1 としたマスク画像を返す
#   - img: 入力画像（二次元の numpy.ndarray，signedであれば型制限なし）
#   - 戻値: マスク画像（二次元の numpy.ndarray，uint8，入力画像と同じサイズ）
def get_zero_crossing(img):
    h = img.shape[0]
    w = img.shape[1]
    s = np.sign(img).astype(np.int32)
    v1 = np.ones((h, 1), dtype=np.int32)
    v2 = np.ones((1, w), dtype=np.int32)
    v3 = np.ones((h-1, 1), dtype=np.int32)
    mul1 = np.append(s[:,:w-1] * s[:,1:], v1, axis=1) # 右との積
    mul2 = np.append(s[:h-1,:] * s[1:,:], v2, axis=0) # 下との積
    mul3 = np.append(np.append(v3, s[:h-1,1:] * s[1:,:w-1], axis=1), v2, axis=0) # 左斜め下との積
    mul4 = np.append(np.append(s[:h-1,:w-1] * s[1:,1:], v3, axis=1), v2, axis=0) # 右斜め下との積
    mul1 = (mul1 + 1) // 2
    mul2 = (mul2 + 1) // 2
    mul3 = (mul3 + 1) // 2
    mul4 = (mul4 + 1) // 2
    return (mul1 * mul2 * mul3 * mul4).astype(np.uint8)


# グレースケール画像 img をグラデーション画像に変換する
#   - img: 入力画像（二次元の numpy.ndarray，uint8 または 0～1 の float32）
#   - b_color: 黒をこの色に変換する（3要素のリスト，各要素は0～255の整数）
#   - w_color: 白をこの色に変換する（3要素のリスト，各要素は0～255の整数）
#   - 戻値: グラデーション画像（三次元の numpy.ndarray，uint8，画像サイズは入力と同じ）
def cvt_gray2gradation(img, b_color, w_color):
    h = img.shape[0]
    w = img.shape[1]
    if img.dtype == np.uint8:
        n = img.astype(np.float32) / 255
    else:
        n = img.copy()
    r = (n * w_color[0] + (1 - n) * b_color[0]).astype(np.uint8).reshape((h, w, 1))
    g = (n * w_color[1] + (1 - n) * b_color[1]).astype(np.uint8).reshape((h, w, 1))
    b = (n * w_color[2] + (1 - n) * b_color[2]).astype(np.uint8).reshape((h, w, 1))
    return np.concatenate([r, g, b], axis=2)


# カラー画像 img に対しマスク画像 mask を適用する
# マスク画像の値が 0 のピクセルはカラー画像でも (0, 0, 0) となるようにする
#   - img: 入力画像（三次元の numpy.ndarray，uint8）
#   - mask: マスク画像（二次元の numpy.ndarray，uint8，画像サイズは入力と同じ）
#   - 戻値: マスク適用後の画像（三次元の numpy.ndarray，uint8，入力画像と同じサイズ）
def apply_mask(img, mask):
    temp = img.transpose(2, 0, 1)
    temp[0] *= mask
    temp[1] *= mask
    temp[2] *= mask
    return temp.transpose(1, 2, 0)
