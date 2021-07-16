import cv2
import tqdm
import pyswarms
import numpy as np


def q(pr):
    x1, y1, x2, y2 = map(int, pr)
    if x1 > x2:
        x1, x2 = x2, x1
    elif x1 == x2:
        x2 += 1
    if y1 > y2:
        y1, y2 = y2, y1
    elif y1 == y2:
        y2 += 1
    return x1, y1, x2, y2


m = {}
def 切片方和(img, roi, n, merge=True):
    # 替代 (a[roi].flatten()**n).sum()，需要保证img是不变量
    d = id(img)
    if (d, n) not in m:
        img = img**n
        s = [*img.shape]
        s[0] += 1
        s[1] += 1
        img2 = np.zeros(shape=s, dtype=img.dtype)
        img2[1:, 1:] = img
        m[d, n] = np.cumsum(np.cumsum(img2, axis=1), axis=0)
    res = _切片方和(d, roi, n)
    if merge:
        res = res.sum()
    return res


def _切片方和(d, roi, n):
    前缀和 = m[d, n]
    a, b = roi
    return 前缀和[a.stop, b.stop]-前缀和[a.start, b.stop]-前缀和[a.stop, b.start]+前缀和[a.start, b.start]


def ya(原图, 目标图, i, verbose=False):
    w, h = 原图.shape[:2]
    差值图 = 原图-目标图
    def 代价(粒子群):
        def 粒子代价(pr):
            x1, y1, x2, y2 = q(pr)
            roi = slice(x1, x2), slice(y1, y2)
            size = (x2-x1)*(y2-y1)
            t = 切片方和(原图, roi, 1, False)
            平均颜色 = t / size
            h = 平均颜色.reshape([1, 1, 3])
            原se = 切片方和(差值图, roi, 2)
            # _原se = ((原图[roi]-目标图[roi]).flatten()**2).sum()
            # print(-0.1 < 原se - _原se < 0.1)
            新se = 切片方和(原图, roi, 2) + (size*h**2).sum() - (2*t*h).sum()
            return 新se-原se
        return [*map(粒子代价, 粒子群)]
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    bounds = (
        [0, 0, 0, 0],
        [w-1, h-1, w-1, h-1],
    )
    optimizer = pyswarms.single.GlobalBestPSO(n_particles=120+i//5, dimensions=4, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(代价, iters=40+i//15, verbose=verbose)
    return best_cost, best_pos


def ride(真原图, epoch, 中间结果存储位置=None, resize=None, verbose='tqdm'):
    真原图 = 真原图.astype(np.float64)
    真原w, 真原h = 真原图.shape[:2]
    if resize:
        原图 = cv2.resize(真原图, resize, interpolation=cv2.INTER_CUBIC)
    else:
        原图 = 真原图.copy()
    目标图 = np.zeros(shape=原图.shape)
    目标图[:] = 原图.mean(axis=(0, 1)).reshape([1, 1, 3])
    块组 = []
    t = range(epoch)
    if verbose == 'tqdm':
        t = tqdm.tqdm(t)
        verbose = False
    for i in t:
        best_cost, best_pos = ya(原图, 目标图, i, verbose=verbose)
        x1, y1, x2, y2 = q(best_pos)
        roi = slice(x1, x2), slice(y1, y2)
        平均颜色 = 原图[roi].mean(axis=(0, 1))
        块组.append([[x1, y1, x2, y2], 平均颜色.astype(int)]) 
        目标图[roi] = np.array(平均颜色).reshape([1, 1, 3])
        目标图 = 目标图.copy()
        if 中间结果存储位置:
            写 = 目标图.astype(np.uint8)
            if 写.shape[:2] != (真原w, 真原h):
                写 = cv2.resize(写, (真原h, 真原w))
            cv2.imwrite(f'{中间结果存储位置}/{i}.jpg', 写)
    return 块组
