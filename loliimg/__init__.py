import concurrent.futures

import cv2
import pyswarms
import numpy as np


def q(x1, y1, x2, y2):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if x1 == x2:
        x2 += 1
    if y1 > y2:
        y1, y2 = y2, y1
    if y1 == y2:
        y2 += 1
    return x1, y1, x2, y2


def ya(原图, 目标图, i, verbose=True):
    w, h = 原图.shape[:2]
    tp = concurrent.futures.ThreadPoolExecutor(max_workers=20)
    def 代价(粒子群):
        def 粒子代价(pr):
            x1, y1, x2, y2 = q(*pr)
            roi = slice(x1, x2), slice(y1, y2)
            平均颜色 = 原图[roi].mean(axis=(0, 1))
            幻图 = 原图[roi] - np.array(平均颜色).reshape([1, 1, 3])
            原se = ((原图[roi]-目标图[roi]).flatten()**2).sum()
            新se = (幻图.flatten()**2).sum()
            return 新se-原se
        return list(tp.map(粒子代价, 粒子群))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    bounds = (
        [0, 0, 0, 0],
        [w-1, h-1, w-1, h-1],
    )
    optimizer = pyswarms.single.GlobalBestPSO(n_particles=int(140+i/2), dimensions=4, options=options, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(代价, iters=int(70+i/4), verbose=verbose)
    return best_cost, best_pos


def ride(真原图, epoch, 中间结果存储位置=None, verbose=True):
    真原图 = 真原图.astype(np.float64)
    真原w, 真原h = 真原图.shape[:2]
    原图 = cv2.resize(真原图, (256, 256), interpolation=cv2.INTER_CUBIC)
    目标图 = np.zeros(shape=原图.shape)
    目标图[:] = 原图.mean(axis=(0, 1)).reshape([1, 1, 3])
    块组 = []
    for i in range(epoch):
        best_cost, best_pos = ya(原图, 目标图, i, verbose=verbose)
        x1, y1, x2, y2 = q(*best_pos)
        roi = slice(x1, x2), slice(y1, y2)
        平均颜色 = 原图[roi].mean(axis=(0, 1))
        块组.append([[x1, y1, x2, y2], 平均颜色.astype(int)])
        幻图 = 原图[roi] - np.array(平均颜色).reshape([1, 1, 3])
        目标图[roi] = np.array(平均颜色).reshape([1, 1, 3])
        if 中间结果存储位置:
            写 = 目标图.astype(np.uint8)
            写 = cv2.resize(写, (真原h, 真原w), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f'{中间结果存储位置}/{i}.png', 写)
    return 块组
