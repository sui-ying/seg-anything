import numpy as np
import cv2
import random
import math

def ransac_line(points, iterations=1000, sigma=1.0, k_min=-1, k_max=1, flag_horizontal=True, show=False):
    """
    RANSAC算法拟合直线 Ax+By+C=0
    :param points: n个点的坐标 [(x1, y1), (x2, y2), ..., (xn, yn)]
    :param iterations: 迭代次数
    :param threshold: 阈值，表示离线的误差允许范围
    :return: 返回直线(k, b)
    """
    line = [0, 0, 0]
    points_num = len(points)
    if points_num < 2:
        print("point num < 2")
        return line

    bestScore = -1
    # bestScore = float("inf")
    for k in range(iterations):
        new_points = []
        i1, i2 = random.sample(range(points_num), 2)
        p1 = points[i1]
        p2 = points[i2]

        dp = p1 - p2  # 直线的方向向量(x0, y0)
        dp = dp.astype(np.float32)
        dp *= 1. / np.linalg.norm(dp)  # 除以模长，进行归一化
        score = 0
        _sum_score = 0
        if dp[0] == 0:
            # A, B, C = 1, 0, -p1[0]
            # k = float("inf")
            # # print(A, B, C, k)
            continue
        else:
            A = dp[1] / dp[0]
            B = -1
            C = p1[1] - A * p1[0]
            k = -A / B
        if flag_horizontal:
            # horizontal
            if k <= k_max and k >= k_min:
                for i in range(points_num):
                    x, y = points[i][0], points[i][1]
                    dis = abs(A * x + B * y + C) / math.sqrt(A ** 2 + B ** 2)  # 计算到直线距离
                    if dis < sigma:
                        new_points.append(points[i])
                        score += 1

        else:
            # vertical
            if k >= k_max or k <= k_min:
                for i in range(points_num):
                    x, y = points[i][0], points[i][1]
                    dis = abs(A * x + B * y + C) / math.sqrt(A ** 2 + B ** 2)  # 计算到直线距离
                    if show:
                        print(dis)
                    if dis < sigma:
                        new_points.append(points[i])
                        score += 1
                # print("len(new_points):", len(new_points), score)

        if score > bestScore:
            bestScore = score
            if len(new_points) > 1:
                xs = np.array([point[0] for point in new_points])
                # print(len(xs), len(points),  len(xs)/ len(points))
                ys = np.array([point[1] for point in new_points])
                A = np.vstack([xs, np.ones(len(xs))]).T
                k, b = np.linalg.lstsq(A, ys, rcond=None)[0]
                error = sum(abs(ys - k * xs - b))
                line = [k, -1, b]
                # print(line)
    return line



aa = np.array([[648, 397],
 [648, 398],
 [648, 399],
 [648, 400],
 [648, 401],
 [648, 402],
 [648, 403],
 [648, 404],
 [648, 405],
 [648, 406],
 [648, 407],
 [648, 408],
 [648, 409],
 [648, 410],
 [648, 411],
 [648, 412],
 [648, 413],
 [648, 414],
 [648, 415],
 [648, 416],
 [648, 417],
 [648, 418],
 [649, 419],
 [649, 420],
 [649, 421],
 [649, 422],
 [649, 423],
 [649, 424],
 [649, 425],
 [649, 426],
 [649, 427],
 [649, 428],
 [649, 429],
 [649, 430],
 [649, 431],
 [649, 432],
 [650, 433],
 [650, 434],
 [650, 435],
 [650, 436],
 [650, 437],
 [650, 438],
 [650, 439],
 [650, 440],
 [650, 441],
 [650, 442],
 [650, 443],
 [650, 444],
 [650, 445],
 [650, 446],
 [650, 447],
 [650, 448],
 [650, 449],
 [650, 450],
 [650, 451],
 [650, 452],
 [650, 453],
 [650, 454],
 [650, 455],
 [650, 456],
 [650, 457],
 [650, 458],
 [650, 459],
 [650, 460],
 [650, 461],
 [650, 462],
 [650, 463],
 [650, 464],
 [650, 465],
 [650, 466],
 [650, 467],
 [650, 468],
 [650, 469],
 [650, 470],
 [650, 471],
 [650, 472],
 [650, 473],
 [650, 474],
 [650, 475],
 [650, 476],
 [651, 477],
 [651, 478],
 [651, 479],
 [651, 480],
 [651, 481],
 [651, 482],
 [651, 483],
 [651, 484],
 [651, 485],
 [651, 486],
 [651, 487],
 [651, 488],
 [651, 489],
 [651, 490],
 [651, 491],
 [651, 492],
 [651, 493],
 [651, 494],
 [651, 495],
 [651, 496],
 [651, 497],
 [651, 498],
 [651, 499],
 [651, 500],
 [651, 501],
 [651, 502],
 [651, 503],
 [651, 504],
 [651, 505],
 [651, 506],
 [651, 507],
 [651, 508]])


# for i in range(5):
#     [A_right, B_right, C_right] = ransac_line(aa, 1000, 20, -1, 1, flag_horizontal=False)
#     print([A_right, B_right, C_right])

img_path = "/cv/all_training_data/plate/cn/yolov7_plate/dataset/all/1649477086000.jpg"
mask_path = "/cv/xyc/segment-anything/output/1649459708000/1649459708000_12_mask.jpg"
import cv2
import numpy as np

# 读取原始图像并转换为灰度图像
img = cv2.imread(mask_path)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


