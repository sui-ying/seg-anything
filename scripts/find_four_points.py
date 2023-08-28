import cv2
import numpy
import numpy as np
import random
import math

from sklearn.cluster import KMeans
from scipy import optimize

def f_1(x, A, B):
    return A * x + B

def find_four_points(points):

    four_points = []    # Four points clockwise
    all_points = []
    # [x1, y1]
    # 使用 argmin 找到第一列中最小值所在的位置
    min_x_index = np.argmin(points[:, 0])
    # 打印所有 x 坐标等于最小值的行: 最左边的所有点
    min_x_rows = points[points[:, 0] == points[min_x_index, 0]]
    # # print("x坐标最小的点为：", min_x_rows)
    # # 取中点
    # x1y1_median = np.median(min_x_rows, axis=0)
    # # 构造中间点坐标
    # x1y1 = [int(x1y1_median[0]), int(x1y1_median[1])]
    # print("x1y1: ", x1y1)
    # four_points.append(x1y1)
    all_points.extend(min_x_rows.tolist())

    four_points.append([np.min(min_x_rows[:, 0]), np.max(min_x_rows[:, 1])])

    # [x2, y2]
    # 找到 y 坐标最小的点所在的行
    min_y_rows = points[points[:, 1] == np.min(points[:, 1])]
    # # print("y 坐标最小的点为：", min_y_rows)
    # # 取中点
    # x2y2_median = np.median(min_y_rows, axis=0)
    # # 构造中间点坐标
    # x2y2 = [int(x2y2_median[0]), int(x2y2_median[1])]
    # print("x2y2: ", x2y2)
    # four_points.append(x2y2)
    all_points.extend(min_y_rows.tolist())
    four_points.append([np.min(min_y_rows[:, 0]), np.max(min_y_rows[:, 1])])

    # [x3, y3]
    # 使用 argmax 找到第一列中最小值所在的位置
    max_x_index = np.argmax(points[:, 0])
    # 打印所有 x 坐标等于最小值的行: 最左边的所有点
    max_x_rows = points[points[:, 0] == points[max_x_index, 0]]
    # # print("x坐标最大的点为：", max_x_rows)
    # # 取中点
    # x3y3_median = np.median(max_x_rows, axis=0)
    # # 构造中间点坐标
    # x3y3 = [int(x3y3_median[0]), int(x3y3_median[1])]
    # print("x3y3: ", x3y3)
    # four_points.append(x3y3)
    all_points.extend(max_x_rows.tolist())
    four_points.append([np.min(max_x_rows[:, 0]), np.min(max_x_rows[:, 1])])

    # [x4, y4]
    # 找到y坐标最大的点所在的行
    max_y_rows = points[points[:, 1] == np.max(points[:, 1])]
    # # print("y坐标最大的点为：", max_y_rows)
    # # 取中点
    # x4y4_median = np.median(max_y_rows, axis=0)
    # # 构造中间点坐标
    # x4y4 = [int(x4y4_median[0]), int(x4y4_median[1])]
    # print("x4y4: ", x4y4)
    # four_points.append(x4y4)
    all_points.extend(max_y_rows)
    four_points.append([np.max(max_y_rows[:, 0]), np.max(max_y_rows[:, 1])])

    return four_points, all_points


def get_sample_points(polygon, min_x, max_x, min_y, max_y, iter_x=10, iter_y=5, sigma=1e-6):
    # 定义采样点间隔和高斯模糊的标准差
    delta_x = iter_x
    delta_y = int((max_y - min_y) / iter_y)
    sigma_x = delta_x * 0.4
    sigma_y = delta_y * 0.4

    # 根据采样点间隔和轮廓的 x、y 坐标最小值和最大值计算采样点的位置
    sample_x = np.arange(min_x, max_x, delta_x)
    sample_y = np.arange(min_y, max_y, delta_y)
    mesh_x, mesh_y = np.meshgrid(sample_x, sample_y)
    sample = np.hstack((mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1))).astype(np.float32)

    # 将轮廓上的点按照 x 或 y 坐标进行分类
    x_points, y_points = [], []
    for p in polygon:
        p = p[0]
        if p[0] in sample_x:
            x_points.append(p.tolist())
        if p[1] in sample_y:
            y_points.append(p.tolist())

    # 对每一列和每一行的点集合，选择纵坐标最小的点作为上采样点，
    # 选择纵坐标最大的点作为下采样点，得到一组上采样点和下采样点
    up_sample_points, down_sample_points = [], []
    for x in sample_x:
        col_points = [p for p in x_points if p[0] == x]
        if not col_points:
            up_sample_points.append([x, min_y - sigma])
            down_sample_points.append([x, max_y + sigma])
        else:
            col_points.sort(key=lambda p: p[1])
            up_sample_points.append(col_points[0])
            down_sample_points.append(col_points[-1])

    for y in sample_y:
        row_points = [p for p in y_points if p[1] == y]
        if not row_points:
            up_sample_points.append([min_x - sigma, y])
            down_sample_points.append([max_x + sigma, y])
        else:
            row_points.sort(key=lambda p: p[0])
            up_sample_points.append(row_points[0])
            down_sample_points.append(row_points[-1])

    return up_sample_points, down_sample_points

def fitLineRansac(points,iterations=1000,sigma=1.0,k_min=-7,k_max=7, h_or_v=True):
    """
    RANSAC 拟合2D 直线
    :param points:输入点集,numpy [points_num,1,2],np.float32
    :param iterations:迭代次数
    :param sigma:数据和模型之间可接受的差值,车道线像素宽带一般为10左右
                （Parameter use to compute the fitting score）
    :param k_min:
    :param k_max:k_min/k_max--拟合的直线斜率的取值范围.
                考虑到左右车道线在图像中的斜率位于一定范围内，
                添加此参数，同时可以避免检测垂线和水平线
    :return:拟合的直线参数,It is a vector of 4 floats
                (vx, vy, x0, y0) where (vx, vy) is a normalized
                vector collinear to the line and (x0, y0) is some
                point on the line.
    """
    line = [0,0,0,0]
    # points_num = points.shape[0]
    points_num = len(points)
    print("points number: ", points_num)
    if points_num < 2:
        return line

    bestScore = -1
    for k in range(iterations):
        i1,i2 = random.sample(range(points_num), 2)
        p1 = points[i1]
        p2 = points[i2]

        dp = p1 - p2  # 直线的方向向量
        dp = dp.astype(np.float32)
        dp *= 1. / np.linalg.norm(dp)  # 除以模长，进行归一化
        score = 0
        a = dp[1]/dp[0]
        if h_or_v:
            # horizontal
            if a <= k_max and a>=k_min:
                for i in range(points_num):
                    v = points[i] - p1
                    dis = v[1]*dp[0] - v[0]*dp[1]  # 向量a与b叉乘/向量b的摸.||b||=1./norm(dp)
                    # score += math.exp(-0.5*dis*dis/(sigma*sigma))误差定义方式的一种
                    if math.fabs(dis) < sigma:
                        score += 1
        else:
            # vertical
            if a >= k_max and a<=k_min:
                for i in range(points_num):
                    v = points[i] - p1
                    dis = v[1]*dp[0] - v[0]*dp[1]  # 向量a与b叉乘/向量b的摸.||b||=1./norm(dp)
                    # score += math.exp(-0.5*dis*dis/(sigma*sigma))误差定义方式的一种
                    if math.fabs(dis) < sigma:
                        score += 1
        if score > bestScore:
            line = [dp[0], dp[1], p1[0], p1[1]]
            bestScore = score

    return line

def ransac(points, n, k, t, d):
    """
    :param points: 数据集，每个元素为一个二元组 (x, y)
    :param n: 迭代次数
    :param k: 每次迭代随机选取的样本数
    :param t: 阈值，用于决策样本点是否适合作为模型内点
    :param d: 模型参数个数
    :return:
    """
    best_model = None  # 最优模型参数
    best_consensus_set = []  # 最优模型对应的内点集合
    best_consensus_num = 0  # 最优模型对应的内点个数
    for i in range(n):
        sample_index = random.sample(range(len(points)), k)  # 随机选出 k 个点
        maybe_inliers = [points[index] for index in sample_index]
        maybe_model = fit_line(maybe_inliers)  # 用随机选出的 k 个点拟合出一条直线
        consensus_set = []  # 计算新模型的内点集合
        for j, point in enumerate(points):
            if j not in sample_index:
                if distance(point, maybe_model) < t:
                    consensus_set.append(point)
        if len(consensus_set) > d and len(consensus_set) > best_consensus_num:
            best_model = fit_line(consensus_set)
            best_consensus_set = consensus_set
            best_consensus_num = len(consensus_set)
    return best_model, best_consensus_set

def fit_line(points):
    """
    :param points: 数据集，每个元素为一个二元组 (x, y)
    :return: 直线参数 [k, b]，其中 k 为斜率，b 为截距
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return [k, b]

def distance(point, line):
    """
    :param point: 样本点 [x, y]
    :param line: 直线参数 [k, b]
    :return: 点到直线的距离
    """
    return abs((line[0] * point[0]) - point[1] + line[1]) / np.sqrt(line[0] ** 2 + 1)


def line_intersection(a1, b1, c1, a2, b2, c2):
    """求解两条直线的交点"""
    d = a1 * b2 - a2 * b1
    if d == 0:
        return None  # 两条直线平行或重合，没有交点
    x = int((b1 * c2 - b2 * c1) / d)
    y = int((a2 * c1 - a1 * c2) / d)
    return [x, y]


def visual_line(k, b, img, color=(255, 0, 0)):
    """
    x = ky + b
    :param k:
    :param b:
    :return:
    """
    p1_y = 720
    p1_x = p1_y * k+b
    p2_y = 360
    p2_x = p2_y * k+b
    p1 = (int(p1_x), int(p1_y))
    p2 = (int(p2_x), int(p2_y))
    cv2.line(img, p1, p2, color, 2)
    cv2.imshow("Four Points", img)
    # cv2.waitKey(5000)

def find_four_sample_points(img, points, min_x, min_y, max_x, max_y, visual=False):

    x_interval = 2 or (max_x - min_x) // 40
    y_interval = 1 or (max_y - min_y) // 30
    x_samples = set()
    y_samples = set()
    if (x_interval > 0 and y_interval > 0):
        sample = min_x
        while (sample < max_x):
            x_samples.add(sample)
            sample += x_interval
        sample = min_y
        while (sample < max_y):
            y_samples.add(sample)
            sample += y_interval

    ave_y = (max_y - min_y) // 2
    ave_x = (max_x - min_x) * 0.9

    x_sampled_points = {}
    y_sampled_points = {}
    for point in points:
        x, y = point
        if (x in x_samples):
            if (x not in x_sampled_points):
                x_sampled_points[x] = []
            x_sampled_points[x].append(point)
        if (y in y_samples):
            if (y not in y_sampled_points):
                y_sampled_points[y] = []
            y_sampled_points[y].append(point)

    up_sample_points, down_sample_points = [], []
    up_sample_x, up_sample_y = [], []
    down_sample_x, down_sample_y = [], []
    for key, value in x_sampled_points.items():
        min_y_point = float("inf")
        max_y_point = 0
        min_y_index = 0
        max_y_index = 0
        if len(value) < 2:
            continue
        for i in range(len(value)):
            _x, _y = value[i]
            if _y < min_y_point:
                min_y_point = _y
                # min_y_index = i
            if _y > max_y_point:
                max_y_point = _y
                # max_y_index = i
        if max_y_point - min_y_point > ave_y:
            up_sample_x.append(key)
            down_sample_x.append(key)
            up_sample_y.append(min_y_point)
            down_sample_y.append(max_y_point)
            # up_sample_points.append(value[min_y_index])
            # down_sample_points.append(value[max_y_index])

    # for i in range(len(up_sample_x)):
    #     up_sample_points.append([up_sample_x[i], up_sample_y[i]])
    # up_sample_points = numpy.array(up_sample_points)
    #
    # for point in up_sample_points:
    #     cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 255, 255), -1)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # [vx, vy, x, y] = fitLineRansac(up_sample_points, 1000, 5, -1, 1)
    # k1 = float(vy) / float(vx)  # 直线斜率
    # b1 = -k1 * x + y
    #
    # p1_x = 720
    # p1_y = p1_x * k1 + b1
    #
    # p2_x = 1440
    # p2_y = p2_x * k1 + b1
    # p1 = (int(p1_x), int(p1_y))
    # p2 = (int(p2_x), int(p2_y))
    # cv2.line(img, p1, p2, (0, 0, 255), 2)
    # cv2.imshow("Four Points", img)
    # cv2.waitKey(5000)


    k1, b1 = optimize.curve_fit(f_1, up_sample_y, up_sample_x)[0]   # x = k1 * y + b1
    k2, b2 = optimize.curve_fit(f_1, down_sample_y, down_sample_x)[0]
    if visual:
        visual_line(k1, b1, img, color=(255, 0, 0))
        visual_line(k2, b2, img, color=(0, 255, 0))

    left_sample_points, right_sample_points = [], []
    left_sample_x, left_sample_y = [], []
    right_sample_x, right_sample_y = [], []
    for key, value in y_sampled_points.items():
        min_x_point = float("inf")
        max_x_point = 0
        min_x_index = 0
        max_x_index = 0
        # if len(value) < 2:
        #     continue
        for i in range(len(value)):
            _x, _y = value[i]
            if _x < min_x_point:
                min_x_point = _x
                min_x_index = i
            if _x > max_x_point:
                max_x_point = _x
                max_x_index = i

        # cv2.circle(img, (int(min_x_point), int(key)), 1, (0, 255, 0), -1)
        # cv2.circle(img, (int(max_x_point), int(key)), 1, (0, 0, 255), -1)
        # cv2.imshow("img", img)

        d1_up = abs(-1*min_x_point+ k1 * key + b1) / math.sqrt((k1 ** 2) + 1)
        d1_down = abs(-1*min_x_point+ k2 * key + b2) / math.sqrt((k2 ** 2) + 1)
        d2_up = abs(-1*max_x_point+ k1 * key + b1) / math.sqrt((k1 ** 2) + 1)
        d2_down = abs(-1*max_x_point+ k2 * key + b2) / math.sqrt((k2 ** 2) + 1)

        distance_thre = 10
        if d1_up > distance_thre and d1_down > distance_thre:
            left_sample_x.append(min_x_point)
            left_sample_y.append(key)
            # left_sample_points.append(value[min_x_index])
            # cv2.circle(img, (int(min_x_point), int(key)), 1, (0, 255, 0), -1)
            # cv2.imshow("img", img)
            # cv2.waitKey()
        if d2_up > distance_thre and d2_down > distance_thre:
            # right_sample_points.append(value[max_x_index])
            right_sample_x.append(max_x_point)
            right_sample_y.append(key)
        #     cv2.circle(img, (int(max_x_point), int(key)), 1, (0, 0, 255), -1)
        #     cv2.imshow("img", img)
        #     cv2.waitKey()


    k3, b3 = optimize.curve_fit(f_1, left_sample_y, left_sample_x)[0]
    k4, b4 = optimize.curve_fit(f_1, right_sample_y, right_sample_x)[0]
    if visual:
        visual_line(k3, b3, img, color=(0, 0, 255))
        visual_line(k4, b4, img, color=(255, 255, 0))


    # p12 = line_intersection(1, -k1, -b1, 1, -k2, -b2)
    up_left = line_intersection(1, -k1, -b1, 1, -k3, -b3)
    up_right = line_intersection(1, -k1, -b1, 1, -k4, -b4)
    down_left = line_intersection(1, -k2, -b2, 1, -k3, -b3)
    down_right = line_intersection(1, -k2, -b2, 1, -k4, -b4)
    # p34 = line_intersection(a3, b3, c3, a4, b4, c4)
    if visual:
        cv2.circle(img, (int(up_left[0]), int(up_left[1])), 4, (0, 255, 255), -1)
        cv2.circle(img, (int(up_right[0]), int(up_right[1])), 4, (0, 255, 255), -1)
        cv2.circle(img, (int(down_left[0]), int(down_left[1])), 4, (0, 255, 255), -1)
        cv2.circle(img, (int(down_right[0]), int(down_right[1])), 4, (0, 255, 255), -1)

        cv2.imshow('Four Points', img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    return [up_left, up_right, down_right, down_left]



if __name__ == '__main__':

    npy_path = "/cv/xyc/segment-anything/output/plate_0_plate_number_0_Left_8581/plate_0_plate_number_0_Left_8581.npy"

    # img_path = "/cv/xyc/segment-anything/test_img/plate_0_plate_number_0_Left_8581.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/plate_0_plate_number_0_Left_8581/plate_0_plate_number_0_Left_8581_7_mask.jpg"

    img_path = "/cv/xyc/segment-anything/test_img/8_330.jpg"
    mask_path = "/cv/xyc/segment-anything/output/8_330/8_330_5_mask.jpg"

    # img_path = "/cv/all_training_data/plate/cn/qqctu/dataset_6nc/images/test/2_62.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/2_62/2_62_6_mask.jpg"

    # img_path = "../notebooks/images/car.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/car/car_11_mask.jpg"

    visual = True

    img = cv2.imread(img_path, -1)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    points = numpy.argwhere(mask==255)
    points[:, [0, 1]] = points[:, [1, 0]]

    all_x = points[:,0]
    all_y = points[:,1]

    if visual and False:
        for point in points:
            x, y = point
            cv2.circle(img, (int(x), int(y)), 1, (155, 155, 155), -1)
            # 显示结果
        cv2.imshow('Four Points', img)
        cv2.waitKey(1000)
        # cv2.destroyAllWindows()

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    for i in range(10):
        four_points = find_four_sample_points(img, points, x_min, y_min, x_max, y_max, visual=True)
    # print(four_points)
