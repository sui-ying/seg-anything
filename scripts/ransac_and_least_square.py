import copy
import numpy
import numpy as np
import cv2
import math
import random
import math
from scipy import optimize


class get_corner():
    def remove_out_group(self, x_list, y_list):
        mean_x = numpy.mean(x_list)
        std_x = numpy.std(x_list) * 3
        mean_y = numpy.mean(y_list)
        std_y = numpy.std(y_list) * 3
        new_points_x = []
        new_points_y = []
        for i in range(len(x_list)):
            single_x = x_list[i]
            single_y = y_list[i]
            if (abs(single_x - mean_x) < std_x) and (abs(single_y - mean_y) < std_y):
                new_points_x.append(single_x)
                new_points_y.append(single_y)
        return new_points_x, new_points_y

    def f_1(self, x, A, B):
        return A * x + B

    def line_intersection(self, A1, B1, C1, A2, B2, C2):
        """
        l1: A1x + B1y + C1 = 0,
        l2: A2x + B2y + C2 = 0
        求解两条直线的交点
        """
        # 当直线l2为平行与y轴的直线时
        if A2 == 0:
            A2 = 1
            B2 = 0
            C2 = -C2

        d = A1 * B2 - A2 * B1
        if d == 0:
            return None  # 两条直线平行或重合，没有交点
        x = int((B1 * C2 - B2 * C1) / d)
        y = int((A2 * C1 - A1 * C2) / d)
        return [x, y]

    def fitLineRansac(self, points, iterations=1000, sigma=1.0, k_min=-1, k_max=1, flag_horizontal=True):
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
        :param flag_horizontal: True for fit horizontal line, false for fit vertical line
        :return:拟合的直线参数,It is a vector of 4 floats
                    (vx, vy, x0, y0) where (vx, vy) is a normalized
                    vector collinear to the line and (x0, y0) is some
                    point on the line.
        """
        line = [0, 0, 0, 0]
        points_num = len(points)
        if points_num < 2:
            return line

        bestScore = -1
        # bestScore = float("inf")
        sum_score = float("inf")
        for k in range(iterations):
            i1, i2 = random.sample(range(points_num), 2)
            p1 = points[i1]
            p2 = points[i2]

            dp = p1 - p2  # 直线的方向向量(x0, y0)
            dp = dp.astype(np.float32)
            dp *= 1. / np.linalg.norm(dp)  # 除以模长，进行归一化

            score = 0
            _sum_score = 0
            if dp[0] == 0:
                A, B, C = 1, 0, -p1[0]
                k = float("inf")
                # print(A, B, C, k)
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
                            score += 1

            else:
                # vertical
                if k >= k_max and k <= k_min:
                    for i in range(points_num):
                        x, y = points[i][0], points[i][1]
                        dis = abs(A * x + B * y + C) / math.sqrt(A ** 2 + B ** 2)  # 计算到直线距离
                        if dis < sigma:
                            score += 1

            if score > bestScore:
                line = [A, B, C, k]
                bestScore = score

        return line

    def ransac_line(self, points, iterations=1000, threshold=0.1):
        """
        RANSAC算法拟合直线 Ax+By+C=0
        :param points: n个点的坐标 [(x1, y1), (x2, y2), ..., (xn, yn)]
        :param iterations: 迭代次数
        :param threshold: 阈值，表示离线的误差允许范围
        :return: 返回直线(k, b)
        """
        best_k, best_b = None, None
        best_error = float('inf')

        for i in range(iterations):
            # 随机选择两个点，用于计算一条直线的系数 k 和 b
            idx1, idx2 = random.sample(range(len(points)), 2)
            x1, y1 = points[idx1]
            x2, y2 = points[idx2]

            if x2 - x1 == 0:
                continue

            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1

            # 统计误差小于阈值的点，用于计算新直线
            new_points = []
            for point in points:
                x, y = point
                error = abs(y - k * x - b)
                if error < threshold:
                    new_points.append(point)

            # 计算新直线的误差
            error = 0
            if len(new_points) > 1:
                xs = np.array([point[0] for point in new_points])
                ys = np.array([point[1] for point in new_points])
                A = np.vstack([xs, np.ones(len(xs))]).T
                k, b = np.linalg.lstsq(A, ys, rcond=None)[0]
                error = sum(abs(ys - k * xs - b))

            # 更新最佳直线
            if error < best_error:
                best_k, best_b = k, b
                best_error = error

        # # 将 Ax + By + C = 0 转换为 y = kx + b 的形式
        # k = -best_k
        # b = -best_b / best_k

        # return [best_k, best_b]
        return [best_k, -1, best_b]

    def visual_line(self, A, B, C, img, color=(255, 0, 0), flag_horizontal=True):
        """
        0: Ax + By + C =0
        1: y = kx + b
        2: x = ky + b
        :param A:
        :param B:
        :param C:
        :param flag_horizontal: True for fit horizontal line, false for fit vertical line
        :param mode: 0 for Ax + By + C =0; 1 for y = kx + b; 2 for x = ky + b
        :return:
        """
        # print(A, B, C)
        if flag_horizontal:
            p1_x = 180
            p2_x = 1800
            if B != 0:
                p1_y = -(A * p1_x + C) / B
                p2_y = -(A * p2_x + C) / B
            else:
                p1_y = -C
                p2_y = -C

        else:
            p1_y = 180
            p2_y = 1080
            if A != 0:
                p1_x = -(B * p1_y + C) / A
                p2_x = -(B * p2_y + C) / A
            else:
                p1_x = -C
                p2_x = -C

        p1 = (int(p1_x), int(p1_y))
        p2 = (int(p2_x), int(p2_y))
        cv2.line(img, p1, p2, color, 1)
        cv2.imshow("color_image", img)
        # cv2.waitKey(1000)

    def visual_line_kb(self, k, b, img, color=(255, 0, 0), flag_horizontal=True):
        p1_y = 180
        p1_x = (p1_y - b) / k
        p2_y = 1080
        p2_x = (p2_y - b) / k
        p1 = (int(p1_x), int(p1_y))
        p2 = (int(p2_x), int(p2_y))
        cv2.line(img, p1, p2, color, 2)
        cv2.imshow("color_image", img)

    def visual_line_f1(self, k, b, img, color=(255, 0, 0), flag_horizontal=True):
        """
        line: x = ky + b, for vertical line
        :param k:
        :param b:
        :param img:
        :param color:
        :param flag_horizontal:
        :return:
        """
        p1_y = 10
        p1_x = k * p1_y + b
        p2_y = 1000
        p2_x = k * p2_y + b
        p1 = (int(p1_x), int(p1_y))
        p2 = (int(p2_x), int(p2_y))
        cv2.line(img, p1, p2, color, 2)
        cv2.imshow("color_image", img)

    def get(self, mask, color_image, visual=True):
        points = numpy.argwhere(mask == 255)
        points[:, [0, 1]] = points[:, [1, 0]]
        all_x_origin = points[:, 0]
        all_y_origin = points[:, 1]
        x_min, x_max = min(all_x_origin), max(all_x_origin)
        y_min, y_max = min(all_y_origin), max(all_y_origin)
        # 每个y坐标可能对应多个x坐标，但最多不会超过x的总数
        max_x_number = (x_max - x_min) + 1
        max_y_number = (y_max - y_min) + 1

        all_x = list(set(all_x_origin))
        all_y = list(set(all_y_origin))

        # 以x为基础，搜索对应的y坐标，用于上下两条线的拟合
        up_point_y = []
        down_point_y = []
        x_points = []
        for i, single_x in enumerate(all_x):
            index = numpy.argwhere(points[:, 0] == single_x)
            search_y = points[index][:, :, 1].squeeze()
            # x_search_y_array[i, :len(search_y)] = search_y
            up_point = numpy.min(search_y)
            down_point = numpy.max(search_y)
            if abs(down_point - up_point) > max_y_number * 0.5:
                up_point_y.append(up_point)
                down_point_y.append(down_point)
                x_points.append(single_x)
        up_points_x, up_points_y = self.remove_out_group(x_points, up_point_y)
        down_point_x, down_point_y = self.remove_out_group(x_points, down_point_y)

        up_sample_points, down_sample_points = [], []
        # up_points
        for i in range(len(up_points_x)):
            up_sample_points.append([up_points_x[i], up_points_y[i]])
        up_sample_points = numpy.array(up_sample_points)
        if visual:
            for point in up_sample_points:
                cv2.circle(color_image, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
            cv2.imshow("color_image", color_image)
            cv2.waitKey(100)
        [A_up, B_up, C_up, k_up] = self.fitLineRansac(up_sample_points, 1000, 3, -1, 1)

        # down_points
        for i in range(len(down_point_x)):
            down_sample_points.append([down_point_x[i], down_point_y[i]])
        down_sample_points = numpy.array(down_sample_points)
        if visual:
            for point in down_sample_points:
                cv2.circle(color_image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
            cv2.imshow("color_image", color_image)
            cv2.waitKey(100)
        [A_down, B_down, C_down, k_down] = self.fitLineRansac(down_sample_points, 1000, 3, -1, 1)

        if visual:
            self.visual_line(A_up, B_up, C_up, color_image, color=(255, 0, 0))
            self.visual_line(A_down, B_down, C_down, color_image, color=(0, 255, 0))

        left_point_x, left_points_y = [], []
        right_point_x, right_point_y = [], []
        distance_thre = 3
        # 以y为基础，搜索对应的x坐标，用于左右两条线的拟合
        for i, single_y in enumerate(all_y):
            index = numpy.argwhere(points[:, 1] == single_y)
            search_x = points[index][:, :, 0].squeeze()
            # x_search_y_array[i, :len(search_y)] = search_y
            left_point = numpy.min(search_x)
            right_point = numpy.max(search_x)

            d1_up = abs(A_up * left_point + B_up * single_y + C_up) / math.sqrt(A_up ** 2 + B_up ** 2)
            d1_down = abs(A_down * left_point + B_down * single_y + C_down) / math.sqrt(A_down ** 2 + B_down ** 2)
            d2_up = abs(A_up * right_point + B_up * single_y + C_up) / math.sqrt(A_up ** 2 + B_up ** 2)
            d2_down = abs(A_down * right_point + B_down * single_y + C_down) / math.sqrt(A_down ** 2 + B_down ** 2)

            if d1_up > distance_thre and d1_down > distance_thre:
                left_point_x.append(left_point)
                left_points_y.append(single_y)
            if d2_up > distance_thre and d2_down > distance_thre:
                right_point_x.append(right_point)
                right_point_y.append(single_y)

        left_point_x, left_point_y = self.remove_out_group(left_point_x, left_points_y)
        right_point_x, right_point_y = self.remove_out_group(right_point_x, right_point_y)

        left_sample_points, right_sample_points = [], []
        # left_points
        for i in range(len(left_point_x)):
            left_sample_points.append([left_point_x[i], left_points_y[i]])
        left_sample_points = numpy.array(left_sample_points)
        if visual:
            for point in left_sample_points:
                cv2.circle(color_image, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
            cv2.imshow("color_image", color_image)
            cv2.waitKey(100)

        # [A_left, B_left, C_left, k_left] = self.fitLineRansac(left_sample_points, 1000, 10, -4, 4, False)
        # [A_left, B_left, C_left] = self.ransac_line(left_sample_points)
        k_left, b_left = optimize.curve_fit(self.f_1, left_point_y, left_point_x)[0]  # x = k1 * y + b1

        # right_points
        for i in range(len(right_point_x)):
            right_sample_points.append([right_point_x[i], right_point_y[i]])
        right_sample_points = numpy.array(right_sample_points)
        if visual:
            for point in right_sample_points:
                cv2.circle(color_image, (int(point[0]), int(point[1])), 1, (255, 0, 255), -1)
            cv2.imshow("color_image", color_image)
            cv2.waitKey(100)
        k_right, b_right = optimize.curve_fit(self.f_1, right_point_y, right_point_x)[0]

        if visual:
            self.visual_line_f1(k_left, b_left, color_image, color=(0, 0, 255), flag_horizontal=False)
            self.visual_line_f1(k_right, b_right, color_image, color=(255, 0, 255), flag_horizontal=False)

        # up-down: Ax + By + C = 0 ; left_right: x = ky + b
        up_left = self.line_intersection(A_up, B_up, C_up, 1, -k_left, -b_left)
        up_right = self.line_intersection(A_up, B_up, C_up, 1, -k_right, -b_right)
        down_right = self.line_intersection(A_down, B_down, C_down, 1, -k_right, -b_right)
        down_left = self.line_intersection(A_down, B_down, C_down, 1, -k_left, -b_left)

        if visual:
            cv2.circle(color_image, (int(up_left[0]), int(up_left[1])), 4, (0, 255, 255), -1)
            cv2.circle(color_image, (int(up_right[0]), int(up_right[1])), 4, (0, 255, 255), -1)
            cv2.circle(color_image, (int(down_left[0]), int(down_left[1])), 4, (0, 255, 255), -1)
            cv2.circle(color_image, (int(down_right[0]), int(down_right[1])), 4, (0, 255, 255), -1)
            cv2.imshow('color_image', color_image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        return [up_left, up_right, down_right, down_left]


if __name__ == '__main__':

    # mask_path = "/home/westwell/Downloads/tmp/西井科技20230512-142710.jpg"
    # img_path = '/home/westwell/Downloads/tmp/西井科技20230512-142710.jpg'


    # img_path = "/cv/xyc/segment-anything/test_img/plate_0_plate_number_0_Left_8581.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/plate_0_plate_number_0_Left_8581/plate_0_plate_number_0_Left_8581_7_mask.jpg"

    img_path = "/cv/xyc/segment-anything/test_img/8_330.jpg"
    mask_path = "/cv/xyc/segment-anything/output/8_330/8_330_5_mask.jpg"

    # img_path = "/cv/all_training_data/plate/cn/qqctu/dataset_6nc/images/test/2_62.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/2_62/2_62_6_mask.jpg"

    # img_path = "../notebooks/images/car.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/car/car_11_mask.jpg"

    # mask_path = "/home/westwell/Downloads/tmp/西井科技20230512-142710.jpg"
    mask = cv2.imread(mask_path, 0)
    # img = cv2.imread(img_path, 1)

    get_corner_cunction = get_corner()
    for i in range(10):
        img = cv2.imread(img_path, 1)
        [left_up, left_right, down_right, down_left] = get_corner_cunction.get(mask, img, visual=True)
        print([left_up, left_right, down_right, down_left])
