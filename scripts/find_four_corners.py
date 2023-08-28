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
        if len(x_list) < 10:
            return x_list, y_list
        mean_x = numpy.mean(x_list)
        std_x = numpy.std(x_list) * 3
        mean_y = numpy.mean(y_list)
        std_y = numpy.std(y_list) * 3
        new_points_x = []
        new_points_y = []
        for i in range(len(x_list)):
            single_x = x_list[i]
            single_y = y_list[i]
            if (std_x == 0 or abs(single_x - mean_x) <= std_x) and (std_y == 0 or abs(single_y - mean_y) <= std_y):
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

    def ransac_line(self, points, iterations=1000, sigma=1.0, k_min=-1, k_max=1, flag_horizontal=True, show=False):
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
                    # print("len(new_points):", len(new_points))

            if score > bestScore:
                bestScore = score
                if len(new_points) > 1:
                    xs = np.array([point[0] for point in new_points])
                    # print(len(xs), len(points),  len(xs)/ len(points))
                    ys = np.array([point[1] for point in new_points])
                    if show:
                        for jj in range(len(xs)):
                            cv2.circle(self.color_image, (int(xs[jj]), int(ys[jj])), 3, (255, 0, 0), -1)
                    A = np.vstack([xs, np.ones(len(xs))]).T
                    k, b = np.linalg.lstsq(A, ys, rcond=None)[0]
                    error = sum(abs(ys - k * xs - b))
                    line = [k, -1, b]

        return line

    def visual_line(self, A, B, C, img, color=(255, 0, 0), flag_horizontal=True):
        """
        Ax + By + C =0
        :param A:
        :param B:
        :param C:
        :param flag_horizontal: True for fit horizontal line, false for fit vertical line
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

    def _mask_erosion(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imagemask = cv2.erode(self.mask, kernel, iterations=1)  # 进行腐蚀操作
        contours, hierarchy = cv2.findContours(imagemask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        imagemask = numpy.zeros(self.mask.shape, numpy.uint8)
        cv2.drawContours(imagemask, [max_contour], -1, 255, -1)
        self.mask = cv2.dilate(imagemask, kernel, iterations=1)  # 进行膨胀操作

    def get_four_corner_points(self):

        # mask Erosion
        self._mask_erosion()

        points = np.argwhere(self.mask == 255)
        points[:, [0, 1]] = points[:, [1, 0]]
        all_x_origin = points[:, 0]
        all_y_origin = points[:, 1]

        all_x = list(set(all_x_origin))
        all_y = list(set(all_y_origin))

        # 上下采样点 纵坐标最大值与最小值的差值的均值
        lst_ymax_ymin = []
        for i in range(1, len(all_x), self.step):
            single_x = all_x[i]
            index = np.argwhere(points[:, 0] == single_x)
            search_y = points[index][:, :, 1].squeeze()
            up_point = np.min(search_y)
            down_point = np.max(search_y)
            lst_ymax_ymin.append(down_point-up_point)
        max_y_number = sum(lst_ymax_ymin) / len(lst_ymax_ymin)

        # 以x为基础，搜索对应的y坐标，用于上下两条线的拟合
        up_point_y = []
        down_point_y = []
        x_points = []
        for i in range(1, len(all_x), self.step):
            single_x = all_x[i]
            index = np.argwhere(points[:, 0] == single_x)
            search_y = points[index][:, :, 1].squeeze()
            # x_search_y_array[i, :len(search_y)] = search_y
            up_point = np.min(search_y)
            down_point = np.max(search_y)
            if abs(down_point - up_point) > max_y_number * 0.8:  # 选点策略
                up_point_y.append(up_point)
                down_point_y.append(down_point)
                x_points.append(single_x)
        up_point_x, up_point_y = self.remove_out_group(x_points, up_point_y)
        down_point_x, down_point_y = self.remove_out_group(x_points, down_point_y)

        # too short line shouldnot use ransac, for up, short means x close
        up_sample_points, down_sample_points = [], []
        # up_points
        # print("up points number:", np.max(up_point_x) - np.min(up_point_x))
        if np.max(up_point_x) - np.min(up_point_x) > self.min_point:
            if len(set(list(up_point_y))) == 1:
                A_up = 0
                B_up = 1
                C_up = -1 * up_point_y[0]
            else:
                for i in range(len(up_point_x)):
                    up_sample_points.append([up_point_x[i], up_point_y[i]])
                up_sample_points = np.array(up_sample_points)
                if self.visual:
                    for point in up_sample_points:
                        cv2.circle(self.color_image, (int(point[0]), int(point[1])), 1, (255, 0, 0), -1)
                    cv2.imshow("color_image", self.color_image)
                    cv2.waitKey(self.wait_time)
                [A_up, B_up, C_up] = self.ransac_line(up_sample_points, 1000, 3, -1, 1)
        else:
            k_y, b_y = optimize.curve_fit(self.f_1, up_point_y, up_point_x)[0]
            A_up = 1
            B_up = -k_y
            C_up = - b_y

        # down_points
        # print("down points number:", np.max(down_point_x) - np.min(down_point_x))
        if np.max(down_point_x) - np.min(down_point_x) > self.min_point:
            if len(set(list(down_point_y))) == 1:
                A_down = 0
                B_down = 1
                C_down = -1 * down_point_y[0]
            else:
                for i in range(len(down_point_x)):
                    down_sample_points.append([down_point_x[i], down_point_y[i]])
                down_sample_points = np.array(down_sample_points)
                if self.visual:
                    for point in down_sample_points:
                        cv2.circle(self.color_image, (int(point[0]), int(point[1])), 1, (0, 255, 0), -1)
                    cv2.imshow("color_image", self.color_image)
                    cv2.waitKey(self.wait_time)
                [A_down, B_down, C_down] = self.ransac_line(down_sample_points, 1000, 3, -1, 1)
        else:
            k_y, b_y = optimize.curve_fit(self.f_1, down_point_y, down_point_x)[0]
            A_down = 1
            B_down = -k_y
            C_down = - b_y
        if self.visual:
            self.visual_line(A_up, B_up, C_up, self.color_image, color=(255, 0, 0))
            self.visual_line(A_down, B_down, C_down, self.color_image, color=(0, 255, 0))

        left_point_x, left_point_y = [], []
        right_point_x, right_point_y = [], []
        distance_thre = 10
        # 以y为基础，搜索对应的x坐标，用于左右两条线的拟合
        for i in range(1, len(all_y), self.step):
            single_y = all_y[i]
            index = np.argwhere(points[:, 1] == single_y)
            search_x = points[index][:, :, 0].squeeze()
            # x_search_y_array[i, :len(search_y)] = search_y
            left_point = np.min(search_x)
            right_point = np.max(search_x)

            d1_up = abs(A_up * left_point + B_up * single_y + C_up) / math.sqrt(A_up ** 2 + B_up ** 2)
            d1_down = abs(A_down * left_point + B_down * single_y + C_down) / math.sqrt(A_down ** 2 + B_down ** 2)
            d2_up = abs(A_up * right_point + B_up * single_y + C_up) / math.sqrt(A_up ** 2 + B_up ** 2)
            d2_down = abs(A_down * right_point + B_down * single_y + C_down) / math.sqrt(A_down ** 2 + B_down ** 2)

            # 通过距离筛选左右采样点
            if d1_up > distance_thre and d1_down > distance_thre:
                left_point_x.append(left_point)
                left_point_y.append(single_y)
                # cv2.circle(self.color_image, (int(left_point), int(single_y)), 3, (0, 255, 0), -1)
            if d2_up > distance_thre and d2_down > distance_thre:
                right_point_x.append(right_point)
                right_point_y.append(single_y)
                # cv2.circle(self.color_image, (int(right_point), int(single_y)), 3, (0, 255, 0), -1)

        left_point_x, left_point_y = self.remove_out_group(left_point_x, left_point_y)
        right_point_x, right_point_y = self.remove_out_group(right_point_x, right_point_y)
        # print("left points number:", np.max(left_point_y) - np.min(left_point_y))
        if np.max(left_point_y) - np.min(left_point_y) > self.min_point:
            if len(set(list(left_point_x))) == 1:
                A_left = 1
                B_left = 0
                C_left = -1 * left_point_x[0]

            else:
                left_sample_points = []
                # left_points
                for i in range(len(left_point_x)):
                    left_sample_points.append([left_point_x[i], left_point_y[i]])
                left_sample_points = np.array(left_sample_points)
                if self.visual:
                    for point in left_sample_points:
                        cv2.circle(self.color_image, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)
                    cv2.imshow("color_image", self.color_image)
                    cv2.waitKey(self.wait_time)
                # [A_left, B_left, C_left, k_left] = self.fitLineRansac(left_sample_points, 1000, 10, -4, 4, False)
                [A_left, B_left, C_left] = self.ransac_line(left_sample_points, 1000, 20, -1, 1, flag_horizontal=False)
        else:
            k_y, b_y = optimize.curve_fit(self.f_1, left_point_y, left_point_x)[0]
            A_left = 1
            B_left = -k_y
            C_left = - b_y

        # right_points
        # print("right points number:", np.max(right_point_y) - np.min(right_point_y))
        if np.max(right_point_y) - np.min(right_point_y) > self.min_point:
            if len(set(list(right_point_x))) == 1:
                A_right = 1
                B_right = 0
                C_right = -1 * right_point_x[0]
            else:
                right_sample_points = []
                for i in range(len(right_point_x)):
                    right_sample_points.append([right_point_x[i], right_point_y[i]])
                right_sample_points = np.array(right_sample_points)
                if self.visual:
                    for point in right_sample_points:
                        cv2.circle(self.color_image, (int(point[0]), int(point[1])), 1, (255, 0, 255), -1)
                    cv2.imshow("color_image", self.color_image)
                    cv2.waitKey(self.wait_time)
                [A_right, B_right, C_right] = self.ransac_line(right_sample_points, 1000, 20, -1, 1, flag_horizontal=False)
                # print(A_right, B_right, C_right)
                # print(right_sample_points)
        else:
            k_y, b_y = optimize.curve_fit(self.f_1, right_point_y, right_point_x)[0]
            A_right = 1
            B_right = -k_y
            C_right = - b_y

        if self.visual:
            self.visual_line(A_left, B_left, C_left, self.color_image, color=(0, 0, 255), flag_horizontal=False)
            self.visual_line(A_right, B_right, C_right, self.color_image, color=(255, 0, 255), flag_horizontal=False)
        up_left = self.line_intersection(A_up, B_up, C_up, A_left, B_left, C_left)
        up_right = self.line_intersection(A_up, B_up, C_up, A_right, B_right, C_right)
        down_right = self.line_intersection(A_down, B_down, C_down, A_right, B_right, C_right)
        down_left = self.line_intersection(A_down, B_down, C_down, A_left, B_left, C_left)

        if self.visual:
            cv2.circle(self.color_image, (int(up_left[0]), int(up_left[1])), 4, (0, 255, 255), -1)
            cv2.circle(self.color_image, (int(up_right[0]), int(up_right[1])), 4, (0, 255, 255), -1)
            cv2.circle(self.color_image, (int(down_left[0]), int(down_left[1])), 4, (0, 255, 255), -1)
            cv2.circle(self.color_image, (int(down_right[0]), int(down_right[1])), 4, (0, 255, 255), -1)
            cv2.imshow("color_image", self.color_image)
            cv2.waitKey(self.wait_time)
            cv2.destroyAllWindows()
        return [up_left, up_right, down_right, down_left]


    def run(self, mask, img, step=2, min_point=50, visual=False, wait_time=100):
        self.mask = mask.astype(np.uint8)
        self.color_image = img
        self.step = step
        self.min_point = min_point
        self.visual = visual
        self.wait_time = wait_time

        [up_left, up_right, down_right, down_left] = self.get_four_corner_points()
        flag = (np.array([up_left, up_right, down_right, down_left]) <= 0).any()   # 过滤边界
        if flag:
            # print([up_left, up_right, down_right, down_left])
            [up_left, up_right, down_right, down_left] = self.get_four_corner_points()

        if self.visual:
            for point in [up_left, up_right, down_right, down_left]:
                cv2.circle(self.color_image, (point[0], point[1]), 1, (255, 0, 0), -1)
                cv2.imshow("color_image", self.color_image)
                cv2.waitKey(self.wait_time)
        # print([up_left, up_right, down_right, down_left])
        return [up_left, up_right, down_right, down_left]


if __name__ == '__main__':

    # container
    # mask_path = "/home/westwell/Downloads/tmp/西井科技20230512-142710.jpg"
    # img_path = '/home/westwell/Downloads/tmp/西井科技20230512-142710.jpg'

    # img_path = "/cv/xyc/segment-anything/test_img/plate_0_plate_number_0_Left_8581.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/plate_0_plate_number_0_Left_8581/plate_0_plate_number_0_Left_8581_7_mask.jpg"

    # img_path = "/cv/debug.jpg"
    # mask_path = "/cv/debug.jpg"

    # img_path = "/cv/all_training_data/plate/cn/qqctu/dataset_6nc/images/test/2_62.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/2_62/2_62_6_mask.jpg"

    # img_path = "../notebooks/images/car.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/car/car_11_mask.jpg"

    # img_path = "/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img/15_54.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/15_54/15_54_35_mask.jpg"

    # img_path = "/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img/01_138.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/01_138/01_138_2_mask.jpg"

    # img_path = "/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img/03_96.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/03_96/03_96_17_mask.jpg"

    # img_path = "/cv/all_training_data/plate/cn/yolov7_plate/dataset/all/1649477086000.jpg"
    # mask_path = "/cv/xyc/segment-anything/output/1649477086000/1649477086000_24_mask.jpg"

    img_path = "/cv/all_training_data/plate/cn/yolov7_plate/dataset/all/plate_0_plate_number_0_Left_2681.jpg"
    mask_path = "/cv/xyc/segment-anything/output/plate_0_plate_number_0_Left_2681/plate_0_plate_number_0_Left_2681_16_mask.jpg"


    mask = cv2.imread(mask_path, 0)
    img = cv2.imread(img_path, 1)
    get_corner_cunction = get_corner()
    [left_up, left_right, down_right, down_left] = get_corner_cunction.run(
                                                   mask, img, step=1, min_point=100, visual=True, wait_time=1000)

    # for test
    # for i in range(5):
    #     print("+++++++++++++++++++++", i)
    #     img = cv2.imread(img_path, 1)
    #     [left_up, left_right, down_right, down_left] = get_corner_cunction.run(mask, img, step=1, min_point=100, visual=True)