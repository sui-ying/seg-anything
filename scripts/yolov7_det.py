import argparse
import time
import os
import copy
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

import sys
sys.path.append("/cv/xyc/yolov7_plate")
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from plate_recognition.plate_rec import get_plate_result, allFilePath, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from utils.datasets import letterbox
from utils.cv_puttext import cv2ImgAddText


class Infer_img_by_yolov7:
    def __init__(self, weight):
        parser = argparse.ArgumentParser()
        parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold, 0.3')
        parser.add_argument('--iou_thres', type=float, default=0.4, help='IOU threshold for NMS, 0.5')
        parser.add_argument('--kpt-label', type=int, default=4, help='number of keypoints')
        parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--output', type=str, default='result', help='source')
        parser.add_argument('--visual_img', type=str, default=False, help='display img')
        parser.add_argument('--expand_roi', type=str, default=False, help='expand roi')

        opt = parser.parse_args()

        self.opt = opt
        self.img_size = self.opt.img_size
        self.conf_thres = self.opt.conf_thres
        self.iou_thres = self.opt.iou_thres
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual = self.opt.visual_img
        self.ROI_model = attempt_load(weight, map_location=self.device)

    def _visual_results(self, result_list):
        for result in result_list:
            x1, y1, x2, y2 = [int(round(x)) if i < 4 else x for i, x in enumerate(result[:4])]  # result[:4]
            score = "%.3f" % result[4]
            cls = int(result[5])   # 车牌的的类型0代表单牌，1代表双层车牌

            four_points = np.zeros((4, 2))
            for i in range(4):
                point_x, point_y = int(result[6 + 3 * i]), int(result[7 + 3 * i])
                four_points[i] = np.array([point_x, point_y])
            four_points_clockwise = self.order_points(four_points)
            # roi_img = self.four_point_transform(self.img, four_points_clockwise)  # 透视变换得到车牌小图
            # cv2.imshow("roi_img", roi_img)
            # cv2.waitKey()

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 画框
            cv2.putText(img, str(cls) + " " + str(score), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for point in four_points_clockwise:
                cv2.circle(self.img, (int(point[0]), int(point[1])), 5, (155, 155, 155), -1)

        cv2.imshow("img", self.img)
        cv2.waitKey()

    def order_points(self, four_points):  # 关键点按照（左上，右上，右下，左下）排列
        rect = np.zeros((4, 2), dtype="float32")
        s = four_points.sum(axis=1)
        rect[0] = four_points[np.argmin(s)]
        rect[2] = four_points[np.argmax(s)]
        diff = np.diff(four_points, axis=1)
        rect[1] = four_points[np.argmin(diff)]
        rect[3] = four_points[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):  # 透视变换
        rect = pts.astype("float32")
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _detect_plate(self):
        im0 = copy.deepcopy(self.img)
        imgsz = (self.img_size, self.img_size)
        img = letterbox(im0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x640X640
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.ROI_model(img)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, kpt_label=4, agnostic=True)

        results = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=4, step=3)
                for j in range(det.size()[0]):
                    if det[j, 4].cpu().numpy() > 0.75:  # conf > 0.6
                        # x1, y1, x2, y2, conf, cls, pt1_x, pt1_y, pt1_conf,
                        # pt2_x, pt2_y, pt2_conf, pt3_x, pt3_y, pt3_conf, pt4_x, pt4_y, pt4_conf
                        results.append(det[j, :].view(-1).tolist())
        return results

    def run(self, img):
        self.img = img
        result_list = self._detect_plate()
        if self.visual:
            self._visual_results(result_list)
        return result_list


if __name__ == '__main__':
    # input
    input_path = "../test_img"
    if not os.path.isdir(input_path):
        targets = [input_path]
    else:
        targets = [f for f in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, f))]
        targets = [os.path.join(input_path, f) for f in targets]
    # init class
    infer_img_by_yolov7 = Infer_img_by_yolov7('/cv/xyc/yolov7_plate/runs/train/exp/weights/best.pt')

    for img_path in targets:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        result_list = infer_img_by_yolov7.run(img)
        print(result_list)
