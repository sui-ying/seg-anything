"""det image with 1 models"""
import os
import shutil
import time

import cv2
import torch
import argparse
import random
import collections

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
from tqdm import tqdm

import numpy as np
import sys
sys.path.append("../../myscripts/")
from tools import not_exists_path_make_dir, not_exists_path_make_dirs

sys.path.append("../../yolov5")
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
     xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Infer_video_by_yolov5:
    def __init__(self, weight):
        parser = argparse.ArgumentParser()
        parser.add_argument('--conf_thres', type=float, default=0.40, help='object confidence threshold, 0.25')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS, 0.45')
        parser.add_argument('--view_img', type=str, default=False, help='display results')
        parser.add_argument('--debug_img', type=str, default=True, help='display img')
        parser.add_argument('--save_text', type=str, default=False, help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--expand_roi', type=str, default=False, help='expand roi')
        opt = parser.parse_args()
        self.opt = opt

        self.img_size = 640
        self.weight = weight
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'
        self.ROI_model = attempt_load(self.weight, map_location=self.device)
        if self.half: self.ROI_model.half()
        print("load ROI detect model successfully...")
        self.stride = int(self.ROI_model.stride.max())
        self.labelmap = self.ROI_model.module.names if hasattr(self.ROI_model, 'module') else self.ROI_model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.labelmap]

    def _box_iou(self, box1, box2):
        """
        计算两个矩形框的交并比(IOU)
        :param box1: 第一个矩形框，格式为[x1, y1, x2, y2, score, cls]
        :param box2: 第二个矩形框，格式为[x1, y1, x2, y2, score, cls]
        :return: 交并比(IOU)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        if x1 < x2 and y1 < y2:
            intersection = (x2 - x1) * (y2 - y1)
            union = area1 + area2 - intersection
            return intersection / union, area1, area2
        else:
            return 0, area1, area2

    def _merge_boxes(self, box1, box2):
        """
        合并两个框为一个大框
        :param box1: 第一个框，格式为[x1, y1, x2, y2, score1, cls1]
        :param box2: 第二个框，格式为[x1, y1, x2, y2, score2, cls2]
        :return: 合并后的大框，格式为[x1, y1, x2, y2, score]
        """
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[2], box2[2])
        y2 = max(box1[3], box2[3])
        return [x1, y1, x2, y2, max(box1[-2], box2[-2])]

    def _filter_boxes(self, boxes, threshold1=0.75, threshold2=0.2):
        """
        过滤重叠的框
        :param boxes: 框列表，[N * [x1, y1, x2, y2, score, cls]]
        :param threshold1: 重叠比例阈值，默认为0.8
        :param threshold1: 接近包含的两个框的阈值，默认为0.5
        :return: 去重后的框列表
        """

        # 框数为0或1，无需过滤
        if len(boxes) < 2:
            return boxes
        new_boxes = []
        keep = [True] * len(boxes)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou, area_i, area_j = self._box_iou(boxes[i], boxes[j])
                # print(i, j, iou)
                if iou >= threshold1:
                    if boxes[i][-2] > boxes[j][-2]:
                        keep[j] = False
                    else:
                        keep[i] = False
                elif iou > threshold2 and iou < threshold1:
                    if area_i > area_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                elif iou <= threshold2 and iou > 0:
                    if self.opt.project_type == "container_recog" and int(boxes[i][-1]) == 0 and int(boxes[j][-1]) == 0:
                        keep[i] = False
                        keep[j] = False
                        _box = self._merge_boxes(boxes[i], boxes[j])
                        _box.append(8.0)
                        new_boxes.append(_box)

        for i in range(len(boxes)):
            if keep[i]:
                new_boxes.append(boxes[i])

        return new_boxes

    def _cv2ImgAddText(self, img, label, xy, size):
        # 导入中文字体
        font_path = '/home/westwell/Downloads/install/font/SimHei.ttf'
        font = ImageFont.truetype(font_path, size=size)

        img_PIL = Image.fromarray(img)
        draw = ImageDraw.Draw(img_PIL)
        draw.text((xy[0], xy[1]), label, font=font, fill=(255, 255, 255), align="left")
        img = np.array(img_PIL)  # PIL to ndarray

        return img

    def _scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """将预测的坐标信息转换回原图尺度"""
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self._clip_coords(coords, img0_shape)
        return coords

    def _clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        # 根据图像形状（高度、宽度）限定xyxy边界框的坐标值, 防止越界
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    def _read_xml_to_lst(self, image_id):
        """
        :param xmlpath: xml path
        :param image_id:
        :return: a 2d lst [[cls, xmin, ymin, xmax, ymax], [...]]
        """

        bbox_xyxy = []  # xyxy
        in_file = open(os.path.join(self.xml_path, '%s.xml' % (image_id)))
        tree = ET.parse(in_file)
        root = tree.getroot()
        # size = root.find('size')
        # w = int(size.find('width').text)
        # h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text
            xmlbox = obj.find('bndbox')
            xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
            ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
            bbox_xyxy.append([cls, xmin, ymin, xmax, ymax])
        return bbox_xyxy

    def _whether_lst2_include_lst1(self, bbox_xyxy, roi_infos):
        """
        To determine whether lst2 include lst1, lst2 ≥ lst
        :param lst1: 1d Child list,
        :param lst2: 1d Parent list
        :return: True or False
        """

        bbox_xyxy_cls = list(map(lambda x: x[0], [item for item in bbox_xyxy]))
        roi_info_cls = list(map(lambda x: self.roi_map[int(x[-1])], [item for item in roi_infos]))

        if self.opt.project_type == "danger_recog":
            for i in range(len(roi_info_cls)):
                # pred_cls = self.roi_map[int(pred_cls)]
                pred_cls = roi_info_cls[i]
                # for danger
                if pred_cls == 'sign_danger_qt5' or pred_cls == 'sign_danger_gr' or pred_cls == 'sign_danger_yd' or pred_cls == 'sign_danger_26':
                    roi_info_cls[i] = 'sign_danger_yd'
                elif pred_cls == 'sign_danger_qt1' or pred_cls == 'sign_danger_yt1':
                    roi_info_cls[i] = 'sign_danger_yt1'
                elif pred_cls == 'sign_danger_qt2' or pred_cls == 'sign_danger_23':
                    roi_info_cls[i] = 'sign_danger_yt2'

        dic1 = {}
        dic2 = {}
        t = 0
        d1 = collections.Counter(roi_info_cls)
        d2 = collections.Counter(bbox_xyxy_cls)
        for k in d1:
            dic1[k] = d1[k]
        for k in d2:
            dic2[k] = d2[k]
        try:
            for i in dic1.keys():
                if dic2[i] >= dic1[i]:
                    t += 1
            if t == len(dic1):
                return True
            else:
                return False
        except KeyError:
            return False

    def _lst1_equal_lst2(self, bbox_xyxy, roi_info):
        """
        :param bbox_xyxy: [[cls, xmin, ymin, xmax, ymax], [...]]
        :param roi_info: [[xmin, ymin, xmax, ymax, conf, cls], [...]]
        :return:
        """
        bbox_xyxy_cls = list(map(lambda x: x[0], [item for item in bbox_xyxy]))
        roi_info_cls = list(map(lambda x: self.roi_map[int(x[-1])], [item for item in roi_info]))

        if self.opt.project_type == "danger_recog":
            for i in range(len(roi_info_cls)):
                # pred_cls = self.roi_map[int(pred_cls)]
                pred_cls = roi_info_cls[i]
                # for danger
                if pred_cls == 'sign_danger_qt5' or pred_cls == 'sign_danger_gr' or pred_cls == 'sign_danger_yd' or pred_cls == 'sign_danger_26':
                    roi_info_cls[i] = 'sign_danger_yd'
                elif pred_cls == 'sign_danger_qt1' or pred_cls == 'sign_danger_yt1':
                    roi_info_cls[i] = 'sign_danger_yt1'
                elif pred_cls == 'sign_danger_qt2' or pred_cls == 'sign_danger_23':
                    roi_info_cls[i] = 'sign_danger_yt2'
        a = Counter(bbox_xyxy_cls)
        b = Counter(roi_info_cls)
        if dict(a) == dict(b):
            return True     # bbox_xyxy_cls == roi_info_cls
        else:
            return False

    def _lable_diff_pred(self, img, img_file, bbox_xyxy, roi_infos):
        """
        :param img: original img
        :param bbox_xyxy: label bbox info
        :param roi_infos: pred bbox info
        :return: True or False
        """

        filename, suffix = os.path.splitext(img_file)

        # show label
        labelled = np.copy(img)
        for i in range(len(bbox_xyxy)):
            cls, xmin, ymin, xmax, ymax = bbox_xyxy[i]
            labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                     self.colors[int(self.roi_map.index(cls))], 2)
            labelled = cv2.putText(labelled, cls.split("sign_danger_")[-1], (int(xmin), int(ymin) - 2),
                                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            # labelled = cv2.putText(labelled, cls.split("sign_danger_")[-1], (int(xmin), int(ymin) - 2), cv2.FONT_HERSHEY_PLAIN,
            #                        2, self.colors[int(self.roi_map.index(cls))], 2)  # font scale, thickness
        # cv2.imshow("labelled", labelled)
        # cv2.waitKey(0)

        # show pred
        pred_img = np.copy(img)
        for i in range(len(roi_infos)):
            xmin, ymin, xmax, ymax, score, pred_cls = roi_infos[i]
            pred_img = cv2.rectangle(pred_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                     self.colors[int(pred_cls)], 2)
            pred_cls = self.roi_map[int(pred_cls)]
            if self.opt.project_type == "danger_recog":
                # for danger
                if pred_cls == 'sign_danger_qt5' or pred_cls == 'sign_danger_gr' or pred_cls == 'sign_danger_yd' or pred_cls == 'sign_danger_26':
                    pred_cls = 'sign_danger_yd'     # 有毒
                elif pred_cls == 'sign_danger_qt1' or pred_cls == 'sign_danger_yt1':
                    pred_cls = 'sign_danger_yt1'    # 白色易燃气体
                elif pred_cls == 'sign_danger_qt2' or pred_cls == 'sign_danger_23':
                    pred_cls = 'sign_danger_yt2'    # 黑色易燃气体
            pred_img = cv2.putText(pred_img, pred_cls.split("sign_danger_")[-1] + " " + str('%.2f' % score),
                                   (int(xmin), int(ymin) - 2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            # pred_img = cv2.putText(pred_img, self.roi_map[int(pred_cls)].split("sign_danger_")[-1] + " " + str('%.2f' % score), (int(xmin), int(ymin) - 2),
            #                        cv2.FONT_HERSHEY_PLAIN, 2, self.colors[int(pred_cls)], 2)  # font scale, thickness
        # cv2.imshow("pred_img", pred_img)
        # cv2.waitKey(0)

        # show concat and save
        # cv2.namedWindow("concat", 0)
        # cv2.resizeWindow("concat", 1920, 1080)
        # cv2.imshow("concat", np.hstack((labelled, pred_img)))
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(self.save_visual, filename + ".jpg"), np.hstack((labelled, pred_img)),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        # copy img and xml
        shutil.copy(os.path.join(self.img_path, img_file), self.save_img)
        shutil.copy(os.path.join(self.xml_path, filename + ".xml"), self.save_xml)

    def _det_roi(self, img0, filter_overlap=False):
        # 原图等比例放缩到640*640, 长边为640
        img = letterbox(img0, self.img_size, stride=self.stride)[0]  # new_H, new_W, c
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # infer
        pred1_ori = self.ROI_model(img, augment=self.opt.augment)[0]
        # Apply NMS
        pred1_nms = non_max_suppression(pred1_ori, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                    agnostic=self.opt.agnostic_nms)
        if filter_overlap:
            _pred1 = torch.tensor(self._filter_boxes(pred1_nms[0].cpu().numpy().tolist()))
            pred1_nms = [_pred1]

        roi_info = pred1_nms[0]  # used only when batch_size == 1
        # Rescale boxes from img_size to im0 size
        if roi_info.numel() != 0:
            roi_info[:, :4] = self._scale_coords(img.shape[2:], roi_info[:, :4], img0.shape).round()

        # for i, roi_info in enumerate(pred1_nms[0]):
        #     # Rescale boxes from img_size to im0 size
        #     roi_info[:, :4] = self._scale_coords(img.shape[2:], roi_info[:, :4], img0.shape).round()

        return roi_info.cpu().numpy().tolist()

    def run(self, img):
        roi_infos = self._det_roi(img, filter_overlap=True)
        return roi_infos


if __name__ == '__main__':
    infer_video_by_yolov5 = Infer_video_by_yolov5("/cv/all_training_data/plate/cn/qqctu/models/qqctu_roi_det_20230331.pt")
    img = cv2.imread("/cv/all_training_data/plate/cn/qqctu/dataset_6nc/images/test/2_62.jpg")
    infer_video_by_yolov5.run(img)
