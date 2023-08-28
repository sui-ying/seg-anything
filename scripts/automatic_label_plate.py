"""
using sam and yolov7 model for labeling:
    plate, psa
Requirements: GPU mem > 8G
"""
import shutil
import time
import cv2  # type: ignore
import random
import sys
import numpy as np
from tqdm import tqdm

import glob
import argparse
import json
import os
from typing import Any, Dict, List

sys.path.append("/cv/xyc/myscripts")
from tools import not_exists_path_make_dir, not_exists_path_make_dirs

sys.path.append("..")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from amg import parser, amg_settings
from amg import write_masks_to_folder, get_amg_kwargs

sys.path.append("../../yolov7_plate")
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from plate_recognition.plate_rec import get_plate_result, allFilePath, init_model, cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from utils.datasets import letterbox
# from utils.cv_puttext import cv2ImgAddText
from yolov7_det import Infer_img_by_yolov7

# # import yolov5
# sys.path.append("../../yolov5")
# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages, letterbox
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#      xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized
# from yolov5_det import Infer_video_by_yolov5


from find_four_points import find_four_sample_points
from find_four_corners import get_corner


class Labeling_plate():

    def __init__(self, args):
        self.opt = args
        print(self.opt)
        self.show_mask = False
        self.save_mask = True
        self.visual = True
        self.save_visual = True
        self.wait_time = 1000
        self.save_points = False

        self.save_path = "/cv/all_training_data/plate/cn/yolov7_plate/dataset"
        self.save_all = os.path.join(self.save_path, "all")
        self.save_visual_path = os.path.join(self.save_path, "visual")
        self.error_path = os.path.join(self.save_path, "error")
        self.save_yolov7_no_results = os.path.join(self.error_path, "yolov7_no_results")
        self.save_sam_no_better_mask = os.path.join(self.error_path, "sam_no_better_mask")
        not_exists_path_make_dirs([self.save_all, self.save_visual_path, self.save_yolov7_no_results])

        # self.yolov7_model_path = '/cv/xyc/yolov7_plate/weights/yolov7-lite-s.pt'
        self.yolov7_model_path = '/cv/xyc/yolov7_plate/runs/train/exp9/weights/best.pt'

    def _box_iou(self, box1, box2):
        """
        计算两个矩形框的交并比(IOU)
        :param box1: 第一个矩形框，格式为[x1, y1, x2, y2, score, cls]
        :param box2: 第二个矩形框，格式为[x1, y1, x2, y2, score, cls]
        :return: 交并比(IOU)
        """
        # if (len(box1) != 4 or len(box1) != 6) or (len(box2) != 4 or len(box2) != 6):
        #     return -1, -1, -1
        if box1[0] > box2[2]:
            return -1, -1, -1
        if box1[1] > box2[3]:
            return -1, -1, -1
        if box1[2] < box2[0]:
            return -1, -1, -1
        if box1[3] < box2[1]:
            return -1, -1, -1

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

    def _xyxy_convert_xywh(self, size, box, four_points_clockwise):
        """
        :param size: (width, height)
        :param box:  (xmin, ymin, xmax, ymax)
        :param four_points_clockwise: [up_left, up_right, down_right, down_left]; up_left = [x, y]
        :return:
        """
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]

        # center x, y;  w, h
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh

        # up_left, up_right, down_right, down_left
        pt0_x = four_points_clockwise[0][0] * dw
        pt0_y = four_points_clockwise[0][1] * dh
        pt1_x = four_points_clockwise[1][0] * dw
        pt1_y = four_points_clockwise[1][1] * dh
        pt2_x = four_points_clockwise[2][0] * dw
        pt2_y = four_points_clockwise[2][1] * dh
        pt3_x = four_points_clockwise[3][0] * dw
        pt3_y = four_points_clockwise[3][1] * dh
        return x, y, w, h, pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y

    def run(self):

        # input
        if not os.path.isdir(args.input):
            targets = [args.input]
        else:
            img_ext = [".jpg", ".jpeg", ".png", ".bmp"]
            targets = []
            # 遍历指定目录及其子文件夹下的所有图片文件
            for root, dirs, files in os.walk(args.input):
                for file in files:
                    if file.lower().endswith(tuple(img_ext)):
                        targets.append(os.path.join(root, file))
        os.makedirs(args.output, exist_ok=True)

        # load model
        print("Loading yolov7_plate model ...")
        infer_img_by_yolov7 = Infer_img_by_yolov7(self.yolov7_model_path)

        print("Loading sam model...")
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        _ = sam.to(device=args.device)
        output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
        amg_kwargs = get_amg_kwargs(args)
        generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

        # import ge_corner
        get_corner_function = get_corner()

        with tqdm(total=len(targets)) as p_bar:
            for t in targets:
                try:
                    base = os.path.basename(t)
                    base = os.path.splitext(base)[0]
                    save_base = os.path.join(args.output, base)
                    # if os.path.exists(os.path.join(self.save_all, base + '.txt')):
                    #     p_bar.update(1)
                    #     continue
                    image0 = cv2.imread(t)  # for visual

                    cv2.imwrite(os.path.join(self.save_all, base + ".jpg"), image0)
                    height, width, channel = image0.shape
                    if image0 is None:
                        print(f"Could not load '{t}' as an image, skipping...")
                        continue
                    viusal_img = np.copy(image0)
                    image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)     # for generate mask

                    # 1. yolov7 det plate: single or double row plate
                    yolov7_results = infer_img_by_yolov7.run(image0)
                    if len(yolov7_results) < 1:
                        print("yolov7 not detect plate roi")
                        if not os.path.exists(os.path.join(self.save_yolov7_no_results, base + ".jpg")):
                            shutil.copy(t, self.save_yolov7_no_results)
                        continue

                    # 2. sam infer masks
                    masks = generator.generate(image)
                    # 按照面积排序
                    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

                    # show sam results
                    if self.save_mask:
                        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(masks)+1)]
                        # show min rectangle
                        if self.show_mask:
                            for i in range(len(masks)):
                                bbox = masks[i]["bbox"]
                                cv2.rectangle(image0, (int(bbox[0]), int(bbox[1])),
                                              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), colors[i], 2)
                                cv2.imshow("img", image0)
                                cv2.waitKey(self.wait_time)

                        os.makedirs(save_base, exist_ok=True)
                        write_masks_to_folder(masks, save_base)

                        # draw masks on a figure
                        mask_img = np.zeros_like(image)
                        i = 0
                        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(os.listdir(save_base)))]
                        for label in os.listdir(save_base):
                            if os.path.splitext(label)[-1] == ".png":
                                single_mask_img = cv2.imread(os.path.join(save_base, label), -1)
                                index = np.argwhere(single_mask_img == 255)
                                for single_index in index:
                                    x, y = single_index[0], single_index[1]
                                    mask_img[x, y, :] = colors[i]
                                i += 1

                        cv2.imwrite(os.path.join(save_base, base + "_pred.jpg"), mask_img)

                    out_file = open(os.path.join(self.save_all, base + '.txt'), 'w')
                    for ii in range(len(yolov7_results)):
                        single_plate_xyxy = yolov7_results[ii][:4]
                        cls = int(yolov7_results[ii][5])
                        best_mask_index = 0
                        max_iou = 0

                        # 3 select mask of (single) plate: max iou between roi and mask
                        for i in range(len(masks)):
                            bbox = masks[i]["bbox"]
                            xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                            iou, area1, area2 = self._box_iou(single_plate_xyxy, xyxy)
                            if iou > max_iou:
                                best_mask_index = i
                                max_iou = iou
                                if self.save_visual:
                                    # max iou mask rectangle
                                    cv2.rectangle(image0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                                  (0, 255, 0), 1)
                                    if self.visual:
                                        cv2.imshow("img", image0)
                                        cv2.waitKey(self.wait_time)
                        if max_iou < 0.2:
                            shutil.move(t, self.save_sam_no_better_mask)
                            print("max_iou: {} sam does not generate better plate mask".format(str(max_iou)))
                            continue

                        bbox = masks[best_mask_index]["bbox"]
                        xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                        # save mask
                        cv2.imwrite(os.path.join(save_base, base + "_" + str(best_mask_index) + "_mask.jpg"),
                                    masks[best_mask_index]["segmentation"] * 255)

                        if self.save_points:
                            # save xyxy_four_points as npy
                            xyxy_four_points = [[bbox[0], bbox[1]],
                                                [bbox[0] + bbox[2],  bbox[1]],
                                                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                                                [bbox[0], bbox[1] + bbox[3]]]
                            np.save(os.path.join(save_base, base + "_xyxy_four_points.npy"), xyxy_four_points)

                            # save "all points of plate mask" as npy
                            nonzero_points = np.transpose(np.nonzero((masks[best_mask_index]["segmentation"] * 255)))
                            # 交换其第一列和第二列
                            nonzero_points[:, [0, 1]] = nonzero_points[:, [1, 0]]
                            np.save(os.path.join(save_base, base + ".npy"), nonzero_points)

                        # 4. find four points
                        four_points_clockwise = get_corner_function.run(
                            masks[best_mask_index]["segmentation"] * 255,
                            viusal_img, step=1, min_point=100, visual=False)
                        # print("four_points_clockwise: ", four_points_clockwise)

                        x, y, w, h, pt0_x, pt0_y, pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y = \
                            self._xyxy_convert_xywh((width, height), xyxy, four_points_clockwise)

                        line = str(cls) + " " \
                               + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " \
                               + str(pt0_x) + " " + str(pt0_y) + " " + str(pt1_x) + " " + str(pt1_y) + " " \
                               + str(pt2_x) + " " + str(pt2_y) + " " + str(pt3_x) + " " + str(pt3_y) + "\n"                                                                       "\n "
                        out_file.write(line)

                        if self.save_visual:
                            # yolov7 rectangle xyxy
                            cv2.rectangle(image0, (int(single_plate_xyxy[0]), int(single_plate_xyxy[1])),
                                          (int(single_plate_xyxy[2]), int(single_plate_xyxy[3])), (255, 0, 0), 2)

                            # four_points_clockwise
                            for point in four_points_clockwise:
                                cv2.circle(image0, (int(point[0]), int(point[1])), 6, (0, 255, 255), -1)
                            cv2.imwrite(os.path.join(self.save_visual_path, base + ".jpg"), image0,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                            cv2.imwrite(os.path.join(save_base, base + ".jpg"), image0,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                            if self.visual:
                                cv2.imshow("img", image0)
                                cv2.waitKey(self.wait_time)

                    out_file.close()
                    p_bar.update(1)
                except Exception as e:
                    if not os.path.exists(os.path.join(self.error_path, base + ".jpg")):
                        shutil.move(t, self.error_path)
                    if os.path.exists(os.path.join(self.save_all, base + '.txt')):
                        os.remove(os.path.join(self.save_all, base + '.txt'))
                    print(base, e)

            print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic mask generation on an input image or directory of images, "
            "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
            "as well as pycocotools if saving in RLE format."
        )
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../weight/sam_vit_b_01ec64.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/cv/all_training_data/plate/cn/yolov7_plate/dataset/error/1649606115000.jpg",
        # default="/cv/all_training_data/plate/cn/yolov7_plate/dataset/error",
        help="Path to either a single input image or folder of images."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../output",
        help=(
            "Path to the directory where masks will be output. Output will be either a folder "
            "of PNGs per image or a single json with COCO-style masks."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    parser.add_argument("--visual_flag", type=bool, default=False, help="visual for mask")

    parser.add_argument("--visual", type=bool, default=False, help="visual for four points")

    parser.add_argument("--save_mask", type=bool, default=True, help="save mask")

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )

    args = parser.parse_args()
    labeling_plate = Labeling_plate(args)
    labeling_plate.run()


