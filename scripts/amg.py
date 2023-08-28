# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import random
import sys
import numpy as np
sys.path.append("..")

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    # default="/cv/all_training_data/plate/cn/qqctu/dataset_6nc/images/test/2_62.jpg",
    # default="/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img/02_32.jpg",
    # default="../notebooks/images/car.jpg",
    # default="/cv/xyc/segment-anything/test_img/plate_0_plate_number_0_Left_8581.jpg",
    # default="../test_img",
    # default="/cv/all_training_data/psa/datasets/img/vlcsnap-2023-04-28-09h08m20s131.png",
    # default="/cv/xyc/segment-anything/test_img/8_330.jpg",
    # default="/cv/xyc/segment-anything/test_img/plate_0_plate_number_0_Left_8594.jpg",
    # default="/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img/15_54.jpg",
    # default="/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img/01_138.jpg",
    # default="/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/dataset/all/1649477086000.jpg",
    # default="/cv/all_training_data/plate/cn/frontback/dataset_5nc_model1st/img/03_96.jpg",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/dataset/all",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/dataset/all/02_276.jpg",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/dataset/all/1649338126000.jpg",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/detect_plate_datasets/train_data/CRPD_TRAIN/",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/dataset_yolo/error/00700191570881-90_90-258&551_421&607-431&611_245&615_245&550_431&546-0_0_7_3_32_29_29-195-14.jpg",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/detect_plate_datasets/train_data/CCPD/0_(53).jpg",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/dataset/error/imperfection/1649459708000.jpg",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/detect_plate_datasets/val_detect/single_yellow_val",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/detect_plate_datasets/val_detect/blue",
    # default="/cv/all_training_data/plate/cn/yolov7_plate/detect_plate_datasets/val_detect/double_val",
    # default="/cv/all_training_data/yancan/nansha/seg/frontback/dataset/rgb/1655277000.3_origin.png",
    default="/cv/all_training_data/plate/psa/dataset/06-08PL3_right_rearplate_video/img/vlcsnap-2023-06-15-15h07m21s539.png",
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


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))



def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image0 = cv2.imread(t)
        if image0 is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)

        if args.visual_flag:
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(masks))]
            for i in range(len(masks)):
                bbox = masks[i]["bbox"]
                cv2.rectangle(image0, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[i], 2)
                cv2.imshow("img", image0)
                cv2.waitKey()

        if args.save_mask:
            mask_img = np.zeros_like(image)
            i = 0
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(os.listdir(save_base)))]
            for label in os.listdir(save_base):
                if os.path.splitext(label)[-1] == ".png":
                    single_mask_img = cv2.imread(os.path.join(save_base, label), -1)
                    index = np.argwhere(single_mask_img == 255)
                    # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    for single_index in index:
                        x, y = single_index[0], single_index[1]
                        mask_img[x, y, :] = colors[i]
                    i += 1

            cv2.imwrite(os.path.join(save_base, base + "_pred.jpg"), mask_img)

    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
