"""
This code aims to check dataset after split.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./')
import os
import numpy as np
import cv2
import argparse
from dataloader.prophesee.src.visualize import vis_utils as vis


def play_files_parallel(root_dir, npy_file, height, width):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    event_dir = os.path.join(root_dir, "events")
    npy_data = np.load(npy_file, allow_pickle=True)

    event_files = sorted(os.listdir(event_dir))

    event_list = []
    for ev in event_files:
        event_path = os.path.join(event_dir, ev)
        event_list.append(event_path)

    for idx in range(len(event_list)):
        # use the naming pattern to find the corresponding box file
        labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.namedWindow('gt', cv2.WINDOW_NORMAL)

        # load events and boxes from all files
        events = np.load(event_list[idx])
        boxes = npy_data[idx]

        im = frame[0:height, 0:width]
        # call the visualization functions
        im = vis.make_binary_histo(events, img=im, width=width, height=height)
        vis.drawing_bboxes(im, boxes, labelmap=labelmap)

        # display the result
        cv2.imshow('gt', frame)
        cv2.waitKey(10)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument(
        '-r', '--records', type=str,
        default="/home/wds/Desktop/split_train14/train",
        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument(
        '-n', "--npy_file", default="/home/wds/Desktop/red_split_gen4_onehead/det_result/prediction.npy", type=str,
        help="model predictions(bounding boxes), type [x_min, y_min, x_max, y_max, confidence, cls_id]")
    parser.add_argument('--height', default=720, type=int, help="image height")
    parser.add_argument('--width', default=1280, type=int, help="image width")

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    play_files_parallel(ARGS.records, ARGS.npy_file, ARGS.height, ARGS.width)
