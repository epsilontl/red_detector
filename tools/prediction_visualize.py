"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import math
import os
import numpy as np
import cv2
import argparse
from dataloader.prophesee.src.visualize import vis_utils as vis
from dataloader.prophesee.src.io.psee_loader import PSEELoader


def play_files_parallel(td_dir, npy_files, delta_t=50000, skip=0):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    td_dirs = os.listdir(td_dir)
    td_files = list()  # .dat files
    for td in td_dirs:
        if td[-3:] == "dat":
            abs_td_files = os.path.join(td_dir, td)
            td_files.append(abs_td_files)

    vis_idx = 0
    bounding_boxes = np.load(npy_files, allow_pickle=True)  # load detection results
    # open the video object for the input files
    for td_file in td_files:
        videos = [PSEELoader(td_file)]

        height, width = videos[0].get_size()
        labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

        # optionally skip n minutes in all videos
        for v in videos:
            v.seek_time(skip)

        # preallocate a grid to display the images
        size_x = int(math.ceil(math.sqrt(len(videos))))
        size_y = int(math.ceil(len(videos) / size_x))
        frame = np.zeros((size_y * height, width * size_x, 3), dtype=np.uint8)
        cv2.namedWindow("prediction_visualize", cv2.WINDOW_NORMAL)

        # while all videos have something to read
        while not sum([video.done for video in videos]):

            # load events and boxes from all files
            events = [video.load_delta_t(delta_t) for video in videos]
            if len(events[0]):
                boxes = [bounding_boxes[vis_idx]]
                vis_idx += 1
            else:  # no event streams
                boxes = list()

            for index, (evs, box) in enumerate(zip(events, list(boxes))):
                y, x = divmod(index, size_x)  # index/size_x and index%size_x
                # put the visualization at the right spot in the grid
                im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
                # call the visualization functions
                im = vis.make_binary_histo(evs, img=im, width=width, height=height)

                vis.drawing_bboxes(im, box, labelmap=labelmap)

            # display the result
            cv2.imshow("prediction_visualize", frame)
            cv2.waitKey(10)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="visualize one or several event files along with their boxes")
    parser.add_argument(
        '-t', "--td_dir",
        default="/home/wds/Desktop/trainfilelist14/train",
        type=str, help="input event files, annotation files are expected to be in the same folder")
    parser.add_argument(
        '-n', "--npy_file", default="/home/wds/Desktop/red_split_gen4/det_result/prediction.npy", type=str,
        help="model predictions(bounding boxes), type [x_min, y_min, x_max, y_max, confidence, cls_id]")
    parser.add_argument('-s', "--skip", default=500000, type=int, help="skip the first n microseconds")
    parser.add_argument('-d', "--delta_t", default=50000, type=int, help="load files by delta_t in microseconds")

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    play_files_parallel(td_dir=ARGS.td_dir, npy_files=ARGS.npy_file, skip=ARGS.skip, delta_t=ARGS.delta_t)
