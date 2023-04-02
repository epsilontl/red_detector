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
from numpy.lib import recfunctions as rfn


def cropToFrame(np_bbox, height, width):
    """Checks if bounding boxes are inside frame. If not crop to border"""
    boxes = []
    for box in np_bbox:
        if box[2] > 1280:  # filter error label
            continue

        if box[0] < 0:  # x < 0 & w > 0
            box[2] += box[0]
            box[0] = 0
        if box[1] < 0:  # y < 0 & h > 0
            box[3] += box[1]
            box[1] = 0
        if box[0] + box[2] > width:  # x+w>1280
            box[2] = width - box[0]
        if box[1] + box[3] > height:  # y+h>720
            box[3] = height - box[1]

        if box[2] > 0 and box[3] > 0 and box[0] < width and box[1] < height:
            boxes.append(box)
    boxes = np.array(boxes).reshape(-1, 5)
    return boxes


def filter_boxes(boxes, min_box_diag=60, min_box_side=20):
    """Filters boxes according to the paper rule.
    To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
    To note: we assume the initial time of the video is always 0
    :param boxes: (np.ndarray)
                 structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence']
                 (example BBOX_DTYPE is provided in src/box_loading.py)
    Returns:
        boxes: filtered boxes
    """
    width = boxes[:, 2]
    height = boxes[:, 3]
    diag_square = width ** 2 + height ** 2
    mask = (diag_square >= min_box_diag ** 2) * (width >= min_box_side) * (height >= min_box_side)
    return boxes[mask]


def draw_bboxes(img, boxes, labelmap):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i][0]), int(boxes[i][1]))
        size = (int(boxes[i][2]), int(boxes[i][3]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        class_id = int(boxes[i][4])
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 2)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


def generate_input_representation(events, shape):
    """
    :param events: [N, 4], where cols are (x, y, t, polarity) and polarity is in {-1, +1}
    :param events: H x W
    """
    pos_events = events[events[:, -1] == 1.0]
    neg_events = events[events[:, -1] == 0.0]
    pos_voxel_grid = generate_event_voxel_grid(pos_events, shape)
    neg_voxel_grid = abs(generate_event_voxel_grid(neg_events, shape))
    return np.concatenate([pos_voxel_grid, neg_voxel_grid], -1)


def generate_event_voxel_grid(events, shape, num_bins=5):
    """
    :param events: (x, y, t, polarity)
    :param shape: [N, 4]
    :param num_bins:
    :return: (tensor) [H, W, num_bins]
    """
    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    events = events.astype(np.float32)

    height, width = shape
    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
    delta_t = (events[-1, 2] - events[0, 2])  # last_stamp - first_stamp
    if delta_t == 0:
        delta_t = 1.0
    events[:, 2] = (num_bins - 1) * (events[:, 2] - events[0, 2]) / delta_t

    xs = events[:, 0].astype(np.int)
    ys = events[:, 1].astype(np.int)
    ts = events[:, 2]
    pols = events[:, 3]

    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts
    valid_indices = tis < num_bins
    np.add.at(voxel_grid,
              xs[valid_indices] + ys[valid_indices] * width + tis[valid_indices] * width * height,
              vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid,
              xs[valid_indices] + ys[valid_indices] * width + (tis[valid_indices]+1) * width * height,
              vals_right[valid_indices])
    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid.transpose(1, 2, 0)


def normalize(histogram):
    """standard normalize"""
    nonzero_ev = (histogram != 0) + 0
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        mean = histogram.sum() / num_nonzeros
        stddev = np.sqrt((histogram ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.astype(np.float)
        histogram = mask * (histogram - mean) / (stddev + 1e-8)
    # nonzero_ev = (histogram != 0)
    # num_nonzeros = nonzero_ev.sum()
    # if num_nonzeros > 0:
    #     mean = histogram.sum() / num_nonzeros
    #     stddev = np.sqrt((histogram ** 2).sum() / num_nonzeros - mean ** 2)
    #     histogram = nonzero_ev * (histogram - mean) / (stddev + 1e-8)

    return histogram


def play_files_parallel(root_dir, height, width):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    event_dir = os.path.join(root_dir, "events")
    label_dir = os.path.join(root_dir, "labels")

    event_files = os.listdir(event_dir)
    label_files = os.listdir(label_dir)

    event_list, label_list = [], []
    for ev, lb in zip(event_files, label_files):
        event_path = os.path.join(event_dir, ev)
        label_path = os.path.join(label_dir, lb)
        for e, l in zip(sorted(os.listdir(event_path)), sorted(os.listdir(label_path))):
            event_path_ = os.path.join(event_path, e)
            label_path_ = os.path.join(label_path, l)
            event_list.append(event_path_), label_list.append(label_path_)

    assert len(event_list) == len(label_list)
    for idx in range(len(event_list)):
        # use the naming pattern to find the corresponding box file
        labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

        frame = np.zeros((height, width, 10), dtype=np.uint8)
        cv2.namedWindow('gt', cv2.WINDOW_NORMAL)

        # load events and boxes from all files
        events = np.load(event_list[idx])
        boxes = np.load(label_list[idx])

        for npz_num in range(len(events)):
            ev_npz = "e" + str(npz_num)
            lb_npz = "l" + str(npz_num)
            events_ = events[ev_npz]
            boxes_ = boxes[lb_npz]

            events_ = rfn.structured_to_unstructured(events_)[:, [1, 2, 0, 3]]  # (x, y, t, p)
            boxes_ = rfn.structured_to_unstructured(boxes_)[:, [1, 2, 3, 4, 5]]  # (x, y, w, h, class_id)
            boxes_ = cropToFrame(boxes_, height, width)
            boxes_ = filter_boxes(boxes_, 60, 20)  # filter boxes
            histogram = generate_input_representation(events_, (height, width))
            # histogram = abs(normalize(histogram))
            histogram = histogram[..., 0:3].copy()
            draw_bboxes(histogram, boxes_, labelmap=labelmap)

            # display the result
            cv2.imshow('gt', histogram[..., :3])
            cv2.waitKey(1)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument(
        '-r', '--records', type=str,
        default="/home/wds/Desktop/exp_prophesee_still/test/testfilelist00",
        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('--height', default=720, type=int, help="image height")
    parser.add_argument('--width', default=1280, type=int, help="image width")

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    play_files_parallel(ARGS.records, ARGS.height, ARGS.width)
