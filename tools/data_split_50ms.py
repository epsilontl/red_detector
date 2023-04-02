"""
To read files more faster,
this file is to split .dat or .npy file into 1200 sequences. (60s/50ms=1200)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from glob import glob
from dataloader.prophesee.src.io.psee_loader import PSEELoader
import tqdm
from numpy.lib import recfunctions as rfn


DELTA_T = 50000
SKIP_T = 50000 * 10  # skip the first 0.5s
HEIGHT = 720
WIDTH = 1280
TYPE = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')]


def cropToFrame(np_bbox, height, width):
    """Checks if bounding boxes are inside frame. If not crop to border"""
    boxes = []
    for box in np_bbox:
        if box[2] > 1280 or box[3] > 800:  # filter error label
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


def split_file(file_path, save_path, file_name, height, width, delta_t=None, skip=None):
    event_videos = [PSEELoader(file_path)]
    box_videos = [PSEELoader(glob(file_path.split('_td.dat')[0] + '*.npy')[0])]

    for v in event_videos + box_videos:
        v.seek_time(skip)

    idx = 0
    while not sum([video.done for video in event_videos]):
        boxes_filter = []
        track_id = []
        events = [video.load_delta_t(delta_t) for video in event_videos]
        boxes = [video.load_delta_t(delta_t) for video in box_videos]

        for box in boxes[0]:
            if box["track_id"] not in track_id:
                boxes_filter.append(box)  # filter same object id in each 50ms
                track_id.append(box["track_id"])

        boxes = [np.array(boxes_filter).astype(TYPE)]

        event_name = file_name[:-4] + '_' + ("%04d" % idx) + ".npy"  # dat_name + 0000 + .npy
        box_name = file_name[:-6] + "bbox_" + ("%04d" % idx) + ".npy"
        save_event_file = os.path.join(save_path, "events", event_name)
        save_box_file = os.path.join(save_path, "labels", box_name)

        # save .npy file if not empty
        if events[0].shape[0] != 0:
            np.save(save_event_file, events[0])
            np.save(save_box_file, boxes[0])
        else:
            print("\033[0;33m Sequence %d [%s] has no event streams! \033[0m" % (idx, event_name))
        idx += 1
        # if idx > 500:
        #     break


if __name__ == "__main__":
    dataset_dir = "/home/wds/Desktop/trainfilelist14/train"
    save_dir = "/home/wds/Desktop/split_train14_s2/train"
    files = os.listdir(dataset_dir)
    files = [time_seq_name for time_seq_name in files if time_seq_name[-3:] == 'dat']

    print("\033[0;31mStarting to splitting the dataset! \033[0m")
    pbar = tqdm.tqdm(total=len(files), unit="File", unit_scale=True)
    for file in files:
        abs_path = os.path.join(dataset_dir, file)
        # skip the first 0.5s
        split_file(abs_path, save_dir, file, height=HEIGHT, width=WIDTH, delta_t=DELTA_T, skip=SKIP_T)
        pbar.update()
    pbar.close()
    print("\033[0;31mDataset is already split! \033[0m")
