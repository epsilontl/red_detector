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


DELTA_T = 1/60 * 1e6  # 16.67ms
SKIP_T = 50000 * 10  # skip the first 0.5s


def split_file(file_path, save_path, file_name, delta_t=None, skip=None):
    event_videos = [PSEELoader(file_path)]
    box_videos = [PSEELoader(glob(file_path.split('_td.dat')[0] + '*.npy')[0])]

    for v in event_videos + box_videos:
        v.seek_time(skip)

    idx = 0  # 50ms
    step = 0  # 16.6ms
    while not sum([video.done for video in event_videos]):
        boxes = [video.load_delta_t(delta_t) for video in box_videos]
        event_name = file_name[:-4] + '_' + ("%04d" % idx) + ".npy"  # dat_name + 0000 + .npy
        box_name = file_name[:-6] + "bbox_" + ("%04d" % idx) + ".npy"
        save_event_file = os.path.join(save_path, "events", event_name)
        save_box_file = os.path.join(save_path, "labels", box_name)

        if step % 3 == 2:
            events = [video.load_delta_t(delta_t*3) for video in event_videos]
            if events[0].shape[0] != 0:  # save .npy file if not empty
                np.save(save_event_file, events[0])
                np.save(save_box_file, boxes[0])
            else:  # no events
                print("\033[0;33m Sequence %d [%s] has no event streams! \033[0m" % (idx, event_name))
            idx += 1
        step += 1


if __name__ == "__main__":
    dataset_dir = "/home/wds/Desktop/trainfilelist14/train"
    save_dir = "/home/wds/Desktop/split_train14/train"
    files = os.listdir(dataset_dir)
    files = [time_seq_name for time_seq_name in files if time_seq_name[-3:] == 'dat']

    print("\033[0;31mStarting to splitting the dataset! \033[0m")
    pbar = tqdm.tqdm(total=len(files), unit="File", unit_scale=True)
    for file in files:
        abs_path = os.path.join(dataset_dir, file)
        split_file(abs_path, save_dir, file, delta_t=DELTA_T, skip=SKIP_T)  # skip the first 0.5s
        pbar.update()
    pbar.close()
    print("\033[0;31mDataset is already split! \033[0m")

