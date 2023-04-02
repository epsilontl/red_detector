"""Loading this dataset is too slow, so we split .dat and .npy file into smaller sequences."""
import os
import tqdm
import numpy as np
from os import listdir
import event_representations as er
from numpy.lib import recfunctions as rfn
import torch

from dataloader.prophesee.src.io import dat_events_tools
from dataloader.prophesee.src.io import npy_events_tools


EVENT_START = 50000  # 50ms
TOTAL_EVENT_PER_FILE = 1200   # a file contains 60s events, we split it into 60s/50ms=1200 sequences
DELTA_TIME = 50000  # 50ms delta time
SKIP_RATE = 10  # skip first 0.5s

DAT_TYPE = [('t', 'u4'), ('x', 'u2'), ('y', 'u2'), ('p', 'u1')]
NPY_TYPE = [('t', 'uint64'), ('x', 'float32'), ('y', 'float32'),
            ('w', 'float32'), ('h', 'float32'), ('class_id', 'uint8'),
            ('class_confidence', 'float32'), ('track_id', 'uint32')]


def getDataloader(name):
    dataset_dict = {"Prophesee": Prophesee}
    return dataset_dict.get(name)


class Prophesee:
    def __init__(self, root, object_classes, height, width, augmentation=False, mode="training",
                 event_representation="histogram"):
        """
        Creates an iterator over the Prophesee object recognition dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param augmentation: flip, shift and random window start for training
        :param mode: "training", "testing" or "validation"
        :param event_representation: "histogram","event_queue" or "voxel_grid"
        """
        if mode == "training":
            mode = "train"
        elif mode == "validation":
            mode = "val"
        elif mode == "testing":
            mode = "test"

        file_dir = os.path.join("detection_dataset_duration_60s_ratio_1.0", mode)
        self.files = listdir(os.path.join(root, file_dir))
        # Remove duplicates (.npy and .dat)
        self.files = [os.path.join(file_dir, time_seq_name[:-9]) for time_seq_name in self.files
                      if time_seq_name[-3:] == "npy"]

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.augmentation = augmentation
        self.event_representation = event_representation

        self.max_nr_bbox = 15

        if object_classes == "all":
            self.nr_classes = 3  # two classes and one background
            self.object_classes = ["Car", "Pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes

        self.sequence_start = []
        self.bbox_start = []

        self.createAllBBoxDataset(mode)
        self.nr_samples = len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        bbox_file = os.path.join(self.root, self.files[idx][0] + "_bbox.npy")
        event_file = os.path.join(self.root, self.files[idx][0] + "_td.dat")

        bounding_boxes = self.readBoxFile(bbox_file, self.bbox_start[idx][0], self.bbox_start[idx][1],
                                          nr_window_boxes=15)
        # Required Information ['x', 'y', 'w', 'h', 'class_id']
        np_bbox = rfn.structured_to_unstructured(bounding_boxes)[:, [1, 2, 3, 4, 5]]
        np_bbox = self.cropToFrame(np_bbox)

        np_bbox = np.array(self.xywh2xyxy(np_bbox)).reshape(-1, 5)

        const_size_bbox = np.zeros([self.max_nr_bbox, 5])
        const_size_bbox[:np_bbox.shape[0], :] = np_bbox

        # Events
        events = self.readEventFile(event_file, self.sequence_start[idx][0], self.sequence_start[idx][1],
                                    nr_window_events=400000)

        histogram = self.generate_input_representation(events, (self.height, self.width))

        return events, const_size_bbox.astype(np.float32), histogram

    def createAllBBoxDataset(self, mode):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the
         unique indices file.
        """
        file_name_bbox_id = []
        print("Building the Prophesee GEN1 Dataset for %s!" % mode)
        pbar = tqdm.tqdm(total=len(self.files), unit="File", unit_scale=True)
        zero_events = []
        for idx, file_name in enumerate(self.files):
            count_zero_event = 0
            bbox_file = os.path.join(self.root, file_name + "_bbox.npy")
            event_file = os.path.join(self.root, file_name + "_td.dat")
            f_bbox = open(bbox_file, "rb")
            start_b, v_type_b, ev_size_b, size_b = npy_events_tools.parse_header(f_bbox)

            f_event = open(event_file, "rb")
            start_e, v_type_e, ev_size_e, size_e = dat_events_tools.parse_header(f_event)

            end = f_event.seek(0, 2)  # the end of file
            end_bbox = f_bbox.seek(0, 2)

            unique_ts = np.arange(EVENT_START, DELTA_TIME*(TOTAL_EVENT_PER_FILE+1), DELTA_TIME)
            for unique_time in unique_ts:
                if unique_time == EVENT_START:
                    start = f_event.seek(start_e)
                    start_bbox = f_bbox.seek(start_b)
                else:
                    start = f_event.seek(ev_end)
                    start_bbox = f_bbox.seek(bbox_end)

                sequence_start, ev_end = self.searchEventSequence(f_event, start, ev_size_e,
                                                                  unique_time, nr_window_events=400000)
                bbox_start, bbox_end = self.searchBoundingBox(f_bbox, start_bbox, ev_size_b,
                                                              unique_time, nr_window_boxes=15)
                if sequence_start[1] == 0 or ev_end >= end:
                    count_zero_event += 1
                    continue
                if unique_time > DELTA_TIME * SKIP_RATE:  # filter first 0.5s sequences
                    self.sequence_start.append(sequence_start)
                    self.bbox_start.append(bbox_start)
            zero_events.append(count_zero_event)
            file_name_bbox_id += [[file_name, i] for i in range(len(unique_ts)-zero_events[idx]-SKIP_RATE)]
            pbar.update(1)

        pbar.close()
        self.files = file_name_bbox_id

    def searchEventSequence(self, f_event, start, ev_size, unique_time, nr_window_events=250000):
        buffer = np.empty((nr_window_events,), dtype=DAT_TYPE)
        dat_events_tools.stream_td_data(f_event, buffer, [('t', 'u4'), ('_', 'i4')], nr_window_events)
        idx = np.searchsorted(buffer['t'], unique_time)  # 50ms
        end = start + idx * ev_size

        return [start, idx], end

    def searchBoundingBox(self, f_bbox, start, ev_size, unique_time, nr_window_boxes=100):
        buffer = np.empty((nr_window_boxes,), dtype=NPY_TYPE)
        npy_events_tools.stream_td_data(f_bbox, buffer, NPY_TYPE, nr_window_boxes)
        idx = np.searchsorted(buffer['t'], unique_time)  # 50ms
        end = start + idx * ev_size
        return [start, idx], end

    def readEventFile(self, event_file, file_position, idx, nr_window_events=250000):
        file_handle = open(event_file, "rb")
        file_handle.seek(file_position)
        buffer = np.empty((nr_window_events,), dtype=DAT_TYPE)
        dat_events_tools.stream_td_data(file_handle, buffer, [('t', 'u4'), ('_', 'i4')], nr_window_events)
        events_np = buffer[:idx]
        events_np = rfn.structured_to_unstructured(events_np)[:, [1, 2, 0, 3]]
        return events_np

    def readBoxFile(self, box_file, file_position, idx, nr_window_boxes=100):
        file_handle = open(box_file, "rb")
        file_handle.seek(file_position)
        buffer = np.empty((nr_window_boxes,), dtype=NPY_TYPE)
        npy_events_tools.stream_td_data(file_handle, buffer, NPY_TYPE, nr_window_boxes)
        npy_box = buffer[:idx]
        return npy_box

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        array_width = np.ones_like(np_bbox[:, 0]) * self.width - 1
        array_height = np.ones_like(np_bbox[:, 1]) * self.height - 1

        np_bbox[:, :2] = np.maximum(np_bbox[:, :2], np.zeros_like(np_bbox[:, :2]))
        np_bbox[:, 0] = np.minimum(np_bbox[:, 0], array_width)
        np_bbox[:, 1] = np.minimum(np_bbox[:, 1], array_height)

        np_bbox[:, 2] = np.minimum(np_bbox[:, 2], array_width - np_bbox[:, 0])
        np_bbox[:, 3] = np.minimum(np_bbox[:, 3], array_height - np_bbox[:, 1])

        return np_bbox

    def generate_input_representation(self, events, shape):
        """
        :param events: [N, 4], where cols are (x, y, t, polarity) and polarity is in {-1, +1}
        :param events: H x W
        """
        if self.event_representation == "histogram":
            return self.generate_event_histogram(events, shape)
        elif self.event_representation == "event_queue":
            return self.generate_event_queue(events, shape)
        elif self.event_representation == "voxel_grid":
            return self.generate_event_voxel_grid(events, shape)

    @staticmethod
    def generate_event_histogram(events, shape):
        """
        :param events: (x, y, t, polarity)
        :param shape: [N, 4]
        """
        H, W = shape
        x, y, t, p = events.T  # T:transpose
        x = x.astype(np.int16)
        y = y.astype(np.int16)

        img_pos = np.zeros((H * W,), dtype="float32")
        img_neg = np.zeros((H * W,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + W * y[p == 1], 1)
        np.add.at(img_neg, x[p == -1] + W * y[p == -1], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((H, W, 2))
        return histogram

    @staticmethod
    def generate_event_queue(events, shape, K=15):
        H, W = shape
        events = events.astype(np.float32)

        if events.shape[0] == 0:
            return np.zeros([H, W, 2 * K], dtype=np.float32)

        # [2, K, height, width],  [0, ...] time, [:, 0, :, :] newest events
        four_d_tensor = er.event_queue_tensor(events, K, H, W, -1).astype(np.float32)

        # Normalize
        four_d_tensor[0, ...] = four_d_tensor[0, 0, None, :, :] - four_d_tensor[0, :, :, :]
        max_timestep = np.amax(four_d_tensor[0, :, :, :], axis=0, keepdims=True)

        # four_d_tensor[0, ...] = np.divide(four_d_tensor[0, ...], max_timestep, where=max_timestep.astype(np.bool))
        four_d_tensor[0, ...] = four_d_tensor[0, ...] / (max_timestep + (max_timestep == 0).astype(np.float))

        return torch.from_numpy(four_d_tensor.reshape([2 * K, H, W]).transpose(1, 2, 0))  # (H, W, 2*K)

    @staticmethod
    def generate_event_voxel_grid(events, shape, num_bins=5):
        """
        :param events: (x, y, t, polarity)
        :param shape: [N, 4]
        :param num_bins:
        """
        assert(events.shape[1] == 4)
        assert(num_bins > 0)
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
        pols[pols == 0] = -1

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
                  xs[valid_indices] + ys[valid_indices] * width + tis[valid_indices] * width * height,
                  vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
        return voxel_grid.transpose(1, 2, 0)

    def xywh2xyxy(self, np_bbox):
        """
        :param np_bbox: (left_x, left_y, w, h)
        return: (xmin, ymin, xmax, ymax) and normalized
        """
        normalized_bbox = []
        for bbox in np_bbox:
            width, height = 304, 240
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox[0] /= width  # x/w
            bbox[1] /= height  # y/h
            bbox[2] /= width
            bbox[3] /= height
            normalized_bbox.append(bbox)
        return normalized_bbox
