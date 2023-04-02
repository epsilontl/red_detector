import os
import numpy as np
import event_representations as er
from numpy.lib import recfunctions as rfn
from torchvision import transforms
import torch
import cv2
import numba as nb


def getDataloader(name):
    dataset_dict = {"Prophesee": Prophesee}
    return dataset_dict.get(name)


class Prophesee:
    def __init__(self, root, object_classes, height, width, mode="training",
                 event_representation="histogram", resize=None):
        """
        Creates an iterator over the Prophesee object recognition dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or "all" for all classes
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

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.augmentation = SSDAugmentation(size=512)
        self.event_representation = event_representation
        self.resize = resize

        filelist_path = os.path.join(self.root, self.mode)

        self.event_files, self.label_files, self.index_files = self.load_data_files(filelist_path, self.root, self.mode)

        assert len(self.event_files) == len(self.label_files)

        if object_classes == "all":
            self.nr_classes = 8  # 7 classes and 1 background
            self.object_classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']

        self.nr_samples = len(self.event_files)
        # self.nr_samples = len(self.event_files) - len(self.index_files)*batch_size
        self.scale = np.array([self.width, self.height, self.width, self.height], dtype=np.float32)

    def __len__(self):
        return len(self.event_files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from split .npy files
        :param idx:
        :return: events: (x, y, t, p)
                 boxes: (N, 4), which is consist of (x_min, y_min, x_max, y_max)
                 histogram: (512, 512, 10)
        """
        boxes_list, histogram_list = [], []
        bbox_file = os.path.join(self.root, self.mode, "labels", self.label_files[idx])
        event_file = os.path.join(self.root, self.mode, "events", self.event_files[idx])

        labels_np = np.load(bbox_file)
        events_np = np.load(event_file)
        for npz_num in range(len(labels_np)):
            try:
                ev_npz = "e" + str(npz_num)
                lb_npz = "l" + str(npz_num)
                events_np_ = events_np[ev_npz]
                labels_np_ = labels_np[lb_npz]
            except:   # avoid error: Bad CRC-32 for file
                ev_npz = "e" + str(npz_num-1)
                lb_npz = "l" + str(npz_num-1)
                events_np_ = events_np[ev_npz]
                labels_np_ = labels_np[lb_npz]

            mask = (events_np_['x'] < 1280) * (events_np_['y'] < 720)  # filter events which are out of bounds
            events_np_ = events_np_[mask]

            labels = rfn.structured_to_unstructured(labels_np_)[:, [1, 2, 3, 4, 5]]  # (x, y, w, h, class_id)
            events = rfn.structured_to_unstructured(events_np_)[:, [1, 2, 0, 3]]  # (x, y, t, p)
            histogram = self.generate_input_representation(events, (self.height, self.width))

            bounding_boxes = self.cropToFrame(labels)
            bounding_boxes = self.filter_boxes(bounding_boxes, 60, 20)  # filter small boxes
            bounding_boxes[:, 2:-1] += bounding_boxes[:, :2]   # [x_min, y_min, x_max, y_max, class_id]

            histogram, bounding_boxes = self.augmentation(histogram, bounding_boxes)
            boxes = bounding_boxes.astype(np.float32)
            # histogram = self.normalize(histogram)  # image normalize
            boxes_list.append(boxes), histogram_list.append(histogram)

        boxes = np.array(boxes_list)
        histogram = np.array(histogram_list)
        return boxes, histogram

    def normalize(self, histogram):
        """standard normalize"""
        nonzero_ev = (histogram != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = histogram.sum() / num_nonzeros
            stddev = np.sqrt((histogram ** 2).sum() / num_nonzeros - mean ** 2)
            histogram = nonzero_ev * (histogram - mean) / (stddev + 1e-8)
        return histogram

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        boxes = []
        for box in np_bbox:
            # if box[2] > 1280 or box[3] > 800:  # filter error label
            if box[2] > 1280:
                continue

            if box[0] < 0:  # x < 0 & w > 0
                box[2] += box[0]
                box[0] = 0
            if box[1] < 0:  # y < 0 & h > 0
                box[3] += box[1]
                box[1] = 0
            if box[0] + box[2] > self.width:  # x+w>1280
                box[2] = self.width - box[0]
            if box[1] + box[3] > self.height:  # y+h>720
                box[3] = self.height - box[1]

            if box[2] > 0 and box[3] > 0 and box[0] < self.width and box[1] <= self.height:
                boxes.append(box)
        boxes = np.array(boxes).reshape(-1, 5)
        return boxes

    def filter_boxes(self, boxes, min_box_diag=60, min_box_side=20):
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
            pos_events = events[events[:, -1] == 1.0]
            neg_events = events[events[:, -1] == 0.0]
            if not len(neg_events):  # empty
                neg_events = pos_events
            pos_voxel_grid = self.generate_event_voxel_grid(pos_events, shape)
            neg_voxel_grid = self.generate_event_voxel_grid(neg_events, shape)
            # return torch.cat([pos_voxel_grid, neg_voxel_grid], -1)
            return np.concatenate([pos_voxel_grid, neg_voxel_grid], -1)

    @staticmethod
    def generate_event_histogram(events, shape):
        """
        :param events: (x, y, t, polarity)
        :param shape: [N, 4]
        """
        H, W = shape
        x, y, t, p = events.T
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

    @staticmethod
    @nb.jit()
    def load_data_files(filelist_path, root, mode):
        idx = 0
        event_files = []
        label_files = []
        index_files = []
        filelist_dir = sorted(os.listdir(filelist_path))
        for filelist in filelist_dir:
            event_path = os.path.join(root, mode, filelist, "events")
            label_path = os.path.join(root, mode, filelist, "labels")
            data_dirs = sorted(os.listdir(event_path))

            for dirs in data_dirs:
                event_path_sub = os.path.join(event_path, dirs)
                label_path_sub = os.path.join(label_path, dirs)
                event_path_list = sorted(os.listdir(event_path_sub))
                label_path_list = sorted(os.listdir(label_path_sub))
                idx += len(event_path_list) - 1
                index_files.append(idx)

                for ev, lb in zip(event_path_list, label_path_list):
                    event_root = os.path.join(event_path_sub, ev)
                    label_root = os.path.join(label_path_sub, lb)
                    event_files.append(event_root)
                    label_files.append(label_root)
        return event_files, label_files, index_files

    def file_index(self):
        return self.index_files


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None):
        for t in self.transforms:
            img, boxes = t(img, boxes)
        return img, boxes


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None):
        # normalize
        boxes[:, :-1] /= np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]], dtype=np.float32)
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return image, boxes


class SSDAugmentation(object):
    def __init__(self, size=300):
        self.size = size
        self.augment = Compose([Resize(self.size)])

    def __call__(self, img, boxes):
        return self.augment(img, boxes)

