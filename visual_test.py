import argparse
import os
import abc
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

import dataloader.dataset
from models.red_network import RED
from dataloader.loader import Loader
from models.functions.ssd_detection import SSD_Detect
from config.settings import Settings
from models.box_utils import box_iou
from utils.metrics import ap_per_class
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.scheduler = None
        self.nr_classes = None   # numbers of classes
        self.test_loader = None
        self.object_classes = None
        self.nr_test_epochs = None
        self.test_file_indexes = None

        if self.settings.event_representation == "histogram":
            self.nr_input_channels = 2
        elif self.settings.event_representation == "event_queue":
            self.nr_input_channels = 30
        elif self.settings.event_representation == "voxel_grid":
            self.nr_input_channels = 10

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)  # Prophesee
        self.dataset_loader = Loader
        self.writer = SummaryWriter(self.settings.vis_dir)  # visualize feature map

        self.createDatasets()  # create test dataset

        self.buildModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.settings.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=settings.exponential_decay)

        self.softmax = nn.Softmax(dim=-1)

        # tqdm progress bar
        self.pbar = None

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        test_dataset = self.dataset_builder(self.settings.dataset_path,
                                            self.settings.object_classes,
                                            self.settings.height,
                                            self.settings.width,
                                            mode="testing",
                                            event_representation=self.settings.event_representation,
                                            resize=self.settings.resize)
        self.test_file_indexes = test_dataset.file_index()
        self.nr_test_epochs = test_dataset.nr_samples // self.settings.batch_size
        self.nr_classes = test_dataset.nr_classes
        self.object_classes = test_dataset.object_classes

        self.test_loader = self.dataset_loader(test_dataset, batch_size=self.settings.batch_size,
                                               device=self.settings.gpu_device, drop_last=False, shuffle=False,
                                               num_workers=self.settings.num_cpu_workers, pin_memory=False,
                                               data_index=self.test_file_indexes)


class RecurrentEventDetection(AbstractTrainer):
    def buildModel(self):
        """Creates the specified model"""
        self.model = RED(num_classes=self.nr_classes, in_channels=self.nr_input_channels, cfg=self.settings)
        self.model.to(self.settings.gpu_device)

    def test(self, args):
        self.pbar = tqdm.tqdm(total=self.nr_test_epochs, unit="Batch", unit_scale=True)
        self.model.load_state_dict(torch.load(args.weights)["state_dict"])
        self.model = self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10).to(self.settings.gpu_device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()  # len(iouv)

        ssd_detector = SSD_Detect(num_classes=self.nr_classes, bg_label=0, top_k=args.top_k,
                                  conf_thresh=args.conf_thresh,
                                  nms_thresh=args.iou_thresh,
                                  settings=self.settings)

        seen = 0
        precision, recall, f1_score, m_precision, m_recall, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        scale = torch.tensor([self.settings.width, self.settings.height, self.settings.width, self.settings.height],
                             dtype=torch.float32).to(self.settings.gpu_device)

        total_detection_result = []  # total detection result for computing mAP

        prev_states = None
        for i_batch, sample_batched in enumerate(self.test_loader):
            detection_result = []  # detection result for computing mAP
            bounding_box, histogram = sample_batched
            histogram = histogram.permute(0, 3, 1, 2)
            with torch.no_grad():
                for idx, his in enumerate(histogram):
                    # Deep Learning Magic
                    sample_detection = torch.tensor([]).reshape(-1, 6)
                    out = self.model(his.unsqueeze(0), prev_states=prev_states)
                    prev_states = out[2]

                    detected_bbox = ssd_detector(out[0][0],                 # loc preds
                                                 self.softmax(out[0][1]),   # conf preds
                                                 out[0][2]).squeeze(0)      # default boxes
                    for j in range(1, detected_bbox.shape[0]):  # j = 0 background
                        dets_ = detected_bbox[j]   # [score, x1, y1, x2, y2]
                        mask = (dets_[:, 0] != 0.)
                        dets_ = dets_[mask]
                        if dets_.size(0) == 0:
                            continue
                        dets_cls = torch.full([dets_.size(0), 1], j-1)  # cls
                        dets = torch.cat([dets_, dets_cls], 1)
                        dets = dets[:, [1, 2, 3, 4, 0, 5]]  # [x1, y1, x2, y2, cls, score]
                        sample_detection = torch.cat([sample_detection, dets], 0)
                    detection_result.append(sample_detection)
                    sample_detection_rescale = sample_detection.clone()
                    sample_detection_rescale[:, :4] *= scale.cpu()
                    total_detection_result.append(sample_detection_rescale.numpy())

                if i_batch == 1:
                    self.writer.add_image("Visualization/Input",
                                          make_grid(histogram[idx].unsqueeze(1).detach().cpu(), nrow=10,
                                                    normalize=False))
                    for hi, state in enumerate(prev_states):
                        hidden_state, ceil_state = state
                        if hi == 0:
                            self.writer.add_image("Visualization/Hidden_State_Feature_Map" + str(hi),
                                                  make_grid(hidden_state.permute(1, 0, 2, 3).detach().cpu(), nrow=2,
                                                            normalize=False))
                            # self.writer.add_image("Visualization/Ceil_State_Feature_Map" + str(hi),
                            #                       make_grid(ceil_state.permute(1, 0, 2, 3).detach().cpu(),
                            #                                 nrow=2, normalize=False))

            if (i_batch*2 in self.test_file_indexes) or (i_batch*2+1 in self.test_file_indexes):
                prev_states = None
            self.pbar.update(1)

            for si, pred in enumerate(detection_result):
                pred = pred.to(self.settings.gpu_device)
                np_labels = bounding_box[bounding_box[:, -1] == si, :-1]
                labels = np_labels[:, [4, 0, 1, 2, 3]]
                labels[:, 1:] *= scale
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.uint8), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                # predictions
                predn = pred.clone()
                predn[:, :4] *= scale
                # assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.uint8).to(self.settings.gpu_device)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]
                    # target boxes
                    tbox = labels[:, 1:5]
                    # per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                        pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices
                        # search for detections
                        if pi.shape[0]:
                            # prediction to target ious
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                            # append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero():
                                d = ti[i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    if len(detected) == nl:  # all targets already located in images
                                        break
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        self.pbar.close()

        # Directories
        save_dir = os.path.join(self.settings.save_dir, "det_result")
        npy_file = os.path.join(save_dir, "prediction.npy")  # save predictions
        if args.save_npy:
            np.save(npy_file, total_detection_result)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)   # make dir

        names = {k: v for k, v in enumerate(self.object_classes, start=0)}  # {0:'Pedestrian', ...}
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            precision, recall, ap, f1_score, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(axis=1)  # AP@0.5, AP@0.5:0.95
            m_precision, m_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nr_classes-1)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = "%8s" + "%18i" * 2 + "%19.3g" * 4  # print format
        print("\033[0;31m    Class            Events              Labels           Precision           Recall          "
              "   mAP@0.5           mAP@0.5:0.95 \033[0m")
        print(pf % ("all", seen, nt.sum(), m_precision, m_recall, map50, map))

        # Print results per class
        if len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], precision[i], recall[i], ap50[i], ap[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test network.")
    parser.add_argument("--settings_file", type=str, default="/home/wds/Desktop/red_detector/config/settings.yaml",
                        help="Path to settings yaml")
    parser.add_argument("--weights", type=str,
                        default="/home/wds/Desktop/red_detector/log/baseline/checkpoints/model_step_23.pth",
                        help="model.pth path(s)")
    parser.add_argument("--top_k", type=int, default=200,
                        help="top_k number of model predictions for both confidence scores and locations")
    parser.add_argument("--conf_thresh", type=float, default=0.4,
                        help="object confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="IoU threshold for NMS")
    parser.add_argument("--save_npy", type=bool, default=True,
                        help="save detection results(predicted bounding boxes), .npy file for visualization")

    args = parser.parse_args()

    settings = Settings(args.settings_file, generate_log=True)

    if settings.model_name == "red":
        tester = RecurrentEventDetection(settings)
    else:
        raise ValueError("Model name %s specified in the settings file is not implemented" % settings.model_name)

    tester.test(args)
