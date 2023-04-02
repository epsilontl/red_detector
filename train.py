"""
Example usage: CUDA_VISIBLE_DEVICES=1, python train.py --settings_file "config/settings.yaml"
"""
import argparse
import os
import abc
import tqdm
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import dataloader.dataset
from models.red_network import RED
from dataloader.loader import Loader
from models.functions.multibox_loss import MultiBoxLoss
from models.functions.auxiliary_loss_end import AuxiliaryLossEnd
from models.functions.ssd_detection import SSD_Detect
from models.functions.smooth_l1_loss import Smooth_L1_Loss
from config.settings import Settings
from models.box_utils import box_iou
from utils.metrics import ap_per_class
from apex import amp


class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.scheduler = None
        self.nr_classes = None   # numbers of classes
        self.train_loader = None
        self.val_loader = None
        self.nr_train_epochs = None
        self.nr_val_epochs = None
        self.train_file_indexes = None
        self.val_file_indexes = None
        self.object_classes = None
        self.prior = 0.01

        if self.settings.event_representation == "histogram":
            self.nr_input_channels = 2
        elif self.settings.event_representation == "event_queue":
            self.nr_input_channels = 30
        elif self.settings.event_representation == "voxel_grid":
            self.nr_input_channels = 10

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)  # Prophesee
        self.dataset_loader = Loader
        self.writer = SummaryWriter(self.settings.ckpt_dir)

        self.createDatasets()  # train_dataset and val_dataset

        self.buildModel()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.settings.init_lr)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.settings.init_lr, momentum=0.9, weight_decay=5e-4)
        # mixed precision training
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=settings.exponential_decay)

        self.batch_step = 0
        self.epoch_step = 0
        self.training_loss = 0
        self.training_accuracy = 0
        self.max_validation_mAP = 0
        self.softmax = nn.Softmax(dim=-1)
        self.smooth_l1_loss = Smooth_L1_Loss(beta=0.11, reduction="sum")

        # tqdm progress bar
        self.pbar = None

        if settings.resume_training:
            self.loadCheckpoint(self.settings.resume_ckpt_file)

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = self.dataset_builder(self.settings.dataset_path,
                                             self.settings.object_classes,
                                             self.settings.height,
                                             self.settings.width,
                                             mode="training",
                                             event_representation=self.settings.event_representation,
                                             resize=self.settings.resize)
        self.train_file_indexes = train_dataset.file_index()
        self.nr_train_epochs = train_dataset.nr_samples // self.settings.batch_size
        self.nr_classes = train_dataset.nr_classes
        self.object_classes = train_dataset.object_classes

        val_dataset = self.dataset_builder(self.settings.dataset_path,
                                           self.settings.object_classes,
                                           self.settings.height,
                                           self.settings.width,
                                           mode="validation",
                                           event_representation=self.settings.event_representation,
                                           resize=self.settings.resize)
        self.val_file_indexes = val_dataset.file_index()
        self.nr_val_epochs = val_dataset.nr_samples

        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.gpu_device, drop_last=True, shuffle=True,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=False,
                                                data_index=self.train_file_indexes)
        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                              device=self.settings.gpu_device, drop_last=False, shuffle=False,
                                              num_workers=self.settings.num_cpu_workers, pin_memory=False,
                                              data_index=self.val_file_indexes)

    def storeLossesObjectDetection(self, loss_list):
        """Writes the different losses to tensorboard"""
        loss_names = ["Location_Loss", "Confidence_Loss", "Auxiliary_Loss1", "Auxiliary_Loss2", "Overall_Loss"]

        for idx in range(len(loss_list)):
            loss_value = loss_list[idx].data.cpu().numpy()
            self.writer.add_scalar("TrainingLoss/" + loss_names[idx], loss_value, self.batch_step)

    def getLearningRate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def loadCheckpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.epoch_step = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def saveCheckpoint(self):
        file_path = os.path.join(self.settings.ckpt_dir, "model_step_" + str(self.epoch_step) + ".pth")
        torch.save({"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "epoch": self.epoch_step}, file_path)
        # torch.save({"state_dict": self.model.state_dict()}, file_path)


class RecurrentEventDetection(AbstractTrainer):
    def buildModel(self):
        """Creates the specified model"""
        self.model = RED(num_classes=self.nr_classes, in_channels=self.nr_input_channels, cfg=self.settings)
        cudnn.benchmark = True
        print("\033[0;33m Starting to initialize parameters! \033[0m")
        self.params_initialize()  # initialize
        self.model.to(self.settings.gpu_device)

        if self.settings.use_pretrained and self.settings.dataset_name == "Prophesee":
            self.loadPretrainedWeights()

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            m.bias.data.zero_()

    def params_initialize(self):
        self.model.layer1.apply(self.weights_init)
        self.model.layer2.apply(self.weights_init)  # SE_layer
        self.model.layer3.apply(self.weights_init)

        for loc1, loc2 in zip(self.model.loc_layer1, self.model.loc_layer2):  # loc_layer
            loc1.apply(self.weights_init)
            loc2.apply(self.weights_init)

        for cls1, cls2 in zip(self.model.conf_layer1, self.model.conf_layer2):  # conf_layer
            cls1.apply(self.weights_init)
            cls2.apply(self.weights_init)
            for idx, b in enumerate(cls1.bias):
                if idx % self.nr_classes == 0:  # first class is background, set bias
                    cls1.bias.data[idx] = torch.tensor(math.log((1.0 - self.prior) / self.prior*(self.nr_classes-1)))
            # for idx, b in enumerate(cls2.bias):
            #     if idx % self.nr_classes == 0:  # first class is background, set bias
            #         cls2.bias.data[idx] = torch.tensor(math.log((1.0 - self.prior) / self.prior*(self.nr_classes-1)))

    def loadPretrainedWeights(self):
        """Loads pretrained model weights"""
        checkpoint = torch.load(self.settings.pretrained_model)
        try:
            pretrained_dict = checkpoint["state_dict"]
        except KeyError:
            pretrained_dict = checkpoint["model"]

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "conv_layers." in k and int(k[12]) <= 4}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def train(self):
        """Main training and validation loop"""
        validation_step = 50 - 45 * (self.settings.dataset_name == "Prophesee")

        while self.epoch_step <= self.settings.epoch:
            self.trainEpoch()
            self.validationEpoch()
            # if (self.epoch_step % validation_step) == (validation_step - 1):
            #     self.validationEpoch()

            self.epoch_step += 1
            # self.scheduler.step()

    def trainEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit="Batch", unit_scale=True,
                              desc="Epoch: {}".format(self.epoch_step))
        self.model = self.model.train()
        multibox_loss = MultiBoxLoss(num_classes=self.nr_classes,
                                     overlap_thresh=0.5,
                                     variance=self.settings.variance,
                                     use_gpu=True,
                                     device=self.settings.gpu_device)
        # Smooth L1 Loss
        auxiliary_loss_end = AuxiliaryLossEnd(num_classes=self.nr_classes,
                                              overlap_thresh=0.5,
                                              variance=self.settings.variance,
                                              use_gpu=True,
                                              device=self.settings.gpu_device)
        # store previous model output
        out_front = torch.zeros([0, 4]).to(self.settings.gpu_device)
        for i_batch, sample_batched in enumerate(self.train_loader):
            prev_states = None
            bounding_box, histogram = sample_batched
            histogram = histogram.permute(1, 0, 4, 2, 3)  # [seq_len, batch, channel, H, W]
            # self.optimizer.zero_grad()
            batch_loss_r, batch_loss_c, batch_loss_t1, batch_loss_t2 = 0, 0, 0, 0
            batch_loss_t2_c = 0
            for idx, his in enumerate(histogram):  # for each length of sequence
                bounding_box_now, bounding_box_next = [], []
                out = self.model(his, prev_states=prev_states)
                prev_states = out[2]
                if idx in self.settings.tbptt:
                    for state in prev_states:
                        state[0] = state[0].detach()
                        state[1] = state[1].detach()
                for idy in range(self.settings.batch_size):  # for each batch
                    mask_now = (bounding_box[:, -1] == (idy*self.settings.seq_len+idx))
                    mask_next = (bounding_box[:, -1] == (idy*self.settings.seq_len+idx+1))
                    bbox_now = bounding_box[mask_now][:, :-1]
                    bbox_next = bounding_box[mask_next][:, :-1]
                    bounding_box_now.append(bbox_now), bounding_box_next.append(bbox_next)

                # main loss (conf loss + reg loss)
                loss_r, loss_c, recur_head, N = multibox_loss(out[0], bounding_box_now)
                batch_loss_r += sum(loss_r)
                batch_loss_c += sum(loss_c)

                # aux1 loss
                if idx != 0:  # sequence front
                    for tdx in range(self.settings.batch_size):
                        loss_t1 = self.smooth_l1_loss(out_front[tdx], recur_head[tdx]) / N[tdx]
                        batch_loss_t1 += loss_t1

                # aux2 loss
                if idx != (self.settings.seq_len - 1):  # sequence end
                    loss_t2, loss_t2_c, prev_head = auxiliary_loss_end(out[1], bounding_box_next)  # Ls(Bk+1', Bk+1*)
                    batch_loss_t2 += sum(loss_t2)
                    batch_loss_t2_c += sum(loss_t2_c)
                    out_front = prev_head

            total_loss = batch_loss_r + batch_loss_c + batch_loss_t2 + batch_loss_t1
            if total_loss != 0:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # total_loss.backward()
            # to visualize loss curve better
            loss_r = torch.tensor(0, dtype=torch.float32) if batch_loss_r == 0 else batch_loss_r
            loss_c = torch.tensor(0, dtype=torch.float32) if batch_loss_c == 0 else batch_loss_c
            loss_t1 = torch.tensor(0, dtype=torch.float32) if batch_loss_t1 == 0 else batch_loss_t1
            loss_t2 = torch.tensor(0, dtype=torch.float32) if batch_loss_t2 == 0 else batch_loss_t2
            total_loss = torch.tensor(0, dtype=torch.float32) if not total_loss else total_loss

            self.pbar.set_postfix(Total_Loss=total_loss.item(),
                                  Loc=loss_r.item(),
                                  Conf=loss_c.item(),
                                  Aux1=loss_t1.item(),
                                  Aux2=loss_t2.item())
            loss_list = [loss_r, loss_c, loss_t1, loss_t2, total_loss]
            # Write losses statistics
            self.storeLossesObjectDetection(loss_list)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.pbar.update(1)
            self.batch_step += 1

        self.writer.add_scalar("Training/Learning_Rate", self.getLearningRate(), self.epoch_step)
        self.pbar.close()

    def validationEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit="Batch", unit_scale=True)
        self.model = self.model.eval()
        iouv = torch.linspace(0.5, 0.95, 10).to(self.settings.gpu_device)
        niou = iouv.numel()  # len(iouv)
        ssd_detector = SSD_Detect(num_classes=self.nr_classes, bg_label=0, top_k=200,
                                  conf_thresh=0.4,
                                  nms_thresh=0.5,
                                  settings=self.settings)

        seen = 0
        precision, recall, f1_score, m_precision, m_recall, map50, map = 0., 0., 0., 0., 0., 0., 0.
        stats, ap, ap_class = [], [], []
        scale = torch.tensor([self.settings.width, self.settings.height, self.settings.width, self.settings.height],
                             dtype=torch.float32).to(self.settings.gpu_device)

        prev_states = None
        for i_batch, sample_batched in enumerate(self.val_loader):
            detection_result = []  # detection result for computing mAP
            bounding_box, histogram = sample_batched
            histogram = histogram.permute(0, 3, 1, 2)
            with torch.no_grad():
                for idx, his in enumerate(histogram):
                    # Deep Learning Magic
                    sample_detection = torch.tensor([]).reshape(-1, 6)
                    out = self.model(his.unsqueeze(0), prev_states=prev_states)
                    prev_states = out[2]

                    detected_bbox = ssd_detector(out[0][0],  # loc preds
                                                 self.softmax(out[0][1]),  # conf preds
                                                 out[0][2]).squeeze(0)  # default boxes
                    for j in range(1, detected_bbox.shape[0]):  # j = 0 background
                        dets_ = detected_bbox[j]  # [score, x1, y1, x2, y2]
                        mask = (dets_[:, 0] != 0.)
                        dets_ = dets_[mask]
                        if dets_.size(0) == 0:
                            continue
                        dets_cls = torch.full([dets_.size(0), 1], j - 1)  # cls
                        dets = torch.cat([dets_, dets_cls], 1)
                        dets = dets[:, [1, 2, 3, 4, 0, 5]]  # [x1, y1, x2, y2, cls, score]
                        sample_detection = torch.cat([sample_detection, dets], 0)
                    detection_result.append(sample_detection)
            if (i_batch*2 in self.val_file_indexes) or (i_batch*2+1 in self.val_file_indexes):
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
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)  # make dir

        names = {k: v for k, v in enumerate(self.object_classes, start=0)}  # {0:'Pedestrian', ...}
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            precision, recall, ap, f1_score, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(axis=1)  # AP@0.5, AP@0.5:0.95
            m_precision, m_recall, map50, map = precision.mean(), recall.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.nr_classes - 1)  # number of targets per class
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

        self.writer.add_scalar("Validation/Validation_mAP", map50, self.epoch_step)

        if self.max_validation_mAP < map50:
            self.max_validation_mAP = map50
            self.saveCheckpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network.")
    parser.add_argument("--settings_file", type=str, default="/home/dut/code/red_detector/config/settings.yaml",help="Path to settings yaml")

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=True)

    if settings.model_name == "red":
        trainer = RecurrentEventDetection(settings)
    else:
        raise ValueError("Model name %s specified in the settings file is not implemented" % settings.model_name)

    trainer.train()
