import torch
from torch.autograd import Function
from models.box_utils import decode, nms


class SSD_Detect(Function):
    """
    At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf scores
    and threshold to a top_k number of output predictions for both confidence score and locations.
    """
    def __init__(self, num_classes, bg_label, top_k, conf_thresh, nms_thresh, settings):
        self.num_classes = num_classes
        self.background_label = bg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")
        self.conf_thresh = conf_thresh
        self.variance = settings.variance
        self.device = settings.gpu_device
        self.scale = torch.tensor([settings.width, settings.height, settings.width, settings.height],
                                  dtype=torch.float32).to(self.device)

    def forward(self, loc_data, conf_data, prior_data):
        """
        :param loc_data: [batch, num_priors, 4], (tensor) Loc preds from loc layers
        :param conf_data: [batch, num_priors, num_classes], (tensor) Conf preds from conf layers
        :param prior_data: [num_priors, 4], (tensor) Prior boxes and variances from priorbox layers
        :return output: [batch, top_k, 6], 6 means [xmin, ymin, xmax, ymax, score, cls_id]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
