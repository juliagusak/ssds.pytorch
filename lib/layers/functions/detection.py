import torch
from torch.autograd import Function
from lib.utils.box_utils import decode, nms



class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg, priors):
        self.num_classes = cfg.NUM_CLASSES
        self.background_label = cfg.BACKGROUND_LABEL
        self.top_k = cfg.MAX_DETECTIONS 
        # Parameters used in nms.
        self.nms_thresh = cfg.IOU_THRESHOLD 
        self.conf_thresh = cfg.SCORE_THRESHOLD
        self.variance = cfg.VARIANCE
        self.priors = priors


    def forward(self, predictions):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc, conf = predictions

        loc_data = loc.data
        conf_data = conf.data
        prior_data = self.priors.data

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
                c_mask = conf_scores[cl].gt(self.conf_thresh).nonzero().view(-1)
                if c_mask.dim() == 0:
                    continue
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0 or scores.dim() == 0:
                    continue
                # l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # boxes = decoded_boxes[l_mask].view(-1, 4)
                boxes = decoded_boxes[c_mask, :]
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output
