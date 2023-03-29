import logging
import torch
from torch import nn


from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import pairwise_iou

from .condinst import CondInst


__all__ = ["BoxTeacher"]
logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class BoxTeacher(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.eval_teacher = cfg.MODEL.BOX_TEACHER.TEACHER_EVAL
        self.use_teacher_inference = cfg.MODEL.BOX_TEACHER.USE_TEACHER_INFERENCE
        # iou thr
        self.filter_iou_thr = cfg.MODEL.BOX_TEACHER.IOU_THR
        self.filter_score_thr = cfg.MODEL.BOX_TEACHER.SCORE_THR

        self.teacher = CondInst(cfg)
        self.student = CondInst(cfg)
        self.freeze_teacher()

    def pseudo_labeling(self, batched_inputs, instances, iou_thr=0.5, score_thr=0.0):

        for idx, instances_per_im in enumerate(instances):
            gt_instances = batched_inputs[idx]["instances"]
            pred_instances = instances_per_im["instances"]
            M = len(pred_instances)
            N = len(gt_instances)
            # print(M, N)
            if N == 0 or not pred_instances.has("pred_masks"):
                batched_inputs[idx]["instances"].gt_masks_flags = torch.zeros(
                    (N,), dtype=torch.float, device=self.device)
                continue
            pred_masks = pred_instances.pred_masks.to(self.device)

            gtboxes = gt_instances.gt_boxes.to(self.device)
            pred_boxes = pred_instances.pred_boxes.to(self.device)
            pred_scores = pred_instances.scores.to(self.device)
            iou = pairwise_iou(pred_boxes, gtboxes)
            
            sort_index = torch.argsort(pred_scores, descending=True)
            dtm = torch.zeros((M,), dtype=torch.long, device=self.device)
            gtm = torch.zeros((N,), dtype=torch.long, device=self.device) - 1
            biou = torch.zeros((N,), dtype=torch.float, device=self.device)

            for i in sort_index:
                if pred_scores[i] < score_thr:
                    continue
                max_iou = -1
                m = -1
                for j in range(N):
                    iou_ = iou[i, j]
                    if gtm[j] > 0 or iou_ < iou_thr or iou_ < max_iou:
                        continue
                    max_iou = iou_
                    m = j
                if m == -1:
                    continue
                dtm[i] = m
                gtm[m] = i
                biou[m] = max_iou

            new_instances = gt_instances
            new_masks_inds = gtm[gtm > -1]
            gt_masks = torch.zeros((N, pred_masks.shape[-2], pred_masks.shape[-1])).to(pred_masks)
            gt_masks[gtm > -1] = pred_masks[new_masks_inds]
            new_instances.gt_bitmasks = gt_masks
            new_instances.gt_masks_flags = (gtm > -1).float()
            new_instances.gt_masks_flags[gtm > -1] = pred_instances.mask_score[new_masks_inds]
            batched_inputs[idx]["instances"] = new_instances

    def freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, batched_inputs):
        if self.training:
            if self.eval_teacher:
                self.teacher.eval()
            else:
                self.teacher.train()

            with torch.no_grad():
                instances = self.teacher.forward_teacher(batched_inputs)

            self.pseudo_labeling(
                batched_inputs,
                instances,
                iou_thr=self.filter_iou_thr,
                score_thr=self.filter_score_thr)
            del instances
            return self.student(batched_inputs)
        else:
            if self.use_teacher_inference:
                return self.teacher(batched_inputs)
            return self.student(batched_inputs)
