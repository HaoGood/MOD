"""
This file contains specific functions for computing losses of FCOS
file
"""

import logging
import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.layers import sigmoid_focal_loss_bce
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 100000000

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.COUNT = [0, 0, 0, 0, 0]

    def gmm_clustter_2(self, cls_loss):
        from sklearn.mixture import GaussianMixture
        import numpy as np
        # mean = torch.mean(cls_loss)
        # sigma = torch.std(cls_loss)
        min_loss = torch.min(cls_loss).cpu().detach().numpy()
        max_loss = torch.max(cls_loss).cpu().detach().numpy()
        means_init = np.array([min_loss, max_loss]).reshape(2, 1)
        precisions_init = np.array([0.1, 0.1]).reshape(2, 1, 1)
        cls_loss = cls_loss.view(-1, 1).cpu().detach().numpy()
        gm = GaussianMixture(n_components=2, weights_init=[0.5, 0.5], 
            means_init=means_init, precisions_init= precisions_init)
        gm.fit(cls_loss)
        results = gm.predict(cls_loss)
        assignments = results == 0

        if len(np.nonzero(assignments)[0]) > 0:
            scores = gm.score_samples(cls_loss)
            score_fgs = scores[assignments]
            fgs_inds = np.nonzero(assignments)[0]
            fgs_thr_ind = np.argmax(score_fgs)
            assignments_ = cls_loss.reshape(-1) <= cls_loss[fgs_inds[fgs_thr_ind]]
            assignments = assignments & assignments_

        return torch.from_numpy(assignments)

    def gmm_clustter(self, cls_loss):
        from sklearn.mixture import GaussianMixture
        import numpy as np
        topk = 12
        topk = min(topk, torch.numel(cls_loss))
        cls_loss = cls_loss.cpu().detach().numpy().flatten()
        lenth = cls_loss.shape[0]
        assign_topk = np.argpartition(cls_loss, topk - 1)[0:topk]
        cls_loss = cls_loss[assign_topk]

        min_loss = np.min(cls_loss)
        max_loss = np.max(cls_loss)

        means_init = np.array([min_loss, max_loss]).reshape(2, 1)
        precisions_init = np.array([0.1, 0.1]).reshape(2, 1, 1)
        cls_loss = cls_loss.reshape((-1, 1))

        gm = GaussianMixture(n_components=2, weights_init=[0.5, 0.5],
            means_init=means_init, precisions_init= precisions_init)
        gm.fit(cls_loss)
        results = gm.predict(cls_loss)
        assign_temp = results == 0
        assignments = np.zeros(lenth, dtype=np.bool)
        assignments[assign_topk[assign_temp]] = True

        # if len(np.nonzero(assignments)[0]) > 0:
        #     scores = gm.score_samples(cls_loss)
        #     score_fgs = scores[assignments]
        #     fgs_inds = np.nonzero(assignments)[0]
        #     fgs_thr_ind = np.argmax(score_fgs)
        #     assignments_ = cls_loss.reshape(-1) < cls_loss[fgs_inds[fgs_thr_ind]]
        #     assignments = assignments & assignments_

        return torch.from_numpy(assignments)

    def topk_clustter(self, cls_loss, k = 9):
        import numpy as np
        # mean = torch.mean(cls_loss)
        # sigma = torch.std(cls_loss)
        min_loss = torch.min(cls_loss).cpu().detach().numpy()
        max_loss = torch.max(cls_loss).cpu().detach().numpy()
        means_init = np.array([min_loss, max_loss]).reshape(2, 1)
        precisions_init = np.array([0.1, 0.1]).reshape(2, 1, 1)
        cls_loss = cls_loss.flatten()
        k = min(k, len(cls_loss))
        cls_loss = 0 - cls_loss
        _, assignments = torch.topk(cls_loss, k)

        return assignments

    def avg_clustter(self, cls_loss):
        mean = torch.mean(cls_loss)
        sigma = torch.std(cls_loss)
        assignments = cls_loss <= mean

        return assignments

    def dbscan_clustter(self, loss):
        from sklearn.clustter import DBSCAN
        import numpy as np

    def get_ious(self, pred, target):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        
        return ious, gious

    def gmm_clustter_2(self, cls_loss):
        from sklearn.mixture import GaussianMixture
        import numpy as np
        # mean = torch.mean(cls_loss)
        # sigma = torch.std(cls_loss)
        min_loss = torch.min(cls_loss).cpu().detach().numpy()
        max_loss = torch.max(cls_loss).cpu().detach().numpy()
        means_init = np.array([min_loss, max_loss]).reshape(2, 1)
        precisions_init = np.array([0.1, 0.1]).reshape(2, 1, 1)
        cls_loss = cls_loss.view(-1, 1).cpu().detach().numpy()
        gm = GaussianMixture(n_components=2, weights_init=[0.5, 0.5], 
            means_init=means_init, precisions_init= precisions_init)
        gm.fit(cls_loss)
        results = gm.predict(cls_loss)
        assignments = results == 0

        if len(np.nonzero(assignments)[0]) > 0:
            scores = gm.score_samples(cls_loss)
            score_fgs = scores[assignments]
            fgs_inds = np.nonzero(assignments)[0]
            fgs_thr_ind = np.argmax(score_fgs)
            assignments_ = cls_loss.reshape(-1) <= cls_loss[fgs_inds[fgs_thr_ind]]
            assignments = assignments & assignments_

        return torch.from_numpy(assignments)

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets_stats(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, labels_reg, reg_targets, mask_for_gt = self.compute_targets_for_locations_stats(
            points_all_level, targets, expanded_object_sizes_of_interest, points
        )
        #-------------------------
        # num_gt = [mask.shape[1] for mask in mask_for_gt]
        #-------------------------
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            labels_reg[i] = torch.split(labels_reg[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            mask_for_gt[i] = torch.split(mask_for_gt[i], num_points_per_level, dim=0)

        labels_level_first = []
        labels_reg_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            labels_reg_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels_reg], dim=0)
            )
            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)
            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)
        #-------------------------
        mask_gt_level_first = []
        levels = len(points)
        self.batch_size = len(labels)
        mask_all_level = []
        for l in range(levels):
            mask_per_level = []
            for i in range(self.batch_size):
                mask_per_im = mask_for_gt[i][l]
                pad_per_im = [torch.zeros_like(mask_per_im) for _ in range(self.batch_size)]
                pad_per_im[i] = mask_per_im
                mask_per_level.append(torch.cat(pad_per_im))
            mask_per_level = torch.cat(mask_per_level, dim=1)
            mask_all_level.append(mask_per_level)
        mask_gt_level_first = torch.cat(mask_all_level).T
        #-------------------------

        return labels_level_first, labels_reg_level_first, reg_targets_level_first, mask_gt_level_first

    def compute_targets_for_locations_stats(self, locations, targets, object_sizes_of_interest, points):
        labels = []
        labels_ = []
        reg_targets = []
        mask_for_gt = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            labels_per_im_reg = torch.zeros_like(labels_per_im)
            labels_per_im_reg[:] = 1
            area = targets_per_im.area()

            sort_inds = torch.sort(area)[1]
            labels_per_im = labels_per_im[sort_inds]
            bboxes = bboxes[sort_inds]
            area = area[sort_inds]
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sampling_radius > 0:
                # is_in_boxes_reg = self.get_sample_region(
                #     bboxes,
                #     self.fpn_strides,
                #     self.num_points_per_level,
                #     xs, ys,
                #     radius=self.center_sampling_radius
                # )
                is_in_boxes_cls = reg_targets_per_im.min(dim=2)[0] > 0
                is_in_boxes_reg = is_in_boxes_cls

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            mask_for_gt.append(is_in_boxes_cls)
            if locations_to_gt_area.shape[1] == 1:
                locations_to_gt_area = locations_to_gt_area.expand(-1, 2)
                is_in_boxes_cls = is_in_boxes_cls.expand(-1, 2)
                labels_per_im = torch.cat([labels_per_im, labels_per_im])
            locations_to_gt_area[is_in_boxes_cls == 0] = INF
            sort_areas, sort_inds = torch.sort(locations_to_gt_area, dim=1)
            locations_to_gt_inds = sort_inds[:, :2]
            locations_to_min_area = sort_areas[:, :2]
            locations_to_gt_area_reg = area[None].repeat(len(locations), 1)
            if locations_to_gt_area_reg.shape[1] == 1:
                locations_to_gt_area_reg = locations_to_gt_area_reg.expand(-1, 2)
                is_in_boxes_reg = is_in_boxes_reg.expand(-1, 2)
                labels_per_im_reg = torch.cat([labels_per_im_reg, labels_per_im_reg])
                reg_targets_per_im = reg_targets_per_im.expand(-1, 2, -1)
            locations_to_gt_area_reg[is_in_boxes_reg == 0] = INF
            sort_areas, sort_inds = torch.sort(locations_to_gt_area_reg, dim=1)
            locations_to_gt_inds_reg = sort_inds[:, :2]
            locations_to_min_area_reg = sort_areas[:, :2]
            reg_targets_per_im = torch.stack([reg_targets_per_im[range(len(locations)), locations_to_gt_inds_reg[:, 0]],
                        reg_targets_per_im[range(len(locations)), locations_to_gt_inds_reg[:, 1]]], dim=-1)

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0
            labels_per_im_reg = labels_per_im_reg[locations_to_gt_inds_reg]
            labels_per_im_reg[locations_to_min_area_reg == INF] = 0

            labels.append(labels_per_im)
            labels_.append(labels_per_im_reg)
            reg_targets.append(reg_targets_per_im)

        return labels, labels_, reg_targets, mask_for_gt

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def _call_for_stats(self, locations, box_cls, box_regression, centerness, labels, reg_targets, gt_mask):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        box_cls_flatten = []
        labels_flatten = []
        box_regression_flatten = []
        reg_targets_flatten = []
        centerness_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            labels_flatten.append(labels[l].reshape(-1, 2))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4, 2))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten[:, 0] > 0).squeeze(1)
        pos_inds_ = torch.nonzero(labels_flatten[:, 1] > 0).squeeze(1)
        neg_inds_ = torch.nonzero(labels_flatten[:, 1] == 0).squeeze(1)
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten[:,:,1] = torch.where((reg_targets_flatten[:,:,1].min(dim=1)[0] < 0)[:, None].expand(-1, 4), 
                    reg_targets_flatten[:,:,0], reg_targets_flatten[:,:,1])
        reg_targets_flatten_stat = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_targets = torch.zeros_like(box_cls_flatten)
        cls_targets_ = torch.zeros_like(box_cls_flatten)
        cls_targets[pos_inds, labels_flatten[pos_inds, 0].long() - 1] = 1
        cls_targets_[pos_inds, labels_flatten[pos_inds, 0].long() - 1] = 1
        cls_targets_[pos_inds_, labels_flatten[pos_inds_, 1].long() - 1] = 1
        cls_loss_pos = sigmoid_focal_loss_bce(box_cls_flatten, cls_targets, funcs = 'stats_focal') / num_pos_avg_per_gpu
        cls_loss_pos_ = sigmoid_focal_loss_bce(box_cls_flatten, cls_targets_, funcs = 'stats_focal') / num_pos_avg_per_gpu
        cls_loss = torch.stack([cls_loss_pos, cls_loss_pos_], dim=1)
        if pos_inds.numel() > 0:
            ious, gious = self.get_ious(box_regression_flatten, reg_targets_flatten_stat[:, :, 0])
            ious_, gious_ = self.get_ious(box_regression_flatten, reg_targets_flatten_stat[:, :, 1])
            reg_loss = self.box_reg_loss_func(
                ious,
                gious,
                funcs = 'stats'
            ) / num_pos_avg_per_gpu
            reg_loss_ = self.box_reg_loss_func(
                ious_,
                gious_,
                funcs = 'stats'
            ) / num_pos_avg_per_gpu
            reg_loss_padding = labels_flatten.new_zeros(labels_flatten.shape, dtype=torch.float)
            reg_loss_padding[pos_inds] = torch.stack([reg_loss, reg_loss_], dim=1)
        #-------------------------------------

        labels_mark = torch.ones_like(labels_flatten, dtype=torch.bool)
        labels_mark[:, 1] = 0
        labels_resample_cls = torch.zeros_like(labels_flatten[:, 0])
        scores_resample_cls = torch.zeros_like(labels_flatten[:, 0], dtype=torch.float32).cuda()
        labels_resample_reg = torch.zeros_like(labels_flatten[:, 0])
        targets_resample_reg = torch.zeros_like(reg_targets_flatten[:, :, 0])
        for gt in gt_mask:
            gt_loss = []
            gt_cls_loss = []
            gt_reg_loss = []
            gt_loss_inds = []
            gt_labels = []
            reg_targets = []
            beg = 0
            for num_points, locs, stride in zip(self.num_points_per_level, locations, self.fpn_strides):
                num_points_ = num_points * self.batch_size
                end = beg + num_points_
                gt_mask_per_level = gt[beg:end]
                cls_loss_per_level = cls_loss[beg:end] * int(num_pos_avg_per_gpu)
                reg_loss_per_level = reg_loss_padding[beg:end] * int(num_pos_avg_per_gpu)
                mark_per_level = labels_mark[beg:end]
                labels_per_level = labels_flatten[beg:end]
                reg_targets_per_level = reg_targets_flatten[beg:end]
                
                gt_mask_per_level = gt_mask_per_level[:, None] & mark_per_level
                gt_per_level_loss = (cls_loss_per_level+reg_loss_per_level)[gt_mask_per_level]
                gt_per_level_cls_loss = (cls_loss_per_level)[gt_mask_per_level]
                gt_per_level_reg_loss = (reg_loss_per_level)[gt_mask_per_level]
                gt_label_per_level = labels_per_level[gt_mask_per_level]
                reg_targets_per_level = reg_targets_per_level.permute(0, 2, 1)[gt_mask_per_level]
                if torch.sum(gt_per_level_loss) == 0:
                    beg = end
                    continue
                else:
                    topk_number = min(9, gt_per_level_loss.numel())
                    aa, bb = (0 - gt_per_level_loss).topk(topk_number)
                    cc = torch.nonzero(gt_mask_per_level[:, 0] | gt_mask_per_level[:,1])
                    gt_loss.append(0 - aa)
                    gt_cls_loss.append(gt_per_level_cls_loss[bb])
                    gt_reg_loss.append(gt_per_level_reg_loss[bb])
                    gt_loss_inds.append(beg + cc[bb])
                    gt_labels.append(gt_label_per_level[bb])
                    reg_targets.append(reg_targets_per_level[bb])
                    beg = end
            if len(gt_loss) == 0:
                labels_mark[gt == 1] = 0
                continue
            if len(gt_loss) == 1:
                gt_loss = gt_loss[0]
                gt_loss_inds = gt_loss_inds[0]
                gt_labels = gt_labels[0]
                gt_cls_loss = gt_cls_loss[0]
                gt_reg_loss = gt_reg_loss[0]
                reg_targets = reg_targets[0]
            else:
                gt_loss_avg = [a.sum()/len(a) for a in gt_loss]
                gt_loss_avg = torch.stack(gt_loss_avg)
                _, level_inds = gt_loss_avg.sort()
                gt_cls_loss = torch.cat([gt_cls_loss[level_inds[0]], gt_cls_loss[level_inds[1]]])
                gt_reg_loss = torch.cat([gt_reg_loss[level_inds[0]], gt_reg_loss[level_inds[1]]])
                gt_cls_weight = torch.softmax(gt_cls_loss, dim=0)
                gt_reg_weight = torch.softmax(gt_reg_loss, dim=0)
                gt_gap = (gt_cls_loss - gt_reg_loss).sigmoid()
                gt_gap[gt_gap < 0.5] = 1 - gt_gap[gt_gap < 0.5]
                gt_loss = torch.sqrt(gt_gap*(gt_cls_loss + gt_reg_loss))
                gt_loss_inds = torch.cat([gt_loss_inds[level_inds[0]], gt_loss_inds[level_inds[1]]])
                gt_labels = torch.cat([gt_labels[level_inds[0]], gt_labels[level_inds[1]]])
                reg_targets = torch.cat([reg_targets[level_inds[0]], reg_targets[level_inds[1]]])
            if torch.numel(gt_loss) == 1:
                assign = 0
            else: 
                assign = self.gmm_clustter_2(gt_loss*10)
            inds = gt_loss_inds[assign].squeeze()
            try: 
                if torch.max(labels_resample_cls[inds]) != 0:
                    logger = logging.getLogger("fcos_core.trainer")
                    logger.info(str(gt_loss))
                    logger.info(str(assign))
                    logger.info(str(inds))
                    aa = labels_mark[:, 0]
                    labels_mark[gt == 1, 0] = 0
                    labels_mark[gt == 1, 1] = 1
                    labels_mark[neg_inds_, 1] = 0
                    labels_mark[(gt == 1) & (aa == 0), 1] = 0
                    continue
            except:
                logger = logging.getLogger("fcos_core.trainer")
                logger.info("Err")
                logger.info(str(gt_loss))
                logger.info(str(assign))
                logger.info(str(inds))
                aa = labels_mark[:, 0]
                labels_mark[gt == 1, 0] = 0
                labels_mark[gt == 1, 1] = 1
                labels_mark[neg_inds_, 1] = 0
                labels_mark[(gt == 1) & (aa == 0), 1] = 0
                continue
            assert torch.max(labels_resample_cls[inds]) == 0
            assert gt_labels[assign].max() - gt_labels[assign].min() == 0
            labels_resample_cls[inds] = gt_labels[assign]
            scores_resample_cls[inds] = 1
            targets_resample_reg[inds] = reg_targets[assign]
            labels_resample_reg[inds] = 1
            aa = torch.zeros_like(labels_mark[:, 0])
            aa[:] = labels_mark[:, 0] 
            labels_mark[gt == 1, 0] = 0
            labels_mark[gt == 1, 1] = 1
            labels_mark[inds, 1] = 0    
            labels_mark[neg_inds_, 1] = 0
            labels_mark[(gt == 1) & (aa == 0), 1] = 0
            assert torch.nonzero((labels_mark[:, 0] == True) & (labels_mark[:, 1] == True)).shape[0] == 0

        return labels_resample_cls, scores_resample_cls, labels_resample_reg, targets_resample_reg

    def _call_for_train(self, locations, box_cls, box_regression, 
        centerness, labels_cls, labels_reg, reg_targets, scores_cls):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = labels_cls
        labels_reg_flatten = []
        scores_cls_flatten = scores_cls
        reg_targets_flatten = reg_targets
        for l in range(len(box_cls)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_reg_flatten.append(labels_reg[l].reshape(-1))
            centerness_flatten.append(centerness[l].reshape(-1))
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_reg_flatten = torch.cat(labels_reg_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # pos_inds_reg = torch.nonzero(labels_reg_flatten > 0).squeeze(1)
        pos_inds_reg = pos_inds
        # ignore_inds = torch.nonzero(labels_flatten == -1).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds_reg]
        reg_targets_flatten = reg_targets_flatten[pos_inds_reg]
        centerness_flatten = centerness_flatten[pos_inds_reg]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        # total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        # num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        num_soft_pos_avg_per_gpu = \
            reduce_sum(scores_cls_flatten.sum()).item() / float(num_gpus)
        total_num_pos_reg = reduce_sum(pos_inds_reg.new_tensor([pos_inds_reg.numel()])).item()
        num_reg_avg_per_gpu = max(total_num_pos_reg / float(num_gpus), 1.0)

        targets = torch.zeros_like(box_cls_flatten)
        targets[pos_inds, labels_flatten[pos_inds].long() - 1] = scores_cls_flatten[pos_inds]
        cls_loss = sigmoid_focal_loss_bce(box_cls_flatten, targets, funcs = 'train') / num_soft_pos_avg_per_gpu

        if pos_inds_reg.numel() > 0:
            # centerness_targets = torch.zeros_like(centerness_flatten)
            # centerness_targets[:] = 1
            ious, gious = self.get_ious(box_regression_flatten, reg_targets_flatten)
            centerness = self.compute_centerness_targets(reg_targets_flatten)
            centerness_targets = gious
            if torch.nonzero(centerness_targets > 1).shape[0] > 0:
                logger = logging.getLogger("fcos_core.trainer")
                logger.info(str(centerness_targets))
                centerness_targets[centerness_targets > 1] = 1
            sum_centerness_avg_per_gpu = \
                reduce_sum(centerness.sum()).item() / float(num_gpus)
            reg_loss = self.box_reg_loss_func(
                ious,
                gious,
                weight=centerness
            ) / sum_centerness_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_reg_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()
        logger = logging.getLogger("fcos_core.trainer")

        return cls_loss, reg_loss, centerness_loss

    def _call_for_visual(self, locations, box_cls, box_regression, 
        centerness, labels_cls, labels_cls_gt, labels_reg, reg_targets):
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = labels_cls
        labels_flatten_ = []
        reg_targets_flatten = []
        for l in range(len(box_cls)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
            labels_flatten_.append(labels_cls_gt[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten_ = torch.cat(labels_flatten_, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten_ > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten_ = centerness_flatten.sigmoid()
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        split_interval = [feature.size(2)*feature.size(3) for feature in box_cls]
        cls_targets = torch.zeros_like(box_cls_flatten)
        cls_targets[pos_inds, labels_flatten_[pos_inds].long() - 1] = 1
        cls_loss = sigmoid_focal_loss_bce(box_cls_flatten, cls_targets, funcs = 'stats_bce') / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            ious, gious = self.get_ious(box_regression_flatten, reg_targets_flatten)
            centerness = self.compute_centerness_targets(reg_targets_flatten)
            centerness_targets = gious
            sum_centerness_avg_per_gpu = \
                reduce_sum(centerness.sum()).item() / float(num_gpus)
            reg_loss = self.box_reg_loss_func(
                ious,
                gious,
                weight=None,
                funcs="stats"
            ) / num_pos_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        split_interval = [feature.size(2)*feature.size(3) for feature in box_cls]
        cls_loss_split = cls_loss.split(split_interval)
        reg_loss_padding = cls_loss.new_zeros(cls_loss.shape)
        reg_loss_padding[pos_inds] = reg_loss
        reg_loss_split = reg_loss_padding.split(split_interval)
        loss_gap = torch.abs(cls_loss - reg_loss_padding) * int(num_pos_avg_per_gpu)
        loss_gap_split = loss_gap.split(split_interval)
        center_loss_padding = cls_loss.new_zeros(cls_loss.shape)
        center_loss_padding[pos_inds] = centerness_loss
        # center_loss_split = center_loss_padding.split(split_interval)
        center_loss_split = centerness_flatten_.split(split_interval)
        labels_cls_split = labels_cls.split(split_interval)
        shape_per_level = [(cls_feature.shape[2], cls_feature.shape[3]) for cls_feature in box_cls]

        location_split = [loc.cpu().numpy().astype(int) for loc in locations]
        loss_with_location = []
        for cls_loss, reg_loss, loss_gap, center_loss, loc, label, shape_ in zip(cls_loss_split, reg_loss_split, 
            loss_gap_split, center_loss_split, location_split, labels_cls_split, shape_per_level):
            loss_with_location.append({'cls_loss': cls_loss.reshape(shape_), 
                'reg_loss': reg_loss.reshape(shape_),
                'loss_gap': loss_gap.reshape(shape_),
                'center_loss': center_loss.reshape(shape_),
                'label': label,
                'loc': loc})

        return loss_with_location
    
    def __call__(self, locations, box_cls, box_regression, centerness, targets, funcs='train'):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        labels_cls_gt, labels_reg, reg_targets, gt_mask = self.prepare_targets_stats(locations, targets)
        labels_cls, scores_cls, labels_reg, reg_targets = self._call_for_stats(locations, box_cls, box_regression, 
            centerness, labels_cls_gt, reg_targets, gt_mask)
        if funcs == 'train':
            return self._call_for_train(locations, box_cls, box_regression, 
                centerness, labels_cls, labels_reg, reg_targets, scores_cls)
        if funcs == 'visual':
            return self._call_for_visual(locations, box_cls, box_regression, 
                centerness, labels_cls, labels_cls_gt, labels_reg, reg_targets)


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
