import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            # if self.use_dcn_in_tower and \
            #         i == cfg.MODEL.FCOS.NUM_CONVS - 1:
            #     conv_func = DFConv2d
            # else:
            #     conv_func = nn.Conv2d
            if i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        # torch.nn.init.constant_(self.centerness.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        head = FCOSHead(cfg, in_channels)

        box_selector_test = make_fcos_postprocessor(cfg)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets #,images
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes #, targets
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        # _ = self.loss_evaluator(
        #     locations, box_cls, box_regression, centerness, targets
        # )
        # losses = {
        #     "loss_cls": torch.nn.Parameter(torch.Tensor([0]).cuda()),
        #     "loss_reg": torch.nn.Parameter(torch.Tensor([0]).cuda()),
        #     "loss_centerness": torch.nn.Parameter(torch.Tensor([0]).cuda())
        # }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def lossmap_2_heatmap(self, feature_map):
        import numpy as np
        heatmap = feature_map
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    def heatmap_pool(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = F.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def get_sample_box(self, gt, strides=[8, 16, 32, 64, 128], radius=1.5):
        num_gts = gt.shape[0]
        cent_xs = (gt[:, 0] + gt[:, 2]) / 2
        cent_ys = (gt[:, 1] + gt[:, 3]) / 2
        sample_boxes = []
        for stride_level in strides:
            stride = stride_level * radius
            l = cent_xs - stride
            r = cent_xs + stride
            t = cent_ys - stride
            b = cent_ys + stride
            l_for_sample = torch.where(l > gt[:, 0], l, gt[:, 0])
            r_for_sample = torch.where(r < gt[:, 2], r, gt[:, 2])
            t_for_sample = torch.where(t > gt[:, 1], t, gt[:, 1])
            b_for_sample = torch.where(b < gt[:, 3], b, gt[:, 3])
            boxes = torch.stack([l_for_sample, t_for_sample, r_for_sample,
                b_for_sample], dim=1)
            sample_boxes.append(boxes)
        return torch.stack(sample_boxes, dim=0)

    def _forward_train_visual_loss(self, locations, box_cls, box_regression, centerness, targets, images):
        import cv2
        import numpy as np
        import os
        loss_with_location = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets, funcs='visual'
        )
        target = targets[0]
        bboxes = target.bbox
        sample_boxes = self.get_sample_box(bboxes).cpu().numpy().astype(int)
        bboxes = bboxes.cpu().numpy().astype(int)

        image_height = images.tensors[0].shape[1]
        image_width = images.tensors[0].shape[2]
        images_ndarray = images.tensors[0].permute(1, 2, 0).cpu().numpy() # C*H*W to H*W*C
        src_image = images_ndarray
        for box in bboxes:
            src_image = cv2.rectangle(src_image, (box[0],box[1]), (box[2],box[3]), color=(0,255,0)) 
        loss_map = np.zeros((image_height, image_width))
        cv2.imwrite(os.path.join('/home/buu/results/loss_map/cls', 'source_' + '.jpg'), src_image)

        level = [0, 1, 2, 3, 4]
        for loss_level, sample_level, l in zip(loss_with_location, sample_boxes, level):
            # import pdb; pdb.set_trace()
            loss_map_level = (loss_level['reg_loss']).detach().cpu().numpy()
            label_level = torch.nonzero(loss_level['label']).squeeze().detach().cpu().numpy()
            loc_level = loss_level['loc']
            pos_samples = loc_level[label_level]
            heatmap_level = self.lossmap_2_heatmap(loss_map_level)
            # cv2.resize destination size is (W*H)
            heatmap_level = cv2.resize(heatmap_level, (image_width, image_height))
            loss_map_level = cv2.resize(loss_map_level, (image_width, image_height))
            loss_map += loss_map_level
            heatmap_level = np.uint8(255 * heatmap_level)
            heatmap_level = cv2.applyColorMap(heatmap_level, cv2.COLORMAP_JET)
            superimposed_img = heatmap_level * 0.4 + src_image
            # for box in sample_level:
            #     superimposed_img = cv2.rectangle(superimposed_img, (box[0],box[1]), 
            #         (box[2],box[3]), color=(0,0,255))
            if len(pos_samples.shape) == 1:
                pos_samples = pos_samples[None]
            for pos_sample in pos_samples:
                if l < 2:
                    cv2.drawMarker(superimposed_img, (pos_sample[0], pos_sample[1]), color=(0,255,0), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=4*(l+1), thickness=1)
                else:
                    cv2.drawMarker(superimposed_img, (pos_sample[0], pos_sample[1]), color=(0,255,0), 
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=4*(l+1), thickness=2)
            cv2.imwrite(os.path.join('/home/buu/results/loss_map/cls', 'loss_map_level_' + 
               str(l) + '.jpg'), superimposed_img)

        loss_map = self.lossmap_2_heatmap(loss_map)
        heatmap = np.uint8(255 * loss_map)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + src_image
        cv2.imwrite(os.path.join('/home/buu/results/loss_map/loss_map', 'loss_map' + '.jpg'), superimposed_img)
        return None, losses

    def _forward_train_visual_output(self, locations, box_cls, box_regression, centerness, targets, images):
        import cv2
        import numpy as np
        import os
        target = targets[0]
        bboxes = target.bbox
        labels = target.get_field('labels')
        labels = [int(label) for label in labels]
        names = target.names
        label_to_name = {l : n for l, n in zip(labels, names)}
        print(labels)
        print(names)
        sample_boxes = self.get_sample_box(bboxes).cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        image_height = images.tensors[0].shape[1]
        image_width = images.tensors[0].shape[2]
        images_ndarray = images.tensors[0].permute(1, 2, 0).cpu().numpy() # C*H*W to H*W*C
        src_image = images_ndarray
        for box in bboxes:
            src_image = cv2.rectangle(src_image, (box[0],box[1]), (box[2],box[3]), color=(0,255,0)) 
        cls_map = np.zeros((image_height, image_width))
        cv2.imwrite(os.path.join('/home/buu/results/loss_map', 'source_' + '.jpg'), src_image)

        level = [0, 1, 2, 3, 4]
        for box_cls_level, cen_level, l in zip(box_cls, centerness, level):
            box_cls_level = box_cls_level.squeeze().sigmoid().detach().cpu().numpy()
            cen_level = cen_level.sigmoid()
            cen_level = self.heatmap_pool(cen_level).squeeze().detach().cpu().numpy()
            # cen_level = np.clip(cen_level, 0.05, 1)
            heatmap_cen = self.lossmap_2_heatmap(cen_level)
            heatmap_cen = cv2.resize(cen_level, (image_width, image_height))
            heatmap_cen = np.uint8(255 * heatmap_cen)
            heatmap_cen = cv2.applyColorMap(heatmap_cen, cv2.COLORMAP_JET)
            superimposed_img_cen = heatmap_cen * 0.4 + src_image
            cv2.imwrite(os.path.join('/home/buu/results/cls_map', 'cen_map_level'
            + str(l) + '_' + '.jpg'), superimposed_img_cen) 
            for label, name in zip(labels, names):
                box_cls_level_channel = box_cls_level[label - 1]
                # box_cls_level_channel = np.clip(box_cls_level_channel, 0.1, 1)
                # box_cls_level = box_cls_level * cen_level
                heatmap = self.lossmap_2_heatmap(box_cls_level_channel)
                heatmap = cv2.resize(heatmap, (image_width, image_height))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + src_image
                cv2.imwrite(os.path.join('/home/buu/results/cls_map', 'cls_map_level'
                + str(l) + '_'+ name + '.jpg'), superimposed_img) 

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
