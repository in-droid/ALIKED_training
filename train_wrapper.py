import os
import logging
import functools
from typing import Optional
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
# import scheduler 
from torch.optim.lr_scheduler import StepLR

import pytorch_lightning as pl


# from torch.
# from training.scheduler import WarmupConstantSchedule
from utils import (mutual_argmin, 
                   plot_keypoints, 
                   EmptyTensorError, 
                   warp, 
                   compute_keypoints_distance, 
                   mutual_argmax,
                   keypoints_normal2pixel)

from losses import (ReprojectionLocLoss,
                    SparseNRELoss,
                    ReliableLoss,
                    DispersityPeakLoss
                    )
from scheduler import WarmupCosineSchedule
import sys
sys.path.append('./imc24lightglue')
from imc24lightglue.aliked import ALIKED


class ALIKEDTrainWrapper(ALIKED, pl.LightningModule):
    def __init__(self,
                 weights,
                 # ================================== model config (from ALIKED default_conf)
                 model_name: str = "aliked-n16",
                 max_num_keypoints: int = -1,
                 detection_threshold: float = 0.2,
                 nms_radius: int = 2,
                 # ================================== training parameters
                 radius: int = 2,
                 top_k: int = 500, scores_th: float = 0.5, n_limit: int = 0,  # in training stage
                 scores_th_eval: float = 0.2, n_limit_eval: int = 5000,  # in evaluation stage
                 # ================================== gt reprojection th
                 train_gt_th: int = 5, eval_gt_th: int = 5,
                 # ================================== loss weight
                #  w_pk: float = 1.0,  # weight of peaky loss
                w_rp: float = 1.0,  # weight of reprojection loss
                w_sparse_nre: float = 5.0,
                w_dispersity: float = 0.5,
                w_reliable: float = 1.0,
                #  w_sp: float = 1.0,  # weight of score map repetability loss
                #  w_ds: float = 1.0,  # weight of descritptor loss
                #  w_triplet: float = 0.,
                 sc_th: float = 0.1,  # score threshold in peaky and  reprojection loss
                 norm: int = 1,  # distance norm
                 temp_sp: float = 0.1,  # temperature in ScoreMapRepLoss
                 temp_ds: float = 0.02,  # temperature in DescReprojectionLoss
                 # ================================== learning rate
                 lr: float = 1e-3,
                 log_freq_img: int = 2000,  # log image every log_freq_img steps
                 # ================================== pretrained_model
                 pretrained_model: Optional[str] = None,
                 lr_scheduler=functools.partial(WarmupCosineSchedule, warmup_steps=100, t_total=1000),  # TODO: use scheduler
                #  lr_scheduler=functools.partial(WarmupConstantSchedule, warmup_steps=10000),
                 debug: bool = False,
                 **conf):
        
        # Initialize ALIKED with the appropriate configuration
        aliked_conf = {
            "model_name": model_name,
            "max_num_keypoints": max_num_keypoints,
            "detection_threshold": detection_threshold,
            "nms_radius": nms_radius,
            **conf
        }
        
        super().__init__(weights, **aliked_conf)
        # self.save_hyperparameters()

        self.lr = lr

        # =================== hyper parameters
        # soft detector parameters
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        
        # loss configs
        self.w_sparse_nre = w_sparse_nre
        self.w_rp = w_rp
        self.w_dispersity = w_dispersity
        self.w_reliable = w_reliable
        # self.w_triplet = w_triplet
        # reprojection loss parameters
        self.train_gt_th = train_gt_th
        # evaluation th for MMA on training dataset
        self.eval_gt_th = eval_gt_th

        self.scores_th_eval = scores_th_eval
        self.n_limit_eval = n_limit_eval

        self.log_freq_img = log_freq_img

        self.pretrained_model = pretrained_model
        self.lr_scheduler = lr_scheduler
        self.debug = debug

        # =================== additional model weight loading (if different from ALIKED constructor)
        if pretrained_model is not None and pretrained_model != weights:
            if os.path.exists(pretrained_model):
                logging.info(f"Loading additional weights from {pretrained_model}")
                if pretrained_model.endswith('ckpt'):
                    state_dict = torch.load(pretrained_model, torch.device('cpu'))['state_dict']
                elif pretrained_model.endswith('pt') or pretrained_model.endswith('pth'):
                    state_dict = torch.load(pretrained_model, torch.device('cpu'))
                else:
                    logging.error(f"Error model file: {pretrained_model}")
                self.load_state_dict(state_dict, strict=False)
            else:
                logging.error(f"File does not exist: {pretrained_model}")



        # # =================== losses
        # if self.w_pk > 0:
        #     from nets.losses import PeakyLoss
        #     self.PeakyLoss = PeakyLoss(scores_th=sc_th)
        if self.w_rp > 0:
            self.ReprojectionLocLoss = ReprojectionLocLoss(norm=norm, scores_th=0.5)
        if self.w_sparse_nre> 0:
            self.SparseNRELoss = SparseNRELoss(t_des = 0.1)
        if self.w_dispersity > 0:
            self.DispersityPeakLoss = DispersityPeakLoss(window_size=5, scores_th=0.5)
        if self.w_reliable > 0:
            self.ReliableLoss = ReliableLoss(t_rel=1)

        # if self.w_sp > 0:
        #     from nets.losses import ScoreMapRepLoss
        #     self.ScoreMapRepLoss = ScoreMapRepLoss(temperature=temp_sp)
        # if self.w_ds > 0:
        #     from nets.losses import DescReprojectionLoss
        #     self.DescReprojectionLoss = DescReprojectionLoss(temperature=temp_ds)
        # if self.w_triplet > 0:
        #     from nets.losses import TripletLoss
        #     self.TripletLoss = TripletLoss()

        # Create a soft detector that matches the training interface
        # This adapts the ALIKED's dkd to work like ALIKE's softdetect
        self.softdetect = self._create_soft_detector()

        # ================== to compute MMA on hpatches
        lim = [1, 15]
        self.rng = np.arange(lim[0], lim[1] + 1)
        self.i_err = {thr: 0 for thr in self.rng}
        self.v_err = {thr: 0 for thr in self.rng}
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []
        # self.errors = load_precompute_errors(str(Path(__file__).parent / 'errors.pkl'))

    def _create_soft_detector(self):
        """Create a soft detector adapter to make ALIKED compatible with training code"""
        class SoftDetectorAdapter:
            def __init__(self, dkd_detector):
                self.dkd = dkd_detector
            
            def detect_keypoints(self, score_map):
                """Adapt ALIKED's dkd.detect_keypoints to match ALIKE's interface"""
                B, C, H, W = score_map.shape
                image_size = torch.tensor([W, H], device=score_map.device).unsqueeze(0).expand(B, 2)
                
                keypoints, kptscores, scoredispersitys = self.dkd(score_map, image_size=image_size)
                
                return keypoints, scoredispersitys, kptscores
        
        return SoftDetectorAdapter(self.dkd)

    def forward_train(self, image, data: dict = None) -> dict:
        """Modified forward method for training that returns additional outputs needed for training"""
        B, C, H, W = image.shape
        if image.shape[1] == 1:
            from kornia.color import grayscale_to_rgb
        #   import grayscale_to_rgb  # Assuming this function exists
            image = grayscale_to_rgb(image)

        # Extract dense feature and score maps
        feature_map, score_map = self.extract_dense_map(image)
        
        # For training, we also need the descriptor maps
        # The feature_map from extract_dense_map should be the descriptor map
        descriptor_map = feature_map  # This is the normalized feature map
        
        image_size = (
            torch.tensor([W, H], device=image.device)
            .unsqueeze(0)              # 1×2
            .expand(B, 2)              # B×2
        )
        
        keypoints, kptscores, scoredispersitys = self.dkd(
            score_map, image_size=image_size
        )
        
        # Get descriptors at keypoint locations
        descriptors, offsets = self.desc_head(feature_map, keypoints)

        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "keypoint_scores": kptscores,
            "scores_map": score_map,  # Changed from score_maps to scores_map for consistency
            "descriptor_map": descriptor_map,  # Add descriptor map for training
        }

    def forward(self, image, data: dict = None) -> dict:
        """Use training forward during training, regular forward during eval"""
        if self.training:
            return self.forward_train(image, data)
        else:
            return super().forward(image, data)

    def on_train_start(self):
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.log_hyperparams(self.hparams)

    # def validation_step(self, batch, batch_idx):
    #     """Validation step for PyTorch Lightning"""
    #     b, c, h, w = batch['image0'].shape
    #     self.validation_step_loss(batch, batch_idx)
    #     # Forward pass
    #     pred0 = self.forward_train(batch['image0'])
    #     pred1 = self.forward_train(batch['image1'])

    #     # Compute correspondences
    #     correspondences, pred0_with_rand, pred1_with_rand = self.compute_correspondence(pred0, pred1, batch)
        
    #     # Compute validation loss (same as training loss but without gradients)
    #     val_loss = 0
    #     val_loss_package = {}

    #     if self.w_rp > 0:
    #         loss_reprojection = self.ReprojectionLocLoss(pred0_with_rand, pred1_with_rand, correspondences)
    #         val_loss += self.w_rp * loss_reprojection
    #         val_loss_package['val/loss_reprojection'] = loss_reprojection

    #     if self.w_sparse_nre > 0:
    #         loss_sparse_nre = self.SparseNRELoss(pred0_with_rand, pred1_with_rand, correspondences)
    #         val_loss += self.w_sparse_nre * loss_sparse_nre
    #         val_loss_package['val/loss_sparse_nre'] = loss_sparse_nre

    #     if self.w_dispersity > 0:
    #         loss_dispersity0 = self.DispersityPeakLoss(pred0_with_rand)
    #         loss_dispersity1 = self.DispersityPeakLoss(pred1_with_rand)
    #         loss_dispersity = (loss_dispersity0 + loss_dispersity1) / 2.
    #         val_loss += self.w_dispersity * loss_dispersity
    #         val_loss_package['val/loss_dispersity'] = loss_dispersity

    #     if self.w_reliable > 0:
    #         loss_reliable = self.ReliableLoss(pred0_with_rand, pred1_with_rand)
    #         val_loss += self.w_reliable * loss_reliable
    #         val_loss_package['val/loss_reliable'] = loss_reliable

    #     # Log validation losses
    #     if hasattr(self, 'logger') and self.logger is not None:
    #         self.logger.log_metrics({'val/loss': val_loss}, step=self.global_step)
    #     # Log the validation loss

    #     self.log('val/loss', val_loss, prog_bar=True)
    #     for k, v in val_loss_package.items():
    #         self.log(k, v)

    #     if hasattr(self, 'logger') and self.logger is not None:
    #         self.logger.log_metrics(val_loss_package, step=self.global_step)
    #     # Compute validation accuracy
    #     pred = {'scores_map0': pred0['scores_map'],
    #             'scores_map1': pred1['scores_map'],
    #             'kpts0': [], 'kpts1': [],
    #             'desc0': [], 'desc1': []}
        
    #     for idx in range(b):
    #         num_det0 = pred0_with_rand['num_det'][idx]
    #         num_det1 = pred1_with_rand['num_det'][idx]
    #         pred['kpts0'].append(
    #             (pred0_with_rand['keypoints'][idx][:num_det0] + 1) / 2 * num_det0.new_tensor([[w - 1, h - 1]]))
    #         pred['kpts1'].append(
    #             (pred1_with_rand['keypoints'][idx][:num_det1] + 1) / 2 * num_det1.new_tensor([[w - 1, h - 1]]))
    #         pred['desc0'].append(pred0_with_rand['descriptors'][idx][:num_det0])
    #         pred['desc1'].append(pred1_with_rand['descriptors'][idx][:num_det1])

    #     val_accuracy = self.evaluate(pred, batch)
    #     self.log('val/acc', val_accuracy, prog_bar=True)
    #     if hasattr(self, 'logger') and self.logger is not None:

    #     return val_loss

    def compute_correspondence(self, pred0, pred1, batch, rand=True):
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['scores_map'][0].new_tensor([[w - 1, h - 1]])

        if self.debug:
            from utils import display_image_in_actual_size
            image0 = batch['image0'][0].permute(1, 2, 0).cpu().numpy()
            image1 = batch['image1'][0].permute(1, 2, 0).cpu().numpy()
            display_image_in_actual_size(image0)
            display_image_in_actual_size(image1)


        pred0_with_rand = pred0
        pred1_with_rand = pred1
        pred0_with_rand['scores'] = []
        pred1_with_rand['scores'] = []
        pred0_with_rand['descriptors'] = []
        pred1_with_rand['descriptors'] = []
        pred0_with_rand['num_det'] = []
        pred1_with_rand['num_det'] = []

        kps0, score_dispersity0, scores0 = self.softdetect.detect_keypoints(pred0['scores_map'])
        kps1, score_dispersity1, scores1 = self.softdetect.detect_keypoints(pred1['scores_map'])
        # take top 400 keypoints
        if rand:
            for idx in range(b):
                if len(kps0[idx]) > 400:
                    top_indices = torch.argsort(scores0[idx], descending=True)[:400]
                    kps0[idx] = kps0[idx][top_indices]
                    score_dispersity0[idx] = score_dispersity0[idx][top_indices]
                    scores0[idx] = scores0[idx][top_indices]


                if len(kps1[idx]) > 400:
                    top_indices = torch.argsort(scores1[idx], descending=True)[:400]
                    kps1[idx] = kps1[idx][top_indices]
                    score_dispersity1[idx] = score_dispersity1[idx][top_indices]
                    scores1[idx] = scores1[idx][top_indices]    

        pred0_with_rand['keypoints'] = kps0
        # print("kps0 len", len(kps0))
        # print("kps1 len", len(kps1[0]))
        pred0_with_rand['score_dispersity'] = score_dispersity0
        pred0_with_rand['keypoint_scores'] = scores0
        pred1_with_rand['keypoints'] = kps1
        pred1_with_rand['score_dispersity'] = score_dispersity1
        pred1_with_rand['keypoint_scores'] = scores1

    
        # pred0_with_rand['keypoints'] = kps
        # pred0_with_rand['score_dispersity'] = score_dispersity

        # kps, score_dispersity, scores = self.softdetect.detect_keypoints(pred1['scores_map'])
        # pred1_with_rand['keypoints'] = kps
        # pred1_with_rand['score_dispersity'] = score_dispersity

        correspondences = []
        for idx in range(b):
            # =========================== prepare keypoints
            kpts0, kpts1 = pred0['keypoints'][idx], pred1['keypoints'][idx]  # (x,y), shape: Nx2
            # print("kpts0 shape", kpts0.shape)
            # print("kpts1 shape", kpts1.shape)
            scores0, scores1 = pred0['keypoint_scores'][idx], pred1['keypoint_scores'][idx]  # shape: N

            # additional random keypoints
            if rand:
                rand0 = torch.rand(len(kpts0), 2, device=kpts0.device) * 2 - 1  # -1~1
                # print("rand0 shape", rand0.shape)
                rand1 = torch.rand(len(kpts1), 2, device=kpts1.device) * 2 - 1  # -1~1
                # print("rand1 shape", rand1.shape)
                kpts0 = torch.cat([kpts0, rand0])
                kpts1 = torch.cat([kpts1, rand1])

                # print("kpts0 after rand shape", kpts0.shape)
                # print("kpts1 after rand shape", kpts1.shape)

                pred0_with_rand['keypoints'][idx] = kpts0
                pred1_with_rand['keypoints'][idx] = kpts1

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]
            scores_kpts0 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()
            scores_kpts1 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                           mode='bilinear', align_corners=True).squeeze()

            kpts0_wh_ = (kpts0 / 2 + 0.5) * wh  # N0x2, (w,h)
            kpts1_wh_ = (kpts1 / 2 + 0.5) * wh  # N1x2, (w,h)

            # ========================= nms
            dist = compute_keypoints_distance(kpts0_wh_.detach(), kpts0_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts0[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts0_wh = kpts0_wh_[valid_mask]
            kpts0 = kpts0[valid_mask]
            scores_kpts0 = scores_kpts0[valid_mask]
            pred0_with_rand['keypoints'][idx] = kpts0

            valid_mask = valid_mask[:len(pred0_with_rand['score_dispersity'][idx])]
            pred0_with_rand['score_dispersity'][idx] = pred0_with_rand['score_dispersity'][idx][valid_mask]
            pred0_with_rand['num_det'].append(valid_mask.sum())

            dist = compute_keypoints_distance(kpts1_wh_.detach(), kpts1_wh_.detach())
            local_mask = dist < self.radius
            valid_cnt = torch.sum(local_mask, dim=1)
            indices_need_nms = torch.where(valid_cnt > 1)[0]
            for i in indices_need_nms:
                if valid_cnt[i] > 0:
                    kpt_indices = torch.where(local_mask[i])[0]
                    scs_max_idx = scores_kpts1[kpt_indices].argmax()

                    tmp_mask = kpt_indices.new_ones(len(kpt_indices)).bool()
                    tmp_mask[scs_max_idx] = False
                    suppressed_indices = kpt_indices[tmp_mask]

                    valid_cnt[suppressed_indices] = 0
            valid_mask = valid_cnt > 0
            kpts1_wh = kpts1_wh_[valid_mask]
            kpts1 = kpts1[valid_mask]
            scores_kpts1 = scores_kpts1[valid_mask]
            pred1_with_rand['keypoints'][idx] = kpts1

            valid_mask = valid_mask[:len(pred1_with_rand['score_dispersity'][idx])]
            pred1_with_rand['score_dispersity'][idx] = pred1_with_rand['score_dispersity'][idx][valid_mask]
            pred1_with_rand['num_det'].append(valid_mask.sum())

            pred0_with_rand['scores'].append(scores_kpts0)
            pred1_with_rand['scores'].append(scores_kpts1)
            descriptor_map0, descriptor_map1 = pred0['descriptor_map'][idx], pred1['descriptor_map'][idx]
            desc0 = torch.nn.functional.grid_sample(descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True)[0, :, 0, :].t()
            desc1 = torch.nn.functional.grid_sample(descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2),
                                                    mode='bilinear', align_corners=True)[0, :, 0, :].t()
            desc0 = torch.nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)

            pred0_with_rand['descriptors'].append(desc0)
            pred1_with_rand['descriptors'].append(desc1)

            # =========================== prepare warp parameters
            warp01_params = {}
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[idx]
            warp10_params = {}
            for k, v in batch['warp10_params'].items():
                warp10_params[k] = v[idx]

            # =========================== warp keypoints across images
            try:
                # valid keypoint, valid warped keypoint, valid indices
                kpts0_wh, kpts01_wh, ids0, ids0_out = warp(kpts0_wh, warp01_params)
                kpts1_wh, kpts10_wh, ids1, ids1_out = warp(kpts1_wh, warp10_params)
                if len(kpts0_wh) == 0 or len(kpts1_wh) == 0 or len(kpts0) == 0 or len(kpts1) == 0:
                    raise EmptyTensorError
            except EmptyTensorError:
                correspondences.append({'correspondence0': None, 'correspondence1': None,
                                        'dist': kpts0_wh.new_tensor(0),
                                        })
                continue

            if self.debug:
                from utils import display_image_in_actual_size
                image0 = batch['image0'][0].permute(1, 2, 0).cpu().numpy()
                image1 = batch['image1'][0].permute(1, 2, 0).cpu().numpy()

                p0 = kpts0_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts0 = plot_keypoints(image0, p0, radius=1, color=(255, 0, 0))

                p1 = kpts1_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts1 = plot_keypoints(image1, p1, radius=1, color=(255, 0, 0))

                p01 = kpts01_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts01 = plot_keypoints(img_kpts1, p01, radius=1, color=(0, 255, 0))
                display_image_in_actual_size(img_kpts01)

                p10 = kpts10_wh[:, [1, 0]].cpu().detach().numpy()
                img_kpts10 = plot_keypoints(img_kpts0, p10, radius=1, color=(0, 255, 0))
                display_image_in_actual_size(img_kpts10)

            # ============================= compute reprojection error
            dist01 = compute_keypoints_distance(kpts0_wh, kpts10_wh)
            dist10 = compute_keypoints_distance(kpts1_wh, kpts01_wh)

            dist_l2 = (dist01 + dist10.t()) / 2.
            # find mutual correspondences by calculating the distance
            # between keypoints (I1) and warpped keypoints (I2->I1)
            mutual_min_indices = mutual_argmin(dist_l2)

            dist_mutual_min = dist_l2[mutual_min_indices]
            valid_dist_mutual_min = dist_mutual_min.detach() < self.train_gt_th

            ids0_d = mutual_min_indices[0][valid_dist_mutual_min]
            ids1_d = mutual_min_indices[1][valid_dist_mutual_min]

            correspondence0 = ids0[ids0_d]
            correspondence1 = ids1[ids1_d]

            # L1 distance
            dist01_l1 = compute_keypoints_distance(kpts0_wh, kpts10_wh, p=1)
            dist10_l1 = compute_keypoints_distance(kpts1_wh, kpts01_wh, p=1)

            dist_l1 = (dist01_l1 + dist10_l1.t()) / 2.

            # =========================== compute cross image descriptor similarity_map
            similarity_map_01 = torch.einsum('nd,dhw->nhw', desc0, descriptor_map1)
            similarity_map_10 = torch.einsum('nd,dhw->nhw', desc1, descriptor_map0)

            similarity_map_01 = similarity_map_01[ids0]  # valid descriptors
            similarity_map_10 = similarity_map_10[ids1]

            kpts01 = 2 * kpts01_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]
            kpts10 = 2 * kpts10_wh.detach() / wh - 1  # N0x2, (x,y), [-1,1]

            matches = torch.stack([correspondence0, 
                                   correspondence1], dim=1)  # Nx2, (kpts0, kpts1)

            correspondences.append({'correspondence0': correspondence0,  # indices of matched kpts0 in all kpts
                                    'correspondence1': correspondence1,  # indices of matched kpts1 in all kpts
                                    'scores0': scores_kpts0[ids0],
                                    'scores1': scores_kpts1[ids1],
                                    'kpts01': kpts01, 'kpts10': kpts10,  # warped valid kpts
                                    'ids0': ids0, 'ids1': ids1,  # valid indices of kpts0 and kpts1
                                    'ids0_out': ids0_out, 'ids1_out': ids1_out,
                                    'ids0_d': ids0_d, 'ids1_d': ids1_d,  # match indices of valid kpts0 and kpts1
                                    'dist_l1': dist_l1,  # cross distance matrix of valid kpts using L1 norm
                                    'dist': dist_l2,  # cross distance matrix of valid kpts using L2 norm
                                    # 'similarity_map_01': similarity_map_01,  # all
                                    # 'similarity_map_10': similarity_map_10,  # all
                                    'similarity_map_01_valid': similarity_map_01,  # valid
                                    'similarity_map_10_valid': similarity_map_10,  # valid
                                    'matches': matches,  # Nx2, (kpts0, kpts1)
                                    })

        return correspondences, pred0_with_rand, pred1_with_rand
    


    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler for PyTorch Lightning"""
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # If lr_scheduler is provided, create the scheduler
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',  # or 'epoch'
                    'frequency': 1,
                }
            }
        else:
            return optimizer
        
   
    def val_match(self, batch):
        b, _, h0, w0 = batch['image0'].shape
        _, _, h1, w1 = batch['image1'].shape
        assert b == 1

        # ==================================== extract keypoints and descriptors
        top_k_old = self.top_k
        scores_th_old = self.scores_th
        n_limit_old = self.n_limit

        self.top_k, self.scores_th, self.n_limit = 0, self.scores_th_eval, self.n_limit_eval
        # self.update_softdetect_parameters()

        pred0 = self.forward(batch['image0'])
        pred1 = self.forward(batch['image1'])

        self.top_k = top_k_old
        self.scores_th = scores_th_old
        self.n_limit = n_limit_old
        # self.update_softdetect_parameters()

        kpts0 = keypoints_normal2pixel(pred0['keypoints'], w0, h0)[0]
        kpts1 = keypoints_normal2pixel(pred1['keypoints'], w1, h1)[0]
        desc0 = pred0['descriptors'][0]
        desc1 = pred1['descriptors'][0]

        num_feat = min(kpts0.shape[0], kpts1.shape[0])  # number of detected keypoints

        # ==================================== pack warp params
        warp01_params, warp10_params = {}, {}
        for k, v in batch['warp01_params'].items():
            warp01_params[k] = v[0]
        for k, v in batch['warp10_params'].items():
            warp10_params[k] = v[0]

        try:
            # ==================================== covisible keypoints
            kpts0_cov, kpts01_cov, _, _ = warp(kpts0, warp01_params)
            kpts1_cov, kpts10_cov, _, _ = warp(kpts1, warp10_params)

            num_cov_feat = (len(kpts0_cov) + len(kpts1_cov)) / 2  # number of covisible keypoints

            # ==================================== get gt matching keypoints
            dist01 = compute_keypoints_distance(kpts0_cov, kpts10_cov)
            dist10 = compute_keypoints_distance(kpts1_cov, kpts01_cov)

            dist_mutual = (dist01 + dist10.t()) / 2.
            imutual = torch.arange(min(dist_mutual.shape), device=dist_mutual.device)
            dist_mutual[imutual, imutual] = 99999  # mask out diagonal

            mutual_min_indices = mutual_argmin(dist_mutual)
            dist = dist_mutual[mutual_min_indices]
            gt_num = (dist <= self.eval_gt_th).sum().cpu()  # number of gt matching keypoints

            # ==================================== putative matches
            matches_est = mutual_argmax(desc0 @ desc1.t())
            mkpts0, mkpts1 = kpts0[matches_est[0]], kpts1[matches_est[1]]

            num_putative = len(mkpts0)  # number of putative matches

            # ==================================== warp putative matches
            mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            mkpts1 = mkpts1[ids0]

            dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1)).cpu()
            if dist.shape[0] == 0:
                dist = dist.new_tensor([float('inf')])

            num_inlier = sum(dist <= self.eval_gt_th)

            return (dist,
                    num_feat,  # feature number
                    gt_num / max(num_cov_feat, 1),  # repeatability
                    num_inlier / max(num_putative, 1),  # accuracy
                    num_inlier / max(num_cov_feat, 1),  # matching score
                    num_inlier / max(gt_num, 1),  # recall
                    )
        except EmptyTensorError:
            return torch.tensor([[0]]), num_feat, 0, 0, 0, 0


    #TODO add the losses logging too!!!
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # here return the validation loss and metrics
        # JUST FOR ACCUMULATION NO LOGGING
        self.validation_step_loss(batch, batch_idx)
        dist, num_feat, repeatability, accuracy, matching_score, recall = self.val_match(batch)
        # self.log('val/kpt_num', num_feat)
        # self.log('val/repeatability', repeatability)
        # # self.log('val/acc', accuracy)
        # self.log('val/matching_score', matching_score)
        # self.log('val/recall', recall)

        # if hasattr(self.logger, 'experiment'):
        #     log_dict = {
        #     'val/kpt_num': num_feat,
        #     'val/repeatability': repeatability,
        #     'val/acc': accuracy,
        #     'val/matching_score': matching_score,
        #     'val/recall': recall,
        #     'step': self.global_step
        #     }
        #     self.logger.experiment.log(log_dict)

        self.num_feat.append(num_feat)
        self.repeatability.append(repeatability)
        self.accuracy.append(accuracy)
        self.matching_score.append(matching_score)

        # compute the MMA
        dist = dist.cpu().detach().numpy()
        if dataloader_idx == 0:
            for thr in self.rng:
                self.i_err[thr] += np.mean(dist <= thr)
        elif dataloader_idx == 1:
            for thr in self.rng:
                self.v_err[thr] += np.mean(dist <= thr)
        else:
            pass

        return {'num_feat': num_feat, 'repeatability': repeatability, 'accuracy': accuracy,
                'matching_score': matching_score}

    def on_validation_epoch_start(self):
        # reset
        for thr in self.rng:
            self.i_err[thr] = 0
            self.v_err[thr] = 0
        self.num_feat = []
        self.repeatability = []
        self.accuracy = []
        self.matching_score = []
        self.val_losses = defaultdict(int)
        self.val_batch_num = 0

    def on_validation_epoch_end(self):
        # ============= compute average
        num_feat_mean = np.mean(np.array(self.num_feat))
        repeatability_mean = np.mean(np.array(self.repeatability))
        accuracy_mean = np.mean(np.array(self.accuracy))
        matching_score_mean = np.mean(np.array(self.matching_score))
        self.log('val/kpt_num_mean', num_feat_mean)
        self.log('val/repeatability_mean', repeatability_mean)
        self.log('val/acc_mean', accuracy_mean)
        self.log('val/matching_score_mean', matching_score_mean)
        self.log('val/mean',
                 (repeatability_mean + accuracy_mean + matching_score_mean) / 3)
        loss_mean = sum(self.val_losses.values())
        self.val_losses['val/loss'] = loss_mean

        for k, v in self.val_losses.items():
            self.val_losses[k] = v / self.val_batch_num
            self.log(k, self.val_losses[k], on_epoch=True)
        

        if hasattr(self.logger, 'experiment'):
            log_dict = {
                'val/kpt_num_mean': num_feat_mean,
                'val/repeatability_mean': repeatability_mean,
                'val/acc_mean': accuracy_mean,
                'val/matching_score_mean': matching_score_mean,
                'val_metrics/mean': (repeatability_mean + accuracy_mean + matching_score_mean) / 3,
                'step': self.global_step,
                **{f'{k}': v for k, v in self.val_losses.items()},
                
            }
            
            # print("LOG DICT", log_dict)
            self.logger.experiment.log(log_dict)

        # ============= compute and draw MMA
        # self.errors['ours'] = (self.i_err, self.v_err, 0)
        # n_i = 52
        # n_v = 56
        # MMA = 0
        # for i in range(10):
        #     MMA += (self.i_err[i + 1] + self.v_err[i + 1]) / ((n_i + n_v) * 5)
        # MMA = MMA / 10
        # # MMA3 = (self.i_err[self.eval_gt_th] + self.v_err[self.eval_gt_th]) / ((n_i + n_v) * 5)
        # self.log('val_mma_mean', MMA)

        # MMA_image = draw_MMA(self.errors)

        # self.logger.experiment.add_image(f'hpatches_MMA', torch.tensor(MMA_image),
        #                                  global_step=self.global_step, dataformats='HWC')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = self.lr_scheduler(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'name': 'scheduled_lr'}]


   
    def log_image_and_score(self, batch, pred, prefix=''):
        """Log images and scores for visualization (placeholder implementation)"""
        # TODO: Implement image logging functionality
        # This is a placeholder to prevent errors
    pass
        

    def validation_step_loss(self, batch, batch_idx):
        """Validation step for PyTorch Lightning"""
        b, c, h, w = batch['image0'].shape

        # Forward pass
        pred0 = self.forward_train(batch['image0'])
        pred1 = self.forward_train(batch['image1'])

        # Compute correspondences
        correspondences, pred0_with_rand, pred1_with_rand = self.compute_correspondence(pred0, pred1, batch)
        
        # Compute validation loss (same as training loss but without gradients)
        val_loss = 0
        val_loss_package = {}

        if self.w_rp > 0:
            loss_reprojection = self.ReprojectionLocLoss(pred0_with_rand, pred1_with_rand, correspondences)
            val_loss += self.w_rp * loss_reprojection
            val_loss_package['val/loss_reprojection'] = loss_reprojection

        if self.w_sparse_nre > 0:
            loss_sparse_nre = self.SparseNRELoss(pred0_with_rand, pred1_with_rand, correspondences)
            val_loss += self.w_sparse_nre * loss_sparse_nre
            val_loss_package['val/loss_sparse_nre'] = loss_sparse_nre

        if self.w_dispersity > 0:
            loss_dispersity0 = self.DispersityPeakLoss(pred0_with_rand)
            loss_dispersity1 = self.DispersityPeakLoss(pred1_with_rand)
            loss_dispersity = (loss_dispersity0 + loss_dispersity1) / 2.
            val_loss += self.w_dispersity * loss_dispersity
            val_loss_package['val/loss_dispersity'] = loss_dispersity

        if self.w_reliable > 0:
            loss_reliable = self.ReliableLoss(pred0_with_rand, pred1_with_rand)
            val_loss += self.w_reliable * loss_reliable
            val_loss_package['val/loss_reliable'] = loss_reliable
        # Log validation losses
        # self.log('val/loss', val_loss, prog_bar=True)
        for k, v in val_loss_package.items():
            self.val_losses[k] += v.item()
        self.val_batch_num += 1

 
        pred = {'scores_map0': pred0['scores_map'],
                'scores_map1': pred1['scores_map'],
                'kpts0': [], 'kpts1': [],
                'desc0': [], 'desc1': []}
        
        for idx in range(b):
            num_det0 = pred0_with_rand['num_det'][idx]
            num_det1 = pred1_with_rand['num_det'][idx]
            pred['kpts0'].append(
                (pred0_with_rand['keypoints'][idx][:num_det0] + 1) / 2 * num_det0.new_tensor([[w - 1, h - 1]]))
            pred['kpts1'].append(
                (pred1_with_rand['keypoints'][idx][:num_det1] + 1) / 2 * num_det1.new_tensor([[w - 1, h - 1]]))
            pred['desc0'].append(pred0_with_rand['descriptors'][idx][:num_det0])
            pred['desc1'].append(pred1_with_rand['descriptors'][idx][:num_det1])

        val_accuracy = self.evaluate(pred, batch)
        # self.log('val/acc', val_accuracy, prog_bar=True)
        # if hasattr(self, 'logger') and self.logger is not None:
            # self.logger.experiment.log({'val/acc': val_accuracy, 'step': self.global_step})
        return val_loss
        
    
    def evaluate(self, pred, batch):
        b = len(pred['kpts0'])

        accuracy = []
        for idx in range(b):
            kpts0, kpts1 = pred['kpts0'][idx][:self.top_k].detach(), pred['kpts1'][idx][:self.top_k].detach()
            desc0, desc1 = pred['desc0'][idx][:self.top_k].detach(), pred['desc1'][idx][:self.top_k].detach()

            matches_est = mutual_argmax(desc0 @ desc1.t())

            mkpts0, mkpts1 = kpts0[matches_est[0]], kpts1[matches_est[1]]

            # warp
            warp01_params = {}
            for k, v in batch['warp01_params'].items():
                warp01_params[k] = v[idx]

            try:
                mkpts0, mkpts01, ids0, _ = warp(mkpts0, warp01_params)
            except EmptyTensorError:
                continue

            mkpts1 = mkpts1[ids0]

            dist = torch.sqrt(((mkpts01 - mkpts1) ** 2).sum(axis=1))
            if dist.shape[0] == 0:
                dist = dist.new_tensor([float('inf')])

            correct = dist < self.eval_gt_th
            accuracy.append(correct.float().mean())

        accuracy = torch.stack(accuracy).mean() if len(accuracy) != 0 else pred['kpts0'][0].new_tensor(0)
        return accuracy
    
    
    def training_step(self, batch, batch_idx):
        b, c, h, w = batch['image0'].shape

        # pred0 = super().extract_dense_map(batch['image0'], True)
        # pred1 = super().extract_dense_map(batch['image1'], True)
        # pred0 = super().extract_dense_map(batch['image0'])
        # pred1 = super().extract_dense_map(batch['image1'])
        pred0 = self.forward_train(batch['image0'])
        pred1 = self.forward_train(batch['image1'])
        # print(pred0.keys())

        correspondences, pred0_with_rand, pred1_with_rand = self.compute_correspondence(pred0, pred1, batch)
        # print(pred1_with_rand[''])
        loss = 0
        loss_package = {}

        # if self.w_pk > 0:
        #     loss_peaky0 = self.PeakyLoss(pred0_with_rand)
        #     loss_peaky1 = self.PeakyLoss(pred1_with_rand)
        #     loss_peaky = (loss_peaky0 + loss_peaky1) / 2.

        #     loss += self.w_pk * loss_peaky
        #     loss_package['loss_peaky'] = loss_peaky

        if self.w_rp > 0:
            loss_reprojection = self.ReprojectionLocLoss(pred0_with_rand, pred1_with_rand, correspondences)
            # print("loss_reprojection", loss_reprojection)
            loss += self.w_rp * loss_reprojection
            loss_package['loss_reprojection'] = loss_reprojection

        if self.w_sparse_nre > 0:

            loss_sparse_nre = self.SparseNRELoss(pred0_with_rand, pred1_with_rand, correspondences)
            # print("loss_sparse_nre", loss_sparse_nre)

            loss += self.w_sparse_nre * loss_sparse_nre
            loss_package['loss_sparse_nre'] = loss_sparse_nre

        if self.w_dispersity > 0:
            loss_dispersity0 = self.DispersityPeakLoss(pred0_with_rand)
            loss_dispersity1 = self.DispersityPeakLoss(pred1_with_rand)
            loss_dispersity = (loss_dispersity0 + loss_dispersity1) / 2.
            # print("loss_dispersity", loss_dispersity)
            loss += self.w_dispersity * loss_dispersity
            loss_package['loss_dispersity'] = loss_dispersity

        if self.w_reliable > 0:
            loss_reliable = self.ReliableLoss(pred0_with_rand, pred1_with_rand)
            # print("loss_reliable", loss_reliable)
            loss += self.w_reliable * loss_reliable
            loss_package['loss_reliable'] = loss_reliable


        #TODO: uncomment
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in loss_package.items():
            self.log(f'train/{k}', v, on_step=True, on_epoch=True, logger=True)

    # Direct WandB logging (as backup)
        if hasattr(self.logger, 'experiment'):
            # print("Logging to WandB")
            log_dict = {'train/loss': loss.item()}
            for k, v in loss_package.items():
                log_dict[f'train/{k}'] = v.item()
                log_dict['step'] = self.global_step
                self.logger.experiment.log(log_dict)

        pred = {'scores_map0': pred0['scores_map'],
                'scores_map1': pred1['scores_map'],
                'kpts0': [], 'kpts1': [],
                'desc0': [], 'desc1': []}
        for idx in range(b):
            num_det0 = pred0_with_rand['num_det'][idx]
            num_det1 = pred1_with_rand['num_det'][idx]
            pred['kpts0'].append(
                (pred0_with_rand['keypoints'][idx][:num_det0] + 1) / 2 * num_det0.new_tensor([[w - 1, h - 1]]))
            pred['kpts1'].append(
                (pred1_with_rand['keypoints'][idx][:num_det1] + 1) / 2 * num_det1.new_tensor([[w - 1, h - 1]]))
            pred['desc0'].append(pred0_with_rand['descriptors'][idx][:num_det0])
            pred['desc1'].append(pred1_with_rand['descriptors'][idx][:num_det1])

        # TODO: uncomment
        accuracy = self.evaluate(pred, batch)
        self.log('train/accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        # Direct WandB logging for accuracy
        if hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({'train/accuracy': accuracy.item(), 'step': self.global_step})

        if batch_idx % self.log_freq_img == 0:
            self.log_image_and_score(batch, pred, 'train_')

        assert not torch.isnan(loss)
        return loss