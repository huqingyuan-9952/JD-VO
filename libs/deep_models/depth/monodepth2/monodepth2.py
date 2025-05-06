''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-05-19
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-07-06
@LastEditors: Huangying Zhan
@Description: This is the interface for Monodepth2 depth network
'''

import numpy as np
import os
import sys
import torch

from .depth_decoder import DepthDecoder
from .layers import disp_to_depth
from .resnet_encoder import ResnetEncoder
from ..deep_depth import DeepDepth


class Monodepth2DepthNet(DeepDepth):
    """This is the interface for Monodepth2 depth network
    """
    def __init__(self, *args, **kwargs):
        super(Monodepth2DepthNet, self).__init__(*args, **kwargs)

        self.enable_finetune = False
        
    def initialize_network_model(self, weight_path, dataset, finetune):
        """initialize network and load pretrained model
        
        Args:
            weight_path (str): a directory stores the pretrained models.
                - **encoder.pth**: encoder model
                - **depth.pth**: depth decoder model
            dataset (str): dataset setup for min/max depth [kitti, tum]
            finetune (bool): finetune model on the run if True
        """
        # initilize network
        self.encoder = ResnetEncoder(18, False)
        self.depth_decoder = DepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        print("==> Initialize Depth-CNN with [{}]".format(weight_path))
        # concatenate encoders and decoders
        self.model = torch.nn.Sequential(self.encoder, self.depth_decoder)


        self.model = torch.nn.DataParallel(self.model)
        checkpoint = torch.load(weight_path)
        self.model.load_state_dict(checkpoint['model'])


        if finetune:
            self.encoder.train()
            self.depth_decoder.train()
        else:
            self.encoder.eval()
            self.depth_decoder.eval()

        # image size
        self.feed_height = 192
        self.feed_width = 640

        # dataset parameters
        if 'kitti' in dataset:
            self.min_depth = 0.1
            self.max_depth = 100
            self.stereo_baseline_multiplier = 5.4
        elif 'tum' in dataset:
            self.min_depth = 0.1
            self.max_depth = 10
            self.stereo_baseline_multiplier = 1
        elif 'robotcar' in dataset:
            self.min_depth = 0.1
            self.max_depth = 100
            self.stereo_baseline_multiplier = 5.4
        else:
            self.min_depth = 0.1
            self.max_depth = 100
            self.stereo_baseline_multiplier = 5.4

    def inference(self, img):
        """Depth prediction

        Args:
            img (tensor, [Nx3HxW]): image 

        Returns:
            a dictionary containing depths and disparities at different scales, resized back to input scale

                - **depth** (dict): depth predictions, each element is **scale-N** (tensor, [Nx1xHxW]): depth predictions at scale-N
                - **disp** (dict): disparity predictions, each element is **scale-N** (tensor, [Nx1xHxW]): disparity predictions at scale-N
        """
        _, _, original_height, original_width = img.shape

        # Prediction
        features = self.encoder(img)
        pred_disps = self.depth_decoder(features)

        outputs = {'depth': {}, 'disp': {}}
        for s in self.depth_scales:
            disp = pred_disps[('disp', s)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode='bilinear', align_corners=True)

            scaled_disp, _ = disp_to_depth(disp_resized, self.min_depth, self.max_depth)
            outputs['depth'][s] = 1. / scaled_disp # monodepth2 assumes 0.1 unit baseline
            outputs['disp'][s] = scaled_disp
            
        return outputs

    def inference_depth(self, img):
        """Depth prediction

        Args:
            img (tensor, [Nx3HxW]): image 

        Returns:
            depth (tensor, [Nx1xHxW]): depth prediction at highest resolution
        """
        if self.enable_finetune:
            predictions = self.inference(img)
        else:
            predictions = self.inference_no_grad(img)
        self.pred_depths = predictions['depth']
        self.pred_disps = predictions['disp']

        depth = self.pred_depths[0].clone() * self.stereo_baseline_multiplier # monodepth2 assumes 0.1 unit baseline
        return depth
