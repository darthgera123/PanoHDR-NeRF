"""
Loss function for training LDR2HDR module
"""

import torch
import os
import sys
import cv2
import json
import argparse
import random
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim
import torchvision.models as tvmodels


import torchvision.transforms as transforms
import numpy as np
from render_transportmat import RenderWithTransportMat
from functools import reduce


class TrainLoss(nn.Module):

	def __init__(self, transport_matrix='transportMat.BumpySphereMiddle.top.e64.r64.half.mat', img_dim=64\
					,isBottom=False):
		super().__init__()
		self.mask_loss = nn.NLLLoss()
		self.log = nn.LogSoftmax(dim=1)
		self.render_eng = RenderWithTransportMat(
		    transportMatFname=transport_matrix, lightHeight=img_dim, doHalfMat=True)
		self.resize = nn.AvgPool2d(2, stride=2)
		self.siMSE = 0
		self.maskLoss = 0
		self.renderLoss = 0
		self.hueLoss = 0
		self.isBottom = isBottom
		self.l2loss = nn.MSELoss()
		self.l1loss = nn.L1Loss()

	def _siMSE(self, gt, pred, alpha=1.0):
		return torch.mean(torch.mean((gt-pred)**2, axis=[1, 2, 3])
								- alpha*torch.pow(torch.mean(gt-pred, axis=[1, 2, 3]), 2))

	def _maskLoss(self, mask_gt, mask_pred):
		mp = mask_pred
		mg = mask_gt.long()
		l1 = self.mask_loss(self.log(mp), mg[:, 0, :, :])
		l2 = self.mask_loss(self.log(mp), mg[:, 0, :, :])
		l3 = self.mask_loss(self.log(mp), mg[:, 0, :, :])
		return l1+l2+l3

	def _renderLoss(self, gt, pred):
		resize_gt = self.resize(self.resize(gt))
		resize_pred = self.resize(self.resize(pred))
		render_gt = self.render_eng.render_top_down(resize_gt)
		render_pred = self.render_eng.render_top_down(resize_pred)
		l1 = self.l2loss(
		    render_gt['top'], render_pred['top'])
		l2 = self.l2loss(
		    render_gt['bottom'], render_pred['bottom'])
		weight = 1
		
		if self.isBottom:
			render_loss = weight*(l1+l2)
		else:
			render_loss = l1
			
		return render_loss

	

	def forward(self,hdr_pred,hdr_gt,mask_pred,mask_gt,fixed_mask):
		mu = torch.tensor([5000]).cuda()
		one = torch.tensor([1]).cuda()

		img_gt = torch.exp(hdr_gt) 
		img_pred = torch.exp(hdr_pred) 
		
		img_gt = img_gt* fixed_mask + torch.tensor([1e-5]).cuda()
		img_pred = img_pred* fixed_mask + torch.tensor([1e-5]).cuda()
		
		self.siMSE = self._siMSE(hdr_gt,hdr_pred)
		self.maskLoss = 0.05*self._maskLoss(mask_gt,mask_pred)
		self.renderLoss = self._renderLoss(img_gt,img_pred)
		return self.siMSE+self.maskLoss+self.renderLoss
		
