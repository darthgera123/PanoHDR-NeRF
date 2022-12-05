"""
Model containing LANet
Dual stream network with Resnet 50 backbone
"""

import numpy as np
import sys
import torch
import torch.nn as nn

import torch.nn.functional as F
from resnet import ResNet50
from loss import TrainLoss

def deconv(in_channel,out_channel,kernel,stride=1):
	return nn.Sequential(
					# nn.ReLU(False),
					nn.SELU(False),
					nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=1),
					nn.InstanceNorm2d(out_channel))

def upconv(in_channel,out_channel,kernel,stride=1):
	return nn.Sequential(
					# nn.ReLU(inplace=False),
					nn.SELU(False),
					nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
					nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=1),
					nn.InstanceNorm2d(out_channel))

def conv(in_channel,out_channel,kernel,stride=2):
	if kernel == 3:
		return nn.Sequential(
						# nn.ReLU(False),
						nn.SELU(False),
						nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=stride,padding=0),
						nn.InstanceNorm2d(out_channel))
	else:
		return nn.Sequential(
						# nn.ReLU(False),
						nn.SELU(False),
						nn.Conv2d(in_channel,out_channel,kernel_size=kernel,stride=1,padding=0),
						nn.InstanceNorm2d(out_channel))



def skip(inp_channel,skip_channel,out_channel,final=False):
	return nn.Conv2d(inp_channel+skip_channel,out_channel,kernel_size=1,stride=1,padding=0)


class LAnet(nn.Module):
	def __init__(self,in_channels=3):
		super(LAnet,self).__init__()

		self.residual = ResNet50(3,deepths=[3,4,6,3])
		self.e6 = deconv(2048,1024,3,stride=2)
		self.e7 = deconv(1024,1024,3,stride=2)
		self.d6 = upconv(1024,1024,3,stride=1)
		

		self.s5 = conv(2048,512,1,stride=1)
		self.d5p = upconv(1024,512,3,stride=2)
		self.d5 = upconv(512,512,3,stride=1)
		self.sc5 = skip(512,512,512)
		
		self.s4 = conv(1024,256,1,stride=1)
		self.d4p = upconv(512,256,3,stride=2)
		self.d4 = upconv(256,256,3,stride=1)
		self.sc4 = skip(256,256,256)

		self.s3 = conv(512,128,1,stride=1)
		self.d3p = upconv(256,128,3,stride=2)
		self.d3 = upconv(128,128,3,stride=1)
		self.sc3 = skip(128,128,128)

		self.s2 = conv(256,64,1,stride=1)
		self.d2p = upconv(128,64,3,stride=2)
		self.d2 = upconv(64,64,3,stride=1)
		self.sc2 = skip(64,64,64)

		self.s1 = conv(64,64,1,stride=1)
		self.d1 = upconv(64,64,3,stride=2)
		self.sc1 = skip(64,64,64)

		# attention module
		self.h1 = upconv(64,64,3,stride=1)

		self.a2p = upconv(128,64,3,stride=2)
		self.a2 = upconv(64,64,3,stride=1)

		self.a1p = upconv(64,64,3,stride=2)
		self.a1 = nn.Sequential(
					nn.ReLU(inplace=False),
					nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
					nn.Conv2d(64,2,kernel_size=3,stride=1,padding=1))
		# self.a1 = upconv(64,2,3,stride=1)

		self.a1p1 = upconv(64,64,3,stride=1)

		self.a0 = nn.Sigmoid()

		self.h0 = upconv(128,64,3,stride=1)
		self.d0p = upconv(64,64,3,stride=2)
		self.d0 = upconv(64,64,3,stride=1)

		self.final = nn.Sequential(
						nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
						nn.Conv2d(64+3,3,kernel_size=3,stride=2,padding=1))

		self.mask = nn.Sequential(
						nn.ReLU(True),
						nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
						nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1))

			

	def forward(self,x):

		inp = x

		out, res = self.residual(x)
		e6 = self.e6(out)
		e7 = self.e7(e6)
		d6 = self.d6(e7)
		

		s5 = self.s5(res[4])
		d5 = self.d5p(d6)
		d5 = self.d5(d5)
		d5p = torch.cat([d5,s5],dim=1)
		d5p = self.sc5(d5p)
		

		s4 = self.s4(res[3])
		d4 = self.d4p(d5p)
		d4 = self.d4(d4)
		d4p = torch.cat([d4,s4],dim=1)
		d4p = self.sc4(d4p)
		

		s3 = self.s3(res[2])
		d3 = self.d3p(d4p)
		d3 = self.d3(d3)
		d3p = torch.cat([d3,s3],dim=1)
		d3p = self.sc3(d3p)
		

		s2 = self.s2(res[1])
		d2 = self.d2p(d3p)
		d2 = self.d2(d2)
		d2p = torch.cat([d2,s2],dim=1)
		d2p = self.sc2(d2p)
		


		s1 = self.s1(res[0])
		d1 = self.d1(d2p)
		d1p = torch.cat([d1,s1],dim=1)
		sc1 = self.sc1(d1p)
		
		
		h1 = self.h1(d1)
		a2p = self.a2p(d3p)
		a2 = self.a2(a2p)
		a1p = self.a1p(a2)
		a1 = self.a1(a1p)
		
		a1p1 = self.a1p1(a1p)
		a0 = self.a0(a1)
		# print(a0.shape)
		att1,att2 = torch.split(a0,1,dim=1)
		mul1 = torch.multiply(h1,att1)
		mul2 = torch.multiply(h1,att2)
		att_out = torch.cat([mul1,mul2],dim=1)
		
		h0 = self.h0(att_out)
		d0p = self.d0p(h0)
		# d0 = self.d0(d0p)
		
		skip_layer = torch.maximum(inp,torch.tensor([1e-8]).cuda())
		skip_layer = torch.log(skip_layer)
		
		
		out = torch.cat([d0p,skip_layer],dim=1)
		out = self.final(out)
		mask = self.mask(a1p1)
		return out,mask


if __name__ == '__main__':
	hdr_gt = torch.rand([8,3,256,512]).cuda()
	mask_gt = torch.rand([8,3,256,512]).cuda()
	torch.autograd.set_detect_anomaly(True)
	model = LAnet().cuda()
	criterion = TrainLoss()
	hdr_pred,mask = model(hdr_gt)
	# loss = criterion(hdr_pred,hdr_gt,mask_pred,mask_gt)
	# loss.backward()
	print(hdr_pred.shape)

			

