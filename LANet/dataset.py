"""
Loading Dataset and applying augmentation
"""

import torch
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import argparse
from skimage.transform import resize
import numpy as np
import random
from io import BytesIO
from color_jitter import colorJitterSmall,colorJitterLarge,colorJitter



class HDRDataset(Dataset):
	def __init__(self, hdr_path, ldr_path, ldr_transforms, width=512, height=256):
		# super().__init__(HDRDataset)
		self.hdr_dir = hdr_path
		self.ldr_dir = ldr_path
		self.width = width
		self.height = height
		self.ldr_transforms = ldr_transforms

	def __len__(self):
		return len(os.listdir(self.hdr_dir))

	def _rotate_image(self, hdr_img, ldr_img):
	    range_roll = np.linspace(100, 500, 150)
	    pick = random.randint(0, len(range_roll)-1)
	    hdr_img = np.roll(hdr_img, int(range_roll[pick]), axis=1)
	    ldr_img = np.roll(ldr_img, int(range_roll[pick]), axis=1)
	    return hdr_img, ldr_img

	def _hsv(self, ldr_img):
		ldr_img = (ldr_img*255).astype('uint8')
		shift_h = random.randint(-10, 10)
		ldr_img_hsv = cv2.cvtColor(ldr_img, cv2.COLOR_RGB2HSV)
		h, s, v = cv2.split(ldr_img_hsv)	    
		h = ((h + shift_h) % 180)
		ldr_shift_hsv = cv2.merge([h, s, v])
		ldr_final = cv2.cvtColor(ldr_shift_hsv,cv2.COLOR_HSV2RGB)

		return ldr_final/255.0

	def _toneshift(self,ldr_img):
		range_roll = np.linspace(-0.1,0.1,100)
		pick = random.randint(0,len(range_roll)-1)
		return np.power(ldr_img,1+range_roll[pick])

	def _intensity(self,hdr_img,ldr_img):
	    range_roll = np.linspace(-1,1,100)
	    pick = random.randint(0,len(range_roll)-1)
	    hdr_img = hdr_img* np.power(2,range_roll[pick])

	    ldr_img = np.clip(ldr_img* np.power(2,range_roll[pick]),0,1)
	    return hdr_img,ldr_img
	
	def _convert(self,img):
		im_gamma_correct = np.clip(np.power(img, 0.45), 0, 1)
		im_fixed = np.uint8(im_gamma_correct*255)
		return im_fixed

	
	def _create_mask(self,img):
		alpha = 0.85
		img_gray = cv2.cvtColor((img*255).astype('uint8'),cv2.COLOR_RGB2GRAY)/255.0
		mask = np.maximum(0,img_gray-alpha)/(1-alpha)
		mask = np.clip(mask,0,1)
		mask = mask.reshape((self.height,self.width,1))
		return mask

	
	def _smooth_mask(self,img):
		alpha = 0.90
		img_gray = cv2.cvtColor((img*255).astype('uint8'),cv2.COLOR_RGB2GRAY)/255.0
		mask = img_gray
		mask[mask<0.7] = 0
		mask[mask>0.85] = 1
		mask = np.maximum(0,mask-0.7)/(1-0.7)
		mask = np.clip(mask,0,1)
		mask = mask.reshape((self.height,self.width,1))
		return mask

	def _binary_mask(self,img):
		alpha = 0.90
		img_gray = cv2.cvtColor((img*255).astype('uint8'),cv2.COLOR_RGB2GRAY)/255.0
		mask = img_gray
		mask[mask<0.85] = 0
		mask[mask>0.85] = 1
		# mask = np.maximum(0,mask-0.7)/(1-0.7)
		mask = np.clip(mask,0,1)
		mask = mask.reshape((self.height,self.width,1))
		return mask

	def _pre_hdr_p2(self,hdr):
		hdr_mean = np.mean(hdr)
		hdr = 0.5 * hdr / (hdr_mean + 1e-6)
		return hdr

	def _unsharp_mask(self,image, kernel_size=(5, 5)):
		
		"""Return a sharpened version of the image, using an unsharp mask."""
		# amount = 1
		# sigma = 1.0
		image = (image*255).astype('uint8')
		range_roll = np.linspace(0,1,100)
		pick = random.randint(0,len(range_roll)-1)
		amount = range_roll[pick]

		range_roll = np.linspace(0,3,100)
		pick = random.randint(0,len(range_roll)-1)
		sigma = range_roll[pick]

		blurred = cv2.GaussianBlur(image, kernel_size, sigma)
		sharpened = float(amount + 1) * image - float(amount) * blurred
		sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
		sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
		sharpened = sharpened.round().astype(np.uint8)
		# if threshold > 0:
		#     low_contrast_mask = np.absolute(image - blurred) < threshold
		#     np.copyto(sharpened, image, where=low_contrast_mask)
		return sharpened/255.0


	def __getitem__(self,index):
		img_name = sorted(os.listdir(self.hdr_dir))[index].split('.')[0]

		hdr_path = os.path.join(self.hdr_dir,img_name+'.exr')
		
		hdr_img = cv2.imread(hdr_path,-1)
		hdr_img = resize(hdr_img,(self.height,self.width),anti_aliasing=True)
		
		ldr_path = os.path.join(self.ldr_dir,img_name+'.jpg')
		ldr_img = cv2.imread(ldr_path)
		# resize makes the range 0-255 -> 0-1
		ldr_img = resize(ldr_img,(self.height,self.width),anti_aliasing=True)
		# augmentations for both hdr and ldr images
		hdr_img,ldr_img = self._rotate_image(hdr_img,ldr_img)
		if random.randint(0,2) %2 == 1:
			hdr_img = np.fliplr(hdr_img)
			ldr_img = np.fliplr(ldr_img)
		
		fixed_mask = self._create_mask(ldr_img)
		# expose HDR correctly
		ldr_img_gamma = ldr_img ** (2.2)
		LDR_I = np.mean(ldr_img_gamma,axis=2)
		mask = np.where(LDR_I > 0.83, 0, 1).reshape((256,512,1))
		mean_LDR = (mask * ldr_img_gamma).reshape(-1,3).mean(0)
		hdr_img_expose = np.clip(hdr_img,1e-5,1e6)
		HDR_I = np.mean(hdr_img_expose,axis=2)
		mean_HDR = (mask * hdr_img_expose).reshape(-1,3).mean(0)
		hdr_img_expose *= mean_LDR/mean_HDR
		

		#augmentations for ldr images
		ldr_img_aug = ldr_img
		# ldr_img_aug = colorJitterSmall(ldr_img.astype('float32'))
		# ldr_img_aug = self._unsharp_mask(ldr_img_aug)
		# ldr_img_aug = ldr_img_aug+0.01*np.random.normal(scale=0.01,size=ldr_img_aug.shape)
		# ldr_img_aug = np.clip(ldr_img_aug,1e-5,1)
		# ldr_img_aug = self._toneshift(ldr_img_aug)
		
		# ldr_img_aug = ldr_img_aug**(2.2)
		ldr_img_gt = torch.from_numpy(np.clip(ldr_img_aug,1e-5,1))
		ldr_img_gt = ldr_img_gt.permute(2,0,1)
		ldr_img_gt = ldr_img_gt.float()


		hdr_img_gt = torch.from_numpy(hdr_img_expose)
		hdr_img_gt = hdr_img_gt.permute(2,0,1)
		hdr_img_gt = torch.log(hdr_img_gt)

		fixed_mask_gt = torch.from_numpy(fixed_mask)
		fixed_mask_gt = fixed_mask_gt.permute(2,0,1).float()
		
		return hdr_img_gt, ldr_img_gt , fixed_mask_gt

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--hdr_dir', type=str, default='')
	parser.add_argument('--ldr_dir', type=str, default='')
	args = parser.parse_args()
	train_transforms = transforms.Compose([
						transforms.ToPILImage(),
						transforms.ToTensor(),
						])
	train_dataset = HDRDataset(args.hdr_dir,args.ldr_dir,train_transforms)
	print(train_dataset.__len__())
	hdr,ldr,mask = train_dataset.__getitem__(5)
	
	
	ldr_gt = ldr
	ldr_gt = ldr_gt.numpy().transpose(1,2,0)

	
	hdr_log = hdr
	hdr = hdr.numpy().transpose(1,2,0)
	hdr = np.exp(hdr+1e-5)
	
	mask = mask.numpy().transpose(1,2,0)
	hdr_log = hdr_log.numpy().transpose(1,2,0)
	ldr = ldr.numpy().transpose(1,2,0)**(1/2.2)
	ldr = ldr*255
	
	cv2.imwrite('results/hdr_img.exr',hdr.astype('float32'))

	cv2.imwrite('results/ldr_img.jpg',ldr.astype('uint8'))
	cv2.imwrite('results/mask_img.jpg',(mask*255).astype('uint8'))

	
	
		

