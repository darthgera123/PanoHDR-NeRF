import cv2
import os
import argparse
from skimage.transform import resize
import numpy as np
from tqdm import tqdm

def calibrate(ldr_img,hdr_img):
	ldr_img = resize(ldr_img,(480,960),anti_aliasing=True)
	hdr_img = resize(hdr_img,(480,960),anti_aliasing=True)

	LDR_I = np.mean(ldr_img,axis=2)
	mask = np.where(LDR_I > 0.83, 0, 1)
	mean_LDR = np.mean(mask * LDR_I)
	hdr_img = np.clip(hdr_img,1e-8,1e4)
	HDR_I = np.mean(hdr_img, axis=2)
	mean_HDR = np.mean(mask * HDR_I)
	hdr_img_expose = hdr_img * mean_LDR / mean_HDR
	return hdr_img_expose

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--hdr_dir', type=str, default='')
	parser.add_argument('--ldr_dir', type=str, default='')
	parser.add_argument('--output_dir', type=str, default='')
	args = parser.parse_args()
	imgs = os.listdir(args.hdr_dir)
	os.makedirs(args.output_dir,exist_ok=True)
	for i in tqdm(range(1,11)):
		ldr = cv2.imread(f'{args.ldr_dir}/ldr{i}.jpg')
		hdr = cv2.imread(f'{args.hdr_dir}/hdr{i}.exr',-1)
		hdr_expose = calibrate(ldr,hdr)
		cv2.imwrite(f'{args.output_dir}/hdr{i}.exr',hdr_expose.astype('float32'))
