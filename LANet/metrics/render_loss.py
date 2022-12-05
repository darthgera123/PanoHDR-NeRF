"""
RMSE: Render MSE Loss between rendered images relit using GT and Pred probes
"""


import os, json, cv2, random
import numpy as np 
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from render_transportmat import RenderWithTransportMat
from skimage.transform import resize
import torch
from torch import nn
import matplotlib.pyplot as plt


def make_mask(img,x,y):
    print(img.shape)
    k = 20
    img[x-k:x+k,y-k:y+k,:] = 1
    return img

def renderLoss(gt, pred):
    resize = nn.AvgPool2d(2, stride=2)
    render_eng = RenderWithTransportMat(
            transportMatFname='transportMat.BumpySphereMiddle.top.e64.r64.half.mat', \
            lightHeight=64, doHalfMat=True)
    l2loss = nn.MSELoss()
    resize_gt = resize(resize(gt))
    resize_pred = resize(resize(pred))
    render_gt = render_eng.render_top_down(resize_gt)
    render_pred = render_eng.render_top_down(resize_pred)
    l1 = l2loss(
        render_gt['top'], render_pred['top'])
    l2 = l2loss(
        render_gt['bottom'], render_pred['bottom'])
    weight = 1
    render_loss = weight*(l1+l2)
    render_stuff = {'resize_gt':resize_gt,'resize_pred':resize_pred,'render_gt':render_gt,\
                    'render_pred':render_pred,'render_loss':render_loss}
    return render_stuff

def np2torch(img):
    img_torch = torch.from_numpy(img).type('torch.FloatTensor').permute(2,0,1)
    img_torch = img_torch.unsqueeze(axis=0)
    return img_torch

def torch2np(img):
    return img.squeeze().permute(1,2,0).numpy()

def make_image(render_stuff,branch='gt'):
    render_top = render_stuff[f'render_{branch}']['top']
    render_bottom = render_stuff[f'render_{branch}']['bottom']
    resize = render_stuff[f'resize_{branch}']

    render_top = torch2np(render_top)
    render_bottom = torch2np(render_bottom)
    resize = torch2np(resize)

    render_top_img = np.clip(np.power(render_top+1e-8, 0.45), 0, 1)*255
    
    render_bottom_img = np.clip(np.power(render_bottom+1e-8, 0.45), 0, 1)*255
    
    resize_img = np.clip(np.power(resize+1e-8, 0.45), 0, 1)*255
    
    render_img = np.hstack((resize_img,render_top_img,render_bottom_img))
    
    return render_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='', help='directory to data')
    parser.add_argument('--gt_dir',type=str,default='',help='directory to data')
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()
    

    # os.makedirs(args.output_dir,exist_ok=True)
    
    imgs = os.listdir(args.pred_dir)
    loss =0
    for img in tqdm(imgs):
        name = img.split('.')[0]
        pred = cv2.imread(f'{args.pred_dir}/{img}',-1)
        pred = resize(pred,(256,512),anti_aliasing=True)
        
        gt = cv2.imread(f'{args.gt_dir}/{img}',-1)
        gt = resize(gt,(256,512),anti_aliasing=True)
        
        pred_torch = np2torch(pred)
        gt_torch = np2torch(gt)
        render_stuff = renderLoss(gt_torch,pred_torch)
        loss += render_stuff['render_loss'].item()
        # gt_img = make_image(render_stuff,'gt')
        # cv2.imwrite(f'{args.output_dir}/{name}_gt.png',gt_img)
        # pred_img = make_image(render_stuff,'pred')
        # cv2.imwrite(f'{args.output_dir}/{name}_pred.png',pred_img)

    print(loss/len(imgs))
    
