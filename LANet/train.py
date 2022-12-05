"""
Pytorch port of Luminance Attention Net
"""

from loss import TrainLoss
from skimage import metrics
from model import LAnet 
from dataset import HDRDataset
import argparse
import cv2
import numpy as np
import os
import random
import tensorboardX
import time
import torch
import torch.nn as nn
from torch.optim import Adam,AdamW,lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from render_transportmat import RenderWithTransportMat
from functools import reduce

import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)



def tensor2np(batched_image):
    return batched_image[0, :, :, :].detach().cpu().numpy()

# Generate mask for LDR2HDR module
def generate_mask(hdr):
    mean_hdr = torch.mean(hdr,axis=1)
    mask_o = torch.where(mean_hdr > 0.1, torch.ones_like(mean_hdr),torch.zeros_like(mean_hdr))
    mask_u = torch.where(mean_hdr < -5.5, torch.ones_like(mean_hdr),torch.zeros_like(mean_hdr))
    mask_c = 1.0 - (mask_o+mask_u)
    mask = torch.cat([mask_o.unsqueeze(dim=1),mask_u.unsqueeze(dim=1),mask_c.unsqueeze(dim=1)],dim=1)
    return mask

# Function for saving image
def view_img(image, gamma_correct=True, exposure=1):
    output = np.clip(np.clip(image, 0, 1)*exposure, 0, 1)
    if gamma_correct:
        output = output ** (1.0/2.2)
    output = output * 255.0
    output = output.astype(np.uint8)
    output = np.transpose(output, (1, 2, 0))
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = np.transpose(output, (2, 0, 1))
    return output


def renderLoss(gt,pred,resize_layer,render_mat,isBottom):
    '''
    Render Loss between GT and Pred rendered images
    '''
    brightness = 0.3
    resize_gt = resize_layer(resize_layer(gt))*brightness
    resize_pred = resize_layer(resize_layer(pred))*brightness
    render_gt = render_mat.render_top_down(resize_gt)
    render_pred = render_mat.render_top_down(resize_pred)
    l1 = siMSE(torch.log(render_gt['top']),torch.log(render_pred['top']))
    l2 = siMSE(torch.log(render_gt['bottom']),torch.log(render_pred['bottom']))
    l2loss = nn.MSELoss()
    l1loss = nn.L1Loss()
    weight = 10
    if isBottom:
        render_loss = weight*(l1+l2)
    else:
        render_loss = weight*(l1)
    render_stuff = {'resize_gt':resize_gt,'resize_pred':resize_pred,'render_gt':render_gt,'render_pred':render_pred}
    return render_loss,render_stuff

if __name__ == '__main__':
    """
    Pass arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ldr', type=str, default='')
    parser.add_argument('--train_hdr', type=str, default='')
    parser.add_argument('--val_ldr', type=str, default='')
    parser.add_argument('--val_hdr', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--save_ckpt', type=int, default=10)
    parser.add_argument('--logs', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--load_ckpt', type=str, default='')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--isBottom', action='store_true',help='consider bottom half of probe or not')
    parser.add_argument('--isMaskRender', action='store_true',help='multiply with probe with mask')
    args = parser.parse_args()

    device = 'cuda'

    named_tuple = time.localtime()
    time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
    time_string = f'{args.exp_name}_{time_string}'
    log_dir = os.path.join(args.logs, time_string)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_dir = args.ckpt + time_string
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    writer = tensorboardX.SummaryWriter(logdir=log_dir)    

    """
    Load Dataset
    """
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()])

    train_dataset = HDRDataset(
        args.train_hdr, args.train_ldr,train_transforms, width=args.width, height=args.height)
    val_dataset = HDRDataset(args.val_hdr, args.val_ldr, train_transforms,
                             width=args.width, height=args.height)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4)

    if args.load_ckpt:
        model = torch.load(args.load_ckpt)
    else:
        model = LAnet()
        model = model.to(device)

    optim = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = TrainLoss(isBottom=args.isBottom)
    criterion_test = TrainLoss(isBottom=args.isBottom)

    # Render transport matrix
    render_mat = RenderWithTransportMat(transportMatFname='transportMat.BumpySphereMiddle.top.e64.r64.half.mat', lightHeight=64, doHalfMat=True)
    resize_layer = nn.AvgPool2d(2, stride=2)

    def siMSE(gt,pred,alpha=1.0):
            return torch.mean(torch.mean((gt-pred)**2,axis=[1,2,3]) \
                                    - alpha*torch.pow(torch.mean(gt-pred, axis=[1,2,3]),2))
    l2loss = nn.MSELoss()

    """
    Training code
    """
    for epoch in tqdm(range(args.epochs)):
        model.train()
        torch.set_grad_enabled(True)
        train_loss = 0
        render_loss = 0
        l2_loss = 0
        simse_loss = 0
        train_img = []
        train_img_half = []
        train_gt_render = []
        train_pred_render = []
        for samples in tqdm(train_dataloader, desc=f'Train: Epoch {epoch}'):

            hdr_gt,ldr,mask_fixed = samples
            hdr_gt,ldr,mask_fixed = hdr_gt.to(device),ldr.to(device),mask_fixed.to(device)
            mask_gt = generate_mask(hdr_gt)
            
            hdr_pred, mask_pred = model(ldr.float())
            optim.zero_grad()
            # multiply probes with mask and only relight with bright spots
            render,render_stuff = renderLoss(torch.exp(hdr_gt*mask_fixed),torch.exp(hdr_pred*mask_fixed),\
                                        resize_layer,render_mat,args.isBottom)
            
            loss = criterion(hdr_pred, hdr_gt,mask_pred,mask_gt,mask_fixed)
            l2_loss += l2loss(hdr_gt,hdr_pred).item()
            simse_loss += siMSE(hdr_gt,hdr_pred).item()
            train_loss += loss.item()
            render_loss += render.item()
            loss.backward()
            optim.step()
            

            # Save the  training images
            # Rendered images with gt and pred probes
            pano_gt = tensor2np(render_stuff['resize_gt'])**(2.2)
            pano_gt = np.clip(np.power(pano_gt+1e-8, 0.45), 0, 1)
            gt_top = tensor2np(render_stuff['render_gt']['top'])
            gt_bottom = tensor2np(render_stuff['render_gt']['bottom'])
            gt_render = np.concatenate((pano_gt,gt_top,gt_bottom),axis=2)

            pano_pred = tensor2np(render_stuff['resize_pred'])**(2.2)
            pano_pred = np.clip(np.power(pano_pred+1e-8, 0.45), 0, 1)
            pred_top = tensor2np(render_stuff['render_pred']['top'])
            pred_bottom = tensor2np(render_stuff['render_pred']['bottom'])
            pred_render = np.concatenate((pano_pred,pred_top,pred_bottom),axis=2)

            train_gt_render.append(gt_render)
            train_pred_render.append(pred_render)
            img_np = np.concatenate((view_img(tensor2np(ldr), False, exposure=1)\
                        ,view_img(np.exp(tensor2np(hdr_pred)),True, exposure=1)\
                        ,view_img(np.exp(tensor2np(hdr_gt)), exposure=1)),axis=2)

            img_np_half = np.concatenate((view_img(tensor2np(ldr), False, exposure=0.01)\
                        ,view_img(np.exp(tensor2np(hdr_pred)),True, exposure=0.01)\
                        ,view_img(np.exp(tensor2np(hdr_gt)), exposure=0.01)),axis=2)



            train_img.append(img_np)
            train_img_half.append(img_np_half)
        
        
        writer.add_scalar("train_loss", train_loss/len(train_dataloader),epoch)
        writer.add_scalar("render_loss", render_loss/len(train_dataloader),epoch)
        

        """
        Validation code
        """

        test_img = []
        test_mask = []
        test_fixed = []
        test_render_gt = []
        test_render_pred = []
        test_img_hdr = []
        model.eval()
        test_loss = 0
        render_test = 0
        chroma_test = 0
        for samples in tqdm(val_dataloader, desc=f'Test: Epoch {epoch}'):
            hdr_gt,ldr,mask_fixed = samples
            hdr_gt,ldr,mask_fixed = hdr_gt.to(device),ldr.to(device),mask_fixed.to(device)
            mask_gt = generate_mask(hdr_gt)

            hdr_pred, mask_pred = model(ldr)
            # hdr_new = torch.log(ldr)+mask_fixed*hdr_pred
            # multiply probes with mask and only relight with bright spots 
            render,render_stuff = renderLoss(torch.exp(hdr_gt*mask_fixed),torch.exp(hdr_pred*mask_fixed),\
                                        resize_layer,render_mat,args.isBottom)
            
            loss = criterion(hdr_pred, hdr_gt,mask_pred,mask_gt,mask_fixed)
            test_loss += loss.item()
            render_test += render.item()
            
            # Save the  test images
            # Rendered images with gt and pred probes
            pano_gt = tensor2np(render_stuff['resize_gt'])**(2.2)
            pano_gt = np.clip(np.power(pano_gt+1e-8, 0.45), 0, 1)
            gt_top = tensor2np(render_stuff['render_gt']['top'])
            gt_bottom = tensor2np(render_stuff['render_gt']['bottom'])
            gt_render = np.concatenate((pano_gt,gt_top,gt_bottom),axis=2)

            pano_pred = tensor2np(render_stuff['resize_pred'])**(2.2)
            pano_pred = np.clip(np.power(pano_pred+1e-8, 0.45), 0, 1)
            pred_top = tensor2np(render_stuff['render_pred']['top'])
            pred_bottom = tensor2np(render_stuff['render_pred']['bottom'])
            pred_render = np.concatenate((pano_pred,pred_top,pred_bottom),axis=2)
     
            mask_pred_np = view_img(tensor2np(mask_pred), False)
            mask_gt_np = view_img(tensor2np(mask_gt), False)
            mask_fixed_np = view_img(tensor2np(mask_fixed), False)

            img_np = np.concatenate((view_img(tensor2np(ldr), False, exposure=1)\
                        ,view_img(np.exp(tensor2np(hdr_pred)),True, exposure=1)\
                        ,view_img(np.exp(tensor2np(hdr_gt)), exposure=1)),axis=2)
            
            mask_np = np.concatenate((mask_pred_np,mask_gt_np,mask_fixed_np),axis=2)

            test_img.append(img_np)
            test_mask.append(mask_np)
            test_render_gt.append(gt_render)
            test_render_pred.append(pred_render)
            
        """
        Save the losses to tensorboard
        """
        # print("test_loss", test_loss/len(val_dataloader))
        # print("render_test", render_test/len(train_dataloader))
        writer.add_scalar("test_loss", test_loss/len(val_dataloader), epoch)
        writer.add_scalar("render_test", render_test/len(train_dataloader),epoch)
        train_ridx = random.randint(0, len(train_img)-1)
        test_ridx = random.randint(0, len(test_img)-1)

        writer.add_image("train gt render", view_img(
            train_gt_render[train_ridx], False, exposure=1), epoch)
        writer.add_image("train pred render", view_img(
            train_pred_render[train_ridx], False, exposure=1), epoch)
        writer.add_image("Img 1 LDR PRED GT",train_img[train_ridx] , epoch)
        writer.add_image("Img 0.05 LDR PRED GT",train_img_half[train_ridx] , epoch)

        writer.add_image("gt render", view_img(
            test_render_gt[test_ridx], False, exposure=1), epoch)
        writer.add_image("pred render", view_img(
            test_render_pred[test_ridx], False, exposure=1), epoch)

        writer.add_image("Test Image LDR PRED GT",test_img[test_ridx] , epoch)
        writer.add_image("Test Mask Pred GT Fixed", test_mask[test_ridx], epoch)
        
        
        if epoch % args.save_ckpt == 0:
            print("Saving Checkpoint")
            torch.save(model, checkpoint_dir + '/epoch_{}.pt'.format(epoch))




    # gt_save = np.transpose(np.exp(test_hdr_gt[ridx]), (1, 2, 0)) 
        # cv2.imwrite('gt.exr', gt_save.astype('float32'))
        # cv2.imwrite('pred.exr', np.transpose(np.exp(test_hdr_out[ridx]), (1, 2, 0)))
        # cv2.imwrite('gt_render.jpg',np.transpose(test_render_gt[ridx],(1,2,0))*255)
        # cv2.imwrite('pred_render.jpg',np.transpose(test_render_pred[ridx],(1,2,0))*255)