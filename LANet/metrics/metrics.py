import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--gt_dir', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()
    global_psnr = 0.0
    global_ssim = 0.0
    psnr_input = 0.0
    psnr_gt = 0.0

    imgs = os.listdir(os.path.join(args.gt_dir))

    for img in tqdm(imgs):
        pred_name = img.split('.')[0]+'.png'
        pred_img = cv2.imread(os.path.join(args.pred_dir, pred_name))
        pred_img = cv2.resize(pred_img, (960, 480),
                              interpolation=cv2.INTER_AREA)
        gt_img = cv2.imread(os.path.join(args.gt_dir, img))
        gt_img = cv2.resize(gt_img, (960, 480), interpolation=cv2.INTER_AREA)
        global_psnr += psnr(gt_img, pred_img)
        global_ssim += ssim(gt_img, pred_img,multichannel=True)

    print("PSNR ", global_psnr/len(imgs))
    print("SSIM ", global_ssim/len(imgs))
