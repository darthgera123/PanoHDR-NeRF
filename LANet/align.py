import sys
import os
import imageio
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdr_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--exposure', type=float, default=1)
    parser.add_argument('--ldr_dir', type=str, default='')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images = os.listdir(args.hdr_dir)

    for img in tqdm(images):
        name = img.split('.')[0]

        hdr_img = cv2.imread(os.path.join(args.hdr_dir,name+'.hdr'),-1)
        ldr_img = cv2.imread(os.path.join(args.ldr_dir,name+'.jpeg'))
        hdr_img = np.fliplr(hdr_img.reshape(-1, 3)).reshape(hdr_img.shape)
        ldr_img = np.fliplr(ldr_img.reshape(-1, 3)).reshape(ldr_img.shape)
        a = np.ones_like(hdr_img)
        hdr_img = np.roll(hdr_img, int(hdr_img.shape[1]*0.5), axis=1)

        edges = cv2.Canny(ldr_img, 90, 110)
        edges_gt = cv2.Canny((hdr_img*255).astype(np.uint8), 90, 110)
        
        off_range = np.arange(-5, 5)
        score = np.ones(1000)*50000000
        right_offset = 0
        score_min = 0
        for i, p1 in enumerate(off_range):
            im_off = np.roll(edges_gt, p1, axis=1)
            diff_sqr = np.square(
                # im_off[400:-400, 500:-500] - edges[400:-400, 500:-500])
                im_off - edges)
            sdc = np.sum(diff_sqr)
            score[i] = sdc
            if score[i] == score.min():
                right_offset = p1
                score_min = score[i]
        print(right_offset)
        right_offset = 700
        hdr_trans = np.roll(hdr_img, right_offset, axis=1)
        edges_trans = np.roll(edges_gt, right_offset, axis=1)
        # cv2.imwrite(f'{args.output_dir}/{name}.png',cv2.cvtColor(hdr_trans**(1/2.2),cv2.COLOR_BGR2RGB)*255)
        cv2.imwrite(f'{args.output_dir}/{img}',cv2.cvtColor(hdr_trans,cv2.COLOR_BGR2RGB))