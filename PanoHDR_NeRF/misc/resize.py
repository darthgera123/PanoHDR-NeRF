import os, json, cv2, random

import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='', help='directory to data')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--img_width', type=int, default=1920)
    parser.add_argument('--img_height', type=int, default=960)
    args = parser.parse_args()
    out = f'{args.output_dir}'
    if(not os.path.isdir(out)):
        os.makedirs(out)

    imgs = os.listdir(os.path.join(args.input_dir))
    for file in tqdm(imgs):
        
        path = f'{args.input_dir}/{file}'
        img = cv2.imread(path)
        frame = cv2.resize(img, (args.img_width, args.img_height))
        cv2.imwrite(f'{out}/{file}',frame)