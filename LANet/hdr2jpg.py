import sys
import os
import imageio
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import numpy as np

from skimage.transform import resize


def convert_exr_to_jpg(exr_file, jpg_file, exposure=1):
    if not os.path.isfile(exr_file):
        return False

    filename, extension = os.path.splitext(exr_file)
    if not extension.lower().endswith('.exr'):
        return False

    # imageio.plugins.freeimage.download() #DOWNLOAD IT
    image = imageio.imread(exr_file)
    # print(image.dtype)

    # remove alpha channel for jpg conversion
    image = image[:, :, :3]

    data = 65535 * image
    data[data > 65535] = 65535

    data = data * (exposure**(1/2.2))
    rgb_image = data.astype('uint16')
    rgb_image = (rgb_image-np.min(rgb_image) /
                 (np.max(rgb_image)-np.min(rgb_image)))*255
    # rgb_image  = rgb_image.clip(0,255)
    # print(rgb_image.dtype)
    #rgb_image = imageio.core.image_as_uint(rgb_image, bitdepth=16)
    print(np.median(rgb_image))
    print("Hello")
    imageio.imwrite(jpg_file, rgb_image, format='jpeg')

    return True


def convert(exr, jpg, exposure=1, ext='hdr'):
    # print(ext)
    if ext == 'hdr' or ext == 'exr':
        # print("Hello")
        im = cv2.imread(exr, -1)
        im = resize(im, (256,512), anti_aliasing=True)
        im = im*exposure

        im_gamma_correct = np.clip(np.power(im+1e-8, 0.45), 0, 1)
        im_fixed = np.uint8(im_gamma_correct*255)
    else:
        im = cv2.imread(exr)
        im = im*exposure
        im_fixed = np.clip(im, 0, 255)

    cv2.imwrite(jpg, im_fixed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdr_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--exposure', type=float, default=1)
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hdrs = sorted(os.listdir(args.hdr_dir))
    print(hdrs)
    for img in tqdm(hdrs):
        name = img.split('.')[0]
        exr = os.path.join(args.hdr_dir, img)
        # jpg = os.path.join(args.output_dir,name+f'_{args.dataset}_{str(args.exposure)}.jpeg')
        jpg = os.path.join(args.output_dir, name+f'.jpeg')
        # convert_exr_to_jpg(exr, jpg,args.exposure)
        convert(exr, jpg, args.exposure)
