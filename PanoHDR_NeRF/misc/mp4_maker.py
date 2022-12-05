import cv2
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a GIF from input images')
    parser.add_argument("--input_Addr", type=str, help='input folder')
    # parser.add_argument("--input_pano_Addr", type=str, help='input folder')
    args = parser.parse_args()

    images = []

    from os import listdir
    from os.path import isfile, join

    img_files = [join(args.input_Addr, f) for f in listdir(args.input_Addr) if isfile(join(args.input_Addr, f))]
    # pano_files = [join(args.input_pano_Addr, f) for f in listdir(args.input_pano_Addr) if isfile(join(args.input_pano_Addr, f))]

    for i in  range(len(img_files)):
        img_add = img_files[i]
        # pano_add = pano_files[i]
        print(img_add)
        # print(pano_add)
        img = cv2.imread(img_add)
        height, width, layers = img.shape
        size = (width, height)
        # pano = cv2.imread(pano_add)
        # small_pano = cv2.resize(pano, (0, 0), fx=0.3, fy=0.3)
        # img[0:small_pano.shape[0], 0:small_pano.shape[1]] = small_pano
        images.append(img)
        
    # out = cv2.VideoWriter('saussan_ldr_light.mp4', cv2.VideoWriter_fourcc(*'FMP4'), 10, size)
    out = cv2.VideoWriter('meeting_room_mask.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 15, size)

    for i in range(len(images)):
        out.write(images[i])
    out.release()
