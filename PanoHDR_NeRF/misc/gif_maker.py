from PIL import Image
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a GIF from input images')
    parser.add_argument("--input_Addr", type=str, help='input folder')
    parser.add_argument("--name", type=str, help='input folder')
    args = parser.parse_args()

    images = []

    from os import listdir
    from os.path import isfile, join

    img_files = [join(args.input_Addr, f) for f in listdir(args.input_Addr) if isfile(join(args.input_Addr, f))]

    for img_add in img_files:
        img = Image.open(img_add)
        images.append(img)

    images[0].save(args.name+'.gif', save_all=True, append_images=images[1:], optimize=False, duration=300, loop=0)
