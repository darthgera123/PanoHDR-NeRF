import os
import sys
import numpy as np
import argparse
import json
import random
from shutil import copyfile


def create_the_folder(inp_path, out_path, json_data, idx, name, islinear=False):
    # Create folders for the set
    outDir = os.path.join(out_path, name)
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    rgb_outDir = os.path.join(outDir, 'rgb')
    if not os.path.exists(rgb_outDir):
        os.mkdir(rgb_outDir)

    intrinsics_outDir = os.path.join(outDir, 'intrinsics')
    if not os.path.exists(intrinsics_outDir):
        os.mkdir(intrinsics_outDir)

    pose_outDir = os.path.join(outDir, 'pose')
    if not os.path.exists(pose_outDir):
        os.mkdir(pose_outDir)

    for i in idx:
        key = list(json_data)[i]
        file_name = key.split(".")
        txt_file = file_name[0] + '.txt'
        img_name = file_name[0] + '.exr' if islinear else key
        # Copy the image
        inp_img_dir = os.path.join(inp_path, img_name)
        out_img_dir = os.path.join(rgb_outDir, img_name)
        copyfile(inp_img_dir, out_img_dir)
        # Write the intrinsics
        intr_file = os.path.join(intrinsics_outDir, txt_file)
        line = " ".join([str(elem) for elem in json_data[key]['K']])
        file = open(intr_file, "w")
        file.write(line)
        file.close()
        # Write the intrinsics
        extr_file = os.path.join(pose_outDir, txt_file)
        line = " ".join([str(elem) for elem in json_data[key]['W2C']])
        file = open(extr_file, "w")
        file.write(line)
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read normalized camera parameters from JSON file and produce the '
                                                 'data set')
    parser.add_argument('--input_img_dir', required=False, help='path to input images')
    parser.add_argument("--hasRGB", action='store_true', help='If there are sRGB images as input')
    parser.add_argument('--input_json_file', help='path to input json file')
    parser.add_argument('--output_dir', metavar='PATH', required=False, help='path to output folder')
    parser.add_argument("--hasLinear", action='store_true', help='If there are EXR images as input')
    parser.add_argument('--input_linear_dir', required=False, help='path to input linear images')
    parser.add_argument('--output_linear_dir', required=False, metavar='PATH', help='path to output linear folder')
    args = parser.parse_args()

    with open(args.input_json_file) as f:
        data = json.load(f)
    # print(json.dumps(data, indent = 4, sort_keys=True))

    # Generate the ids for train, test, validation split
    all_ids = list(range(1, len(data)))
    train_ids = random.sample(range(1, len(data)), int(0.8 * len(data)))
    test_val_ids = list(set(all_ids) - set(train_ids))
    test_ids = random.sample(test_val_ids, int(0.1 * len(data)))
    val_ids = list(set(test_val_ids) - set(test_ids))

    # Generate the keys for train, test, validation split
    # all_keys = data.keys()
    # test_keys = ["A.jpg", "C.jpg", "E.jpg"]
    # val_keys = ["B.jpg", "D.jpg", "F.jpg"]
    # train_keys = list(set(all_keys) - set(test_keys) - set(val_keys))

    if args.hasRGB:
        create_the_folder(args.input_img_dir, args.output_dir, data, train_ids, "train")
        create_the_folder(args.input_img_dir, args.output_dir, data, test_ids, "test")
        create_the_folder(args.input_img_dir, args.output_dir, data, val_ids, "validation")

    if args.hasLinear:
        create_the_folder(args.input_linear_dir, args.output_linear_dir, data, train_ids, "train", True)
        create_the_folder(args.input_linear_dir, args.output_linear_dir, data, test_ids, "test", True)
        create_the_folder(args.input_linear_dir, args.output_linear_dir, data, val_ids, "validation", True)
