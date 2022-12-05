import numpy as np
import argparse
from normalize_cam_dict import normalize_cam_dict
import json


def read_json_file(path, w, h, isPerspective):

    with open(path) as f:
        data = json.load(f)
    # print(json.dumps(data, indent = 4, sort_keys=True))
    camera_dict = {}
    camera_dict_file = './kai_cameras.json'

    l = 0
    for s in data[l]['shots']:

        rvec = np.array(tuple(map(float, data[l]['shots'][s]['rotation'])))
        tvec = np.array(tuple(map(float, data[l]['shots'][s]['translation'])))
        img_size = [w, h]

        camera_dict[s] = {}
        camera_dict[s]['img_size'] = img_size

        K = np.eye(4)
        if isPerspective:
            K[0, 0] = float(data[l]['cameras']['v2 unknown unknown -1 -1 perspective 0']['focal']) * w
            K[1, 1] = float(data[l]['cameras']['v2 unknown unknown -1 -1 perspective 0']['focal']) * w
        K[0, 2] = 0.5 * w
        K[1, 2] = 0.5 * h
        camera_dict[s]['K'] = list(K.flatten())

        from scipy.spatial.transform import Rotation as R
        rot = R.from_rotvec(rvec).as_matrix()
        W2C = np.eye(4)
        W2C[:3, :3] = rot
        W2C[:3, 3] = np.array(tvec)
        camera_dict[s]['W2C'] = list(W2C.flatten())

    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

    translate, scale = normalize_cam_dict('./kai_cameras.json', './kai_cameras_normalized.json')

    print("translate: ", translate)
    print("scale: ", scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and write COLMAP binary and text models')
    parser.add_argument('--input_model', help='path to input model folder')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument("--isPerspective", action='store_true')
    args = parser.parse_args()
    read_json_file(args.input_model, args.width, args.height, args.isPerspective)
