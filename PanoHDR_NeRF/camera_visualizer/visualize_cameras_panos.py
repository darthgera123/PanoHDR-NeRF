import numpy as np
import argparse
import os
from mayavi import mlab
from tvtk.tools import visual
from traits.api import HasTraits, Instance, Button, on_trait_change
import json
import random


def read_file(address):
    f = open(address, "r")
    params = f.read().split(" ")
    c2w = np.array(params).reshape(4, 4)
    key = c2w[0, 3] + ',' + c2w[1, 3] + ',' + c2w[2, 3]
    return c2w, key


def load_data(address, mode):
    pose_folder = os.path.join(address, 'pose')
    from os import listdir
    from os.path import isfile, join
    poseFiles = [f for f in listdir(pose_folder) if isfile(join(pose_folder, f))]
    extrinsics = []
    dic = {}
    for poseFile in poseFiles:
        c2w, key = read_file(os.path.join(pose_folder, poseFile))
        dic[key] = os.path.join(mode, 'rgb', poseFile.split('.')[0])
        extrinsics.append(c2w)
    return extrinsics, dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the input to Pano-NeRF')
    parser.add_argument('--input_type', type=str)
    parser.add_argument('--input_dir', help='path to input data')
    parser.add_argument("--show_camera_path", action='store_true', help='Show the interpolation camera poses')
    args = parser.parse_args()

    if args.input_type == 'nerf':
        base_dir = args.input_dir

        test_poses, test_dic = load_data(os.path.join(base_dir, 'test'), 'test')
        val_poses, val_dic = load_data(os.path.join(base_dir, 'validation'), 'validation')
        train_poses, train_dic = load_data(os.path.join(base_dir, 'train'), 'train')
        camera_path_is_there = False
        if os.path.exists(os.path.join(base_dir, 'camera_path')) and args.show_camera_path:
            camera_path_is_there = True
            path_poses, path_dic = load_data(os.path.join(base_dir, 'camera_path'), 'camera_path')

        pose_dic = train_dic
        pose_dic.update(val_dic)
        pose_dic.update(test_dic)
        if camera_path_is_there:
            pose_dic.update(path_dic)

        cp_x = []
        cp_y = []
        cp_z = []
        cp_x_u = []
        cp_x_v = []
        cp_x_w = []
        cp_y_u = []
        cp_y_v = []
        cp_y_w = []
        cp_z_u = []
        cp_z_v = []
        cp_z_w = []

        # Create a figure
        f = mlab.figure()
        # # Tell visual to use this as the viewer.
        visual.set_viewer(f)

        test_x = []
        test_y = []
        test_z = []
        for c2w in test_poses:
            extrinsics = np.array([[float(c2w[0, 0]), float(c2w[0, 1]), float(c2w[0, 2]), float(c2w[0, 3])],
                                   [float(c2w[1, 0]), float(c2w[1, 1]), float(c2w[1, 2]), float(c2w[1, 3])],
                                   [float(c2w[2, 0]), float(c2w[2, 1]), float(c2w[2, 2]), float(c2w[2, 3])],
                                   [float(c2w[3, 0]), float(c2w[3, 1]), float(c2w[3, 2]), float(c2w[3, 3])]])
            ext_inv = extrinsics
            test_x.append(ext_inv[0, 3])
            test_y.append(ext_inv[1, 3])
            test_z.append(ext_inv[2, 3])
            cp_x.append(ext_inv[0, 3])
            cp_y.append(ext_inv[1, 3])
            cp_z.append(ext_inv[2, 3])
            vec_x = np.matmul(ext_inv[0:3, 0:3], np.array([1, 0, 0]))
            vec_y = np.matmul(ext_inv[0:3, 0:3], np.array([0, 1, 0]))
            vec_z = np.matmul(ext_inv[0:3, 0:3], np.array([0, 0, 1]))
            cp_x_u.append(vec_x[0])
            cp_x_v.append(vec_x[1])
            cp_x_w.append(vec_x[2])
            cp_y_u.append(vec_y[0])
            cp_y_v.append(vec_y[1])
            cp_y_w.append(vec_y[2])
            cp_z_u.append(vec_z[0])
            cp_z_v.append(vec_z[1])
            cp_z_w.append(vec_z[2])

        train_x = []
        train_y = []
        train_z = []
        for c2w in train_poses:
            extrinsics = np.array([[float(c2w[0, 0]), float(c2w[0, 1]), float(c2w[0, 2]), float(c2w[0, 3])],
                                   [float(c2w[1, 0]), float(c2w[1, 1]), float(c2w[1, 2]), float(c2w[1, 3])],
                                   [float(c2w[2, 0]), float(c2w[2, 1]), float(c2w[2, 2]), float(c2w[2, 3])],
                                   [float(c2w[3, 0]), float(c2w[3, 1]), float(c2w[3, 2]), float(c2w[3, 3])]])
            ext_inv = extrinsics
            train_x.append(ext_inv[0, 3])
            train_y.append(ext_inv[1, 3])
            train_z.append(ext_inv[2, 3])
            cp_x.append(ext_inv[0, 3])
            cp_y.append(ext_inv[1, 3])
            cp_z.append(ext_inv[2, 3])
            vec_x = np.matmul(ext_inv[0:3, 0:3], np.array([1, 0, 0]))
            vec_y = np.matmul(ext_inv[0:3, 0:3], np.array([0, 1, 0]))
            vec_z = np.matmul(ext_inv[0:3, 0:3], np.array([0, 0, 1]))
            cp_x_u.append(vec_x[0])
            cp_x_v.append(vec_x[1])
            cp_x_w.append(vec_x[2])
            cp_y_u.append(vec_y[0])
            cp_y_v.append(vec_y[1])
            cp_y_w.append(vec_y[2])
            cp_z_u.append(vec_z[0])
            cp_z_v.append(vec_z[1])
            cp_z_w.append(vec_z[2])

        val_x = []
        val_y = []
        val_z = []
        for c2w in val_poses:
            extrinsics = np.array([[float(c2w[0, 0]), float(c2w[0, 1]), float(c2w[0, 2]), float(c2w[0, 3])],
                                   [float(c2w[1, 0]), float(c2w[1, 1]), float(c2w[1, 2]), float(c2w[1, 3])],
                                   [float(c2w[2, 0]), float(c2w[2, 1]), float(c2w[2, 2]), float(c2w[2, 3])],
                                   [float(c2w[3, 0]), float(c2w[3, 1]), float(c2w[3, 2]), float(c2w[3, 3])]])
            ext_inv = extrinsics
            val_x.append(ext_inv[0, 3])
            val_y.append(ext_inv[1, 3])
            val_z.append(ext_inv[2, 3])
            cp_x.append(ext_inv[0, 3])
            cp_y.append(ext_inv[1, 3])
            cp_z.append(ext_inv[2, 3])
            vec_x = np.matmul(ext_inv[0:3, 0:3], np.array([1, 0, 0]))
            vec_y = np.matmul(ext_inv[0:3, 0:3], np.array([0, 1, 0]))
            vec_z = np.matmul(ext_inv[0:3, 0:3], np.array([0, 0, 1]))
            cp_x_u.append(vec_x[0])
            cp_x_v.append(vec_x[1])
            cp_x_w.append(vec_x[2])
            cp_y_u.append(vec_y[0])
            cp_y_v.append(vec_y[1])
            cp_y_w.append(vec_y[2])
            cp_z_u.append(vec_z[0])
            cp_z_v.append(vec_z[1])
            cp_z_w.append(vec_z[2])

        if camera_path_is_there:
            path_x = []
            path_y = []
            path_z = []
            for c2w in path_poses:
                extrinsics = np.array([[float(c2w[0, 0]), float(c2w[0, 1]), float(c2w[0, 2]), float(c2w[0, 3])],
                                       [float(c2w[1, 0]), float(c2w[1, 1]), float(c2w[1, 2]), float(c2w[1, 3])],
                                       [float(c2w[2, 0]), float(c2w[2, 1]), float(c2w[2, 2]), float(c2w[2, 3])],
                                       [float(c2w[3, 0]), float(c2w[3, 1]), float(c2w[3, 2]), float(c2w[3, 3])]])
                ext_inv = extrinsics
                path_x.append(ext_inv[0, 3])
                path_y.append(ext_inv[1, 3])
                path_z.append(ext_inv[2, 3])
                cp_x.append(ext_inv[0, 3])
                cp_y.append(ext_inv[1, 3])
                cp_z.append(ext_inv[2, 3])
                vec_x = np.matmul(ext_inv[0:3, 0:3], np.array([1, 0, 0]))
                vec_y = np.matmul(ext_inv[0:3, 0:3], np.array([0, 1, 0]))
                vec_z = np.matmul(ext_inv[0:3, 0:3], np.array([0, 0, 1]))
                cp_x_u.append(vec_x[0])
                cp_x_v.append(vec_x[1])
                cp_x_w.append(vec_x[2])
                cp_y_u.append(vec_y[0])
                cp_y_v.append(vec_y[1])
                cp_y_w.append(vec_y[2])
                cp_z_u.append(vec_z[0])
                cp_z_v.append(vec_z[1])
                cp_z_w.append(vec_z[2])

        all_x = train_x + val_x + test_x
        all_y = train_y + val_y + test_y
        all_z = train_z + val_z + test_z
        if camera_path_is_there:
            all_x = all_x + path_x
            all_y = all_y + path_y
            all_z = all_z + path_z
        all_mesh = mlab.points3d(all_x, all_y, all_z, figure=f, color=(1, 1, 1), scale_factor=0.01)
        all_mesh_array = all_mesh.glyph.glyph_source.glyph_source.output.points.to_array()

        mlab.points3d(train_x, train_y, train_z, figure=f, color=(1, 1, 0), scale_factor=0.01)
        mlab.points3d(test_x, test_y, test_z, figure=f, color=(1, 0, 1), scale_factor=0.01)
        mlab.points3d(val_x, val_y, val_z, figure=f, color=(0, 1, 1), scale_factor=0.01)
        if camera_path_is_there:
            mlab.points3d(path_x, path_y, path_z, figure=f, color=(0, 0, 0), scale_factor=0.01)

        mlab.quiver3d(cp_x, cp_y, cp_z, cp_x_u, cp_x_v, cp_x_w, color=(1, 0, 0), line_width=1, scale_factor=0.05)
        mlab.quiver3d(cp_x, cp_y, cp_z, cp_y_u, cp_y_v, cp_y_w, color=(0, 1, 0), line_width=1, scale_factor=0.05)
        mlab.quiver3d(cp_x, cp_y, cp_z, cp_z_u, cp_z_v, cp_z_w, color=(0, 0, 1), line_width=1, scale_factor=0.05)

        for index in range(len(pose_dic)):
            fullname = list(pose_dic.values())[index]
            imgname = fullname.split('\\')[2]
            mlab.text3d(all_x[index], all_y[index], all_z[index], imgname, color=(0, 0, 0), scale=0.005, figure=f)


        def picker_callback(picker_obj):
            picked = picker_obj.actors
            if all_mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
                # m.mlab_source.points is the points array underlying the vtk
                # dataset. GetPointId return the index in this array.
                point_id = int(picker_obj.point_id/all_mesh_array.shape[0])
                point_key = str(all_x[point_id]) + ',' + str(all_y[point_id]) + ',' + str(all_z[point_id])
                image_name = pose_dic[point_key] + '.jpeg'
                print(image_name)
                print(point_key)
                import imageio
                img = imageio.imread(os.path.join(base_dir, image_name))
                import pylab as pl
                pl.imshow(img)
                pl.axis('off')
                pl.show()

        f.on_mouse_pick(picker_callback)
        mlab.show()

    elif args.input_type == 'json':

        print("Input: JSON")
        # Create a figure
        f = mlab.figure()
        # # Tell visual to use this as the viewer.
        visual.set_viewer(f)

        with open(args.input_dir) as file:
            print("Loading JSON file...")
            data = json.load(file)
        # print(json.dumps(data, indent = 4, sort_keys=True))
        print("Loading JSON file done!")
        for l in range(len(data)):
            positions = []
            for s in data[l]['shots']:
                rvec = np.array(tuple(map(float, data[l]['shots'][s]['rotation'])))
                tvec = np.array(tuple(map(float, data[l]['shots'][s]['translation'])))
                positions.append(tvec)

            test_x = [float(f[0]) for f in positions]
            test_y = [float(f[1]) for f in positions]
            test_z = [float(f[2]) for f in positions]
            print("cluster id: " + str(l) )
            print("number of points: " + str(len(test_x)))
            mlab.points3d(test_x, test_y, test_z, figure=f, color=(random.random(), random.random(), random.random()), scale_factor=5)

        mlab.show()

    elif args.input_type == 'comparison':
        vector_visualizing_threshold = 5
        base_dir = args.input_dir

        col_test_poses, col_test_dic = load_data(os.path.join(base_dir, 'colmap/test'), 'test')
        col_val_poses, col_val_dic = load_data(os.path.join(base_dir, 'colmap/validation'), 'validation')
        col_train_poses, col_train_dic = load_data(os.path.join(base_dir, 'colmap/train'), 'train')
        if os.path.exists(os.path.join(base_dir, 'colmap/camera_path')):
            col_path_poses, col_path_dic = load_data(os.path.join(base_dir, 'colmap/camera_path'), 'camera_path')

        openSFM_test_poses, openSFM_test_dic = load_data(os.path.join(base_dir, 'openSFM/test'), 'test')
        openSFM_val_poses, openSFM_val_dic = load_data(os.path.join(base_dir, 'openSFM/validation'), 'validation')
        openSFM_train_poses, openSFM_train_dic = load_data(os.path.join(base_dir, 'openSFM/train'), 'train')
        if os.path.exists(os.path.join(base_dir, 'openSFM/camera_path')):
            openSFM_path_poses, openSFM_path_dic = load_data(os.path.join(base_dir, 'openSFM/camera_path'), 'camera_path')

        # Create a figure
        f = mlab.figure()
        # # Tell visual to use this as the viewer.
        visual.set_viewer(f)

        col_x_u = []
        col_x_v = []
        col_x_w = []
        col_y_u = []
        col_y_v = []
        col_y_w = []
        col_z_u = []
        col_z_v = []
        col_z_w = []

        col_all = col_test_poses + col_val_poses + col_train_poses
        count = 0
        for c2w in col_all:
            if count % 1 == 0:
                rot = np.array([[float(c2w[0, 0]), float(c2w[0, 1]), float(c2w[0, 2])],
                                [float(c2w[1, 0]), float(c2w[1, 1]), float(c2w[1, 2])],
                                [float(c2w[2, 0]), float(c2w[2, 1]), float(c2w[2, 2])]])
                vec_x = np.matmul(rot, np.array([1, 0, 0]))
                vec_y = np.matmul(rot, np.array([0, 1, 0]))
                vec_z = np.matmul(rot, np.array([0, 0, 1]))
                col_x_u.append(vec_x[0])
                col_x_v.append(vec_x[1])
                col_x_w.append(vec_x[2])
                col_y_u.append(vec_y[0])
                col_y_v.append(vec_y[1])
                col_y_w.append(vec_y[2])
                col_z_u.append(vec_z[0])
                col_z_v.append(vec_z[1])
                col_z_w.append(vec_z[2])
            count = count + 1

        col_test_x = [float(f[0, 3]) for f in col_test_poses]
        col_test_y = [float(f[1, 3]) for f in col_test_poses]
        col_test_z = [float(f[2, 3]) for f in col_test_poses]

        col_train_x = [float(f[0, 3]) for f in col_train_poses]
        col_train_y = [float(f[1, 3]) for f in col_train_poses]
        col_train_z = [float(f[2, 3]) for f in col_train_poses]

        col_val_x = [float(f[0, 3]) for f in col_val_poses]
        col_val_y = [float(f[1, 3]) for f in col_val_poses]
        col_val_z = [float(f[2, 3]) for f in col_val_poses]

        col_all_x = col_train_x + col_test_x + col_val_x
        col_all_y = col_train_y + col_test_y + col_val_y
        col_all_z = col_train_z + col_test_z + col_val_z
        mlab.points3d(col_all_x, col_all_y, col_all_z, figure=f, color=(1, 0, 0), scale_factor=0.01)
        mlab.quiver3d(col_all_x, col_all_y, col_all_z, col_x_u, col_x_v, col_x_w, color=(1, 0, 0), line_width=1, scale_factor=0.1)
        mlab.quiver3d(col_all_x, col_all_y, col_all_z, col_y_u, col_y_v, col_y_w, color=(0, 1, 0), line_width=1, scale_factor=0.1)
        mlab.quiver3d(col_all_x, col_all_y, col_all_z, col_z_u, col_z_v, col_z_w, color=(0, 0, 1), line_width=1, scale_factor=0.1)

        openSFM_x_u = []
        openSFM_x_v = []
        openSFM_x_w = []
        openSFM_y_u = []
        openSFM_y_v = []
        openSFM_y_w = []
        openSFM_z_u = []
        openSFM_z_v = []
        openSFM_z_w = []

        openSFM_all = openSFM_test_poses + openSFM_val_poses + openSFM_train_poses
        count = 0
        for c2w in openSFM_all:
            if count % 1 == 0:
                rot = np.array([[float(c2w[0, 0]), float(c2w[0, 1]), float(c2w[0, 2])],
                                [float(c2w[1, 0]), float(c2w[1, 1]), float(c2w[1, 2])],
                                [float(c2w[2, 0]), float(c2w[2, 1]), float(c2w[2, 2])]])
                from scipy.spatial.transform import Rotation as R
                r = R.from_euler('x', -90, degrees=True).as_matrix()
                vec_x = np.matmul(rot, np.array([1, 0, 0]))
                vec_x = np.matmul(r, vec_x)
                vec_y = np.matmul(rot, np.array([0, 1, 0]))
                vec_y = np.matmul(r, vec_y)
                vec_z = np.matmul(rot, np.array([0, 0, 1]))
                vec_z = np.matmul(r, vec_z)
                openSFM_x_u.append(vec_x[0])
                openSFM_x_v.append(vec_x[1])
                openSFM_x_w.append(vec_x[2])
                openSFM_y_u.append(vec_y[0])
                openSFM_y_v.append(vec_y[1])
                openSFM_y_w.append(vec_y[2])
                openSFM_z_u.append(vec_z[0])
                openSFM_z_v.append(vec_z[1])
                openSFM_z_w.append(vec_z[2])
            count = count + 1
        openSFM_test_x = [float(f[0, 3]) for f in openSFM_test_poses]
        openSFM_test_y = [float(f[1, 3]) for f in openSFM_test_poses]
        openSFM_test_z = [float(f[2, 3]) for f in openSFM_test_poses]

        openSFM_train_x = [float(f[0, 3]) for f in openSFM_train_poses]
        openSFM_train_y = [float(f[1, 3]) for f in openSFM_train_poses]
        openSFM_train_z = [float(f[2, 3]) for f in openSFM_train_poses]

        openSFM_val_x = [float(f[0, 3]) for f in openSFM_val_poses]
        openSFM_val_y = [float(f[1, 3]) for f in openSFM_val_poses]
        openSFM_val_z = [float(f[2, 3]) for f in openSFM_val_poses]

        openSFM_all_x = openSFM_train_x + openSFM_test_x + openSFM_val_x
        openSFM_all_y = openSFM_train_y + openSFM_test_y + openSFM_val_y
        openSFM_all_z = openSFM_train_z + openSFM_test_z + openSFM_val_z
        mlab.points3d(openSFM_all_x, openSFM_all_y, openSFM_all_z, figure=f, color=(0, 1, 0), scale_factor=0.01)
        mlab.quiver3d(openSFM_all_x, openSFM_all_y, openSFM_all_z, openSFM_x_u, openSFM_x_v, openSFM_x_w, color=(1, 0, 0),
                      line_width=1, scale_factor=0.1)
        mlab.quiver3d(openSFM_all_x, openSFM_all_y, openSFM_all_z, openSFM_y_u, openSFM_y_v, openSFM_y_w, color=(0, 1, 0),
                      line_width=1, scale_factor=0.1)
        mlab.quiver3d(openSFM_all_x, openSFM_all_y, openSFM_all_z, openSFM_z_u, openSFM_z_v, openSFM_z_w, color=(0, 0, 1),
                      line_width=1, scale_factor=0.1)

        # mlab.points3d(train_x, train_y, train_z, figure=f, color=(1, 1, 0), scale_factor=0.01)
        # mlab.points3d(test_x, test_y, test_z, figure=f, color=(1, 0, 1), scale_factor=0.01)
        # mlab.points3d(val_x, val_y, val_z, figure=f, color=(0, 1, 1), scale_factor=0.01)
        #
        # mlab.points3d(cp_x, cp_y, cp_z, color=(1, 0, 0), scale_factor=0.01)
        # mlab.quiver3d(cp_x, cp_y, cp_z, cp_u, cp_v, cp_w, line_width=1, scale_factor=0.1)


        mlab.show()