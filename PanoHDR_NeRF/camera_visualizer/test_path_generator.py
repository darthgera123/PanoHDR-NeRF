from operator import le
from mayavi.tools.camera import view
import numpy as np
import argparse
import os
from mayavi import mlab
from tvtk.tools import visual
from traits.api import HasTraits, Instance, Button, on_trait_change
import json
import random
from shutil import copyfile
from scipy import interpolate
from tvtk.api import tvtk


def read_pcd(file_path1, fig):
    # if argument pt_cloud is specified, reads a .ply file to render a point cloud of the scene
    from plyfile import PlyData
    plydata = PlyData.read(open(file_path1, 'rb'))

    scaling_factor = 20 # factor used so that the pcd coordinates correlate with the camera positions
    x = plydata['vertex'].data['x']/scaling_factor
    y = plydata['vertex'].data['y']/scaling_factor
    z = plydata['vertex'].data['z']/scaling_factor
    r = np.array(plydata['vertex'].data['red'])
    g = np.array(plydata['vertex'].data['green'])
    b = np.array(plydata['vertex'].data['blue'])
    colors = np.array([r,g,b])
    colors = np.array([r,g,b]).transpose()
    
    pts = mlab.points3d(x, y, z,
                        #col,  # Values used for Color
                        mode="point",
                        #colormap='coolwarm',  # 'bone', 'copper', 'gnuplot'
                        #color=(1, 1, 1),   # Use a fixed (r,g,b) instead
                        figure=fig,
                        scale_factor=0.01
                        )

    # hack to assign a color to each point
    sc = tvtk.UnsignedCharArray()
    sc.from_array(colors[::])
    pts.mlab_source.dataset.point_data.scalars = sc
    pts.mlab_source.dataset.modified()
    print("pcd done")

def read_file(address):
    f = open(address, "r")
    params = f.read().split(" ")
    c2w = np.array(params).reshape(4, 4)
    key = c2w[0, 3] + ',' + c2w[1, 3] + ',' + c2w[2, 3]
    return c2w, key


def load_data(address, mode):
    rgb_folder = os.path.join(address, 'rgb')
    pose_folder = os.path.join(address, 'pose')
    from os import listdir
    from os.path import isfile, join
    poseFiles = [f for f in listdir(pose_folder) if isfile(join(pose_folder, f))]
    extrinsics = []
    dic = {}
    for poseFile in poseFiles:
        c2w, key = read_file(os.path.join(pose_folder, poseFile))
        dic[key] = os.path.join(mode, 'pose', poseFile.split('.')[0])
        extrinsics.append(c2w)
    return extrinsics, dic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize the input to Pano-NeRF')
    parser.add_argument('--input_dir', help='path to input data')
    parser.add_argument('--output_dir', help='path to input data')
    parser.add_argument('--n', '--num', type=int, default=100, help='number of intermediate points to generate')
    parser.add_argument('--p', '--points', type=int, default=2, help='number of points to select')
    parser.add_argument('--d', '--degree', type=int, default=3, help='manually change the degree of the interpolation (max is 5)')
    parser.add_argument('--pt_cloud', default=None, help='path to point cloud file')

    args = parser.parse_args()

    base_dir = args.input_dir

    test_poses, test_dic = load_data(os.path.join(base_dir, 'test'), 'test')
    val_poses, val_dic = load_data(os.path.join(base_dir, 'validation'), 'validation')
    train_poses, train_dic = load_data(os.path.join(base_dir, 'train'), 'train')

    pose_dic = train_dic
    pose_dic.update(val_dic)
    pose_dic.update(test_dic)


    # Create a figure
    f = mlab.figure()
    # # Tell visual to use this as the viewer.
    visual.set_viewer(f)

    if args.pt_cloud is not None:
        read_pcd(str(args.pt_cloud), f)

    test_x = [float(f[0,3]) for f in test_poses]
    test_y = [float(f[1,3]) for f in test_poses]
    test_z = [float(f[2,3]) for f in test_poses]

    train_x = [float(f[0,3]) for f in train_poses]
    train_y = [float(f[1,3]) for f in train_poses]
    train_z = [float(f[2,3]) for f in train_poses]

    val_x = [float(f[0,3]) for f in val_poses]
    val_y = [float(f[1,3]) for f in val_poses]
    val_z = [float(f[2,3]) for f in val_poses]

    all_x = train_x + test_x + val_x
    all_y = train_y + test_y + val_y
    all_z = train_z + test_z + val_z

    all_mesh = mlab.points3d(all_x, all_y, all_z, figure=f, color=(1, 1, 0), scale_factor=0.03)
    all_mesh_array = all_mesh.glyph.glyph_source.glyph_source.output.points.to_array()
    nb_paths = []
    selected_points = []

    def picker_callback(picker_obj):
        view = mlab.view()
        cam,foc = mlab.move()

        picked = picker_obj.actors
        if all_mesh.actor.actor._vtk_obj in [o._vtk_obj for o in picked]:
            # m.mlab_source.points is the points array underlying the vtk
            # dataset. GetPointId return the index in this array.
            point_id = int(picker_obj.point_id/all_mesh_array.shape[0])
            point_key = str(all_x[point_id]) + ',' + str(all_y[point_id]) + ',' + str(all_z[point_id])
            pose_file_name = pose_dic[point_key] + '.txt'
            mlab.points3d(all_x[point_id], all_y[point_id], all_z[point_id], figure=f, color=(0, 1, 1), scale_factor=0.03)
            c2w, key = read_file(os.path.join(base_dir, pose_file_name))
            selected_points.append(c2w)

            if len(selected_points) == args.p:
                # Do the calculation
                selected_x = [float(f[0,3]) for f in selected_points]
                selected_y = [float(f[1,3]) for f in selected_points]
                selected_z = [float(f[2,3]) for f in selected_points]
                mlab.points3d(selected_x, selected_y, selected_z, figure=f, color=(1, 1, 1), scale_factor=0.02)
                interp_degree = args.d
                if args.p == 2:
                    interp_degree = 1
                elif args.p == 3:
                    interp_degree = 2

                tck, u = interpolate.splprep([selected_x, selected_y, selected_z], k=interp_degree)
                if np.array_equal(selected_points[0], selected_points[-1]):
                    tck, u = interpolate.splprep([selected_x, selected_y, selected_z], k=interp_degree, s=0, per=True)
                x_fine, y_fine, z_fine = interpolate.splev(np.linspace(0,1,args.n), tck)
                mlab.points3d(x_fine, y_fine, z_fine, figure=f, color=(0, 0, 0), scale_factor=0.01)

                output_dir_name = args.output_dir
                num = 0
                if len(nb_paths) != 0:
                    num = nb_paths[-1] +1
                nb_paths.append(num)
                output_dir_name = args.output_dir + '_{:04d}'.format(num)
                if not os.path.exists(output_dir_name):
                    os.mkdir(output_dir_name)
                intrinsics_outDir = os.path.join(output_dir_name, 'intrinsics')
                if not os.path.exists(intrinsics_outDir):
                    os.mkdir(intrinsics_outDir)

                pose_outDir = os.path.join(output_dir_name, 'pose')
                if not os.path.exists(pose_outDir):
                    os.mkdir(pose_outDir)

                # the rotation is fixed
                rot = selected_points[0][0:3,0:3]
                for i in range(0, args.n):
                        inter_mat = np.eye(4)
                        inter_mat[0:3, 0:3] = rot
                        inter_mat[0, 3] = str(x_fine[i])
                        inter_mat[1, 3] = str(y_fine[i])
                        inter_mat[2, 3] = str(z_fine[i])
                        line = " ".join([str(elem) for elem in inter_mat.flatten()])
                        inter_pose_file = os.path.join(pose_outDir, '{:04d}'.format(i) + '.txt')
                        file = open(inter_pose_file, "w")
                        file.write(line)
                        file.close()
                        inter_intr_file = os.path.join(intrinsics_outDir, '{:04d}'.format(i) + '.txt')
                        copyfile(os.path.join(base_dir, pose_file_name.replace('pose', 'intrinsics')), inter_intr_file)

                print("poses and intrinsics exported to {}".format(output_dir_name))
                selected_points.clear()
        
        cam_after,foc = mlab.move()
        delta_x = (cam_after[0]-cam[0]) * -1
        delta_y = (cam_after[0]-cam[0]) * -1
        delta_z = (cam_after[0]-cam[0]) * -1
        mlab.move(delta_x, delta_y, delta_z)
        mlab.view(*view, reset_roll=False)
    
    picker = f.on_mouse_pick(picker_callback)
    picker.tolerance = 0.001
    mlab.show()
