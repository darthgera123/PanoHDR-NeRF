import bpy
import logging
import mathutils
import math
import os
import bpy_extras
from mathutils import Matrix, Vector
from os.path import isfile, join
import time


obj_list = ['Sphere', 'Camera']
env_map_w = 512
env_map_h = 256
output_addr =  '/Users/momo/Desktop/temp' # '/gel/usr/mokad6/Desktop/blender_rendering'
# timestr = time.strftime("%Y%m%d-%H%M%S")
scene_name = "Test"
output_folder = join(output_addr, scene_name)
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
opensfm_translate = Vector((0.01277995, -0.00811234, 0.00482027))
opensfm_scale = 0.03525961780914236


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K


def mat_to_str(inp_mat):
    matrix_flatten = []
    for i in range(4):
        for j in range(4):
            matrix_flatten.append(inp_mat[i][j])
    output_str = " ".join([str(elem) for elem in matrix_flatten])
    return output_str
    

def get_RT(obj_name):
    obj_loc = opensfm_scale * (bpy.data.objects[obj_name].matrix_world.translation + opensfm_translate)
    obj_dir = bpy.data.objects[obj_name].matrix_world.to_euler()
#    
    conversion_eul = mathutils.Euler((math.radians(180.0), 0.0, 0.0), 'XYZ')
    conv_mat = conversion_eul.to_matrix()
    
    # source: https://docs.blender.org/api/current/mathutils.html
    mat_rot = obj_dir.to_matrix()
    mat_loc = mathutils.Matrix.Translation(obj_loc)
    if bpy.app.version[0] > 2 or  bpy.app.version[1] > 79:
        mat_rot = conv_mat @ mat_rot
        mat = mat_loc @ mat_rot.to_4x4()
    else:
        mat_rot = conv_mat * mat_rot
        mat = mat_loc * mat_rot.to_4x4()
    return mat

for obj in obj_list:
    dir = os.path.join(output_folder, obj)
    if not os.path.exists(dir):
        os.mkdir(dir)
    pose_dir = os.path.join(dir, 'pose')
    if not os.path.exists(pose_dir):
        os.mkdir(pose_dir)
    intrinsics_dir = os.path.join(dir, 'intrinsics')
    if not os.path.exists(intrinsics_dir):
        os.mkdir(intrinsics_dir)

sce = bpy.context.scene

# Source https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
scene_addr = join(output_folder, "main_scene.blend")
bpy.ops.wm.save_as_mainfile(filepath=scene_addr, copy=True)

# Source: https://blender.stackexchange.com/questions/8387/how-to-get-keyframe-data-from-python
for f in range(sce.frame_start, sce.frame_end+1):
    sce.frame_set(f)
    print("Frame %i" % f)
    file_name = '{:04d}'.format(f) + '.txt'

    for obj in obj_list:
        rt = get_RT(obj)
        rt_str = mat_to_str(rt)
        file_addr = os.path.join(output_folder, obj, 'pose', file_name)
        with open(file_addr, 'w') as f:
            f.write(rt_str)
            
        intrinsics = mathutils.Matrix.Identity(4)
        if obj.startswith('Camera'):
            cam = bpy.data.objects[obj]
            K = get_calibration_matrix_K_from_blender(cam.data)
            for m in range(3):
                for n in range(3):
                    intrinsics[m][n] = K[m][n]
        else:
            intrinsics[0][2] = env_map_w / 2.0
            intrinsics[1][2] = env_map_h / 2.0
        int_str = mat_to_str(intrinsics)
        file_addr = os.path.join(output_folder, obj, 'intrinsics', file_name)
        with open(file_addr, 'w') as f:
            f.write(int_str) 
print("Done")