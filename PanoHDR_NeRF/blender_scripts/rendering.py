import bpy
from mathutils import Matrix, Vector, Euler
import cycles
import random
from math import radians
import os
from os import listdir
from os.path import isfile, join
import bpy
import bpy_extras
import numpy as np
from PIL import Image
import cv2


opensfm_translate = Vector((0.01277995, -0.00811234, 0.00482027))
opensfm_scale = 0.03525961780914236

def get_calibration_matrix_K_from_blender():
    camd = bpy.data.objects['Camera'].data
    
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


def load_process_depth_map(fg_depth_addr, bg_depth_addr):
    # Load both FG and BG depth maps
    img = np.asarray(Image.open(fg_depth_addr).convert('L'))
    bg_img = np.asarray(Image.open(bg_depth_addr).convert('L'))
    # Combine deopth maps
    bg_depth_cut = bg_img.copy()
    bg_depth_cut[bg_depth_cut[:, :] > 0] = 1
    fg_filter = img.copy()
    fg_filter[bg_depth_cut[:, :] == 1] = 255
    # Filter the depth map
#    median_blur = cv2.medianBlur(fg_filter, 11)
#    depth = median_blur.astype(np.float32) / 256
    median_blur = cv2.medianBlur(img, 11)
    depth = median_blur.astype(np.float32) / 256
    
    return depth


def project_depth_and_render(fg_depth_addr, bg_depth_addr, out_addr, output_name):
    
    depth = load_process_depth_map(fg_depth_addr, bg_depth_addr)
    
    vertices = []
    edges = []
    faces = []
    
    c2w = bpy.data.objects['Camera'].matrix_world.copy()
#    c2w.invert()
    
    obj_loc = c2w.translation # / opensfm_scale - opensfm_translate
    obj_dir = c2w.to_euler()
    
    conversion_eul = Euler((radians(180.0), 0.0, 0.0), 'XYZ')
    conv_mat = conversion_eul.to_matrix()
    
    H = depth.shape[0]
    W = depth.shape[1]

    K = get_calibration_matrix_K_from_blender()
#    K.invert()
    
    for v in range(H):
        for u in range(W):
            Z = depth[v][u] / opensfm_scale
            pixel = Vector((u, v, 1))
#            cam_pos = K @ pixel
#            cam_pos = cam_pos / cam_pos[2]
#            world_pos = Z * cam_pos
            X = (u - K[0][2]) * Z / K[0][0]
            Y = (v - K[1][2]) * Z / K[0][0]
            world_pos = Vector((X, Y, Z))
            vec = Vector((world_pos[0], world_pos[1], world_pos[2], 1))
#            vec = c2w @ vec
            mat_rot = conv_mat @ obj_dir.to_matrix()
            mat_loc = Matrix.Translation(obj_loc)
            mat = mat_loc @ mat_rot.to_4x4()
            vec = mat @ vec
            vec = vec / vec[3]
            vertices.append([vec[0], vec[1], vec[2]]) 
            
    for v in range(H-1):
        for u in range(W-1): 
            up_left = v * W + u
            up_right = v * W + (u + 1)
            down_left = (v + 1) * W + u
            down_right = (v + 1) * W + (u + 1)
            faces.append([up_left, up_right, down_left])
            faces.append([up_right, down_left, down_right])
    
    mymesh = bpy.data.meshes.new("shadow_catcher_mesh")
    mymesh.from_pydata(vertices, edges, faces)
    
    mymat = bpy.data.materials.new("Plane_mat")
    mymat.diffuse_color = [1.0, 0.0, 1.0, 1]
    mymesh.materials.append(mymat)
    
    myobject = bpy.data.objects.new("shadow_catcher", mymesh)
    myobject.location = [0,0,0]
    
    bpy.context.collection.objects.link(myobject)
    
    # Set plane to be a shadow catcher
    myobject.is_shadow_catcher = True
    myobject.visible_glossy = False
    myobject.visible_diffuse = True # False
    myobject.visible_transmission = False
    myobject.visible_volume_scatter = False
    
    bpy.context.scene.render.filepath = os.path.join(out_addr, output_name)
    bpy.ops.render.render(write_still = True)
    
    bpy.data.objects['shadow_catcher'].select_set(True)
    print("Get test")
    print(bpy.context.scene.objects["shadow_catcher"].select_get())
#    if bpy.context.scene.objects["shadow_catcher"].select_get():
#        bpy.ops.object.delete()
    


base_addr = "/Users/momo/Desktop/PhD/HDR_PanoNeRF/blender_scenes/straight_line_camera_tilted_10f"
nerf_addr = join(base_addr, "nerf_output")
env_addr = join(nerf_addr, "render_Sphere_500000/exr")
cam_addr = join(nerf_addr, "render_Camera_500000")
background_addr = join(cam_addr, "exr")
fg_depth_addr = join(cam_addr, "fg_depth")
bg_depth_addr = join(cam_addr, "bg_depth")
out_addr = join(base_addr, "rendering")
if not os.path.exists(out_addr):
    os.mkdir(out_addr)

S = bpy.context.scene
requested_frames = [] # [31, 33, 44, 56, 57, 67, 68, 69, 129, 190] # Set the frame ids, empty for all
if len(requested_frames) == 0:
    envFiles = [f for f in listdir(env_addr) if isfile(join(env_addr, f)) and f.endswith('.exr')]
    envFiles.sort()
    requested_frames = range(len(envFiles))
else:
    envFiles = []
    for f in requested_frames:
        file_name = '{:06d}'.format(f) + '.exr'
        print(file_name)
        if isfile(join(env_addr, file_name)):
            envFiles.append(file_name)
        

print(envFiles)

cam = bpy.context.scene.camera
cam.data.show_background_images = True
bg = cam.data.background_images.new()

sce = bpy.context.scene
    
for f, env_map in enumerate(envFiles):
    sce.frame_set(requested_frames[f])
    env_file = join(env_addr, env_map)
#    bpy.ops.image.open(filepath=env_file, directory=env_addr, files=[{"name":env_map, "name":env_map}], show_multiview=False)
    output_name = "render_" + env_map.split('.')[0]
    print(output_name)
    # bpy.ops.image.save_as(save_as_render=True, copy=True, filepath=join(out_addr, output_name), show_multiview=False, use_multiview=False)
    
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.world.use_nodes = True

    #select world node tree
    wd = scn.world
    nt = bpy.data.worlds[wd.name].node_tree
    backNode = nt.nodes['Environment Texture']
    backNode.image = bpy.data.images.load(env_file)
    
    bgi_file = join(background_addr, env_map)
    print(bgi_file)
#    img = bpy.data.images.load(bgi_file)
#    bg.image = img
    bg_node = bpy.data.scenes["Scene"].node_tree.nodes["Image"]
    bg_node.image = bpy.data.images.load(bgi_file)
    
    
    depth_name = env_map.split('.')[0] + '.png'
    fg_depth_file = join(fg_depth_addr, depth_name)
    bg_depth_file = join(bg_depth_addr, depth_name)
    project_depth_and_render(fg_depth_file, bg_depth_file, out_addr, output_name)

print("Done!")