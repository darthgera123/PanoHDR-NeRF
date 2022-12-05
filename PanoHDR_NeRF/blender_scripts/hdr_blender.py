'''
Run as blender scene_new.blend -P hdr_blender.py -b
'''

import bpy
from mathutils import Matrix, Vector, Euler
import cycles
import random
from math import radians
import os
from os import listdir
from os.path import isfile, join
import bpy


path = ''

datasets = ['chess_room','old_class_spot','old_class_neon','small_class'\
            ,'stairway','cafeteria']

inp_dirs = [''] 
out_dirs = [''] 
pairs = list(zip(inp_dirs,out_dirs))
for pair in pairs:
    inp,out = pair    
    inp = os.path.join(path,inp)    
    out = os.path.join(path,out)
    os.makedirs(out,exist_ok=True)


    S = bpy.context.scene

    envFiles = [f for f in listdir(inp) if isfile(join(inp, f)) and f.endswith('.hdr')]
    print(envFiles)
    i = 0
    for env_map in envFiles:
        env_file = join(inp, env_map)
        bpy.ops.image.open(filepath=env_file, directory=inp, files=[
                           {"name": env_map, "name": env_map}], show_multiview=False)
        output_name = "render_" + env_map.split('.')[0] +'.exr'
        # print(output_name)
        # bpy.ops.image.save_as(save_as_render=True, copy=True, filepath=join(out_addr, output_name), show_multiview=False, use_multiview=False)

        scn = bpy.context.scene
        scn.render.engine = 'CYCLES'
        scn.world.use_nodes = True
        # bpy.context.space_data.context = 'VIEW_LAYER'
        # bpy.context.space_data.context = 'OUTPUT'
        # bpy.context.scene.render.resolution_x = 960
        # bpy.context.scene.render.resolution_y = 480
        # bpy.context.space_data.context = 'RENDER'
        # bpy.context.scene.cycles.samples = 32
        # bpy.context.scene.cycles.device = 'GPU'
        # select world node tree
        wd = scn.world
        nt = bpy.data.worlds[wd.name].node_tree
        backNode = nt.nodes['Environment Texture']
        backNode.image = bpy.data.images.load(env_file)

        bpy.context.scene.render.filepath = os.path.join(out, output_name)
        print("Output",os.path.join(out, output_name))
        bpy.ops.render.render(write_still=True)
