import os
import subprocess
from extract_sfm import extract_all_to_dir
from normalize_cam_dict import normalize_cam_dict

#########################################################################
# Note: configure the colmap_bin to the colmap executable on your machine
#########################################################################

def bash_run(cmd):
    colmap_bin = '/home/zhangka2/code/colmap/build/__install__/bin/colmap'
    cmd = colmap_bin + ' ' + cmd
    print('\nRunning cmd: ', cmd)

    subprocess.check_call(['/bin/bash', '-c', cmd])


gpu_index = '-1'


def run_sift_matching(img_dir, db_file, remove_exist=False):
    print('Running sift matching...')

    if remove_exist and os.path.exists(db_file):
        os.remove(db_file) # otherwise colmap will skip sift matching

    # feature extraction
    # if there's no attached display, cannot use feature extractor with GPU
    cmd = ' feature_extractor --database_path {} \
                                    --image_path {} \
                                    --ImageReader.single_camera 1 \
                                    --ImageReader.camera_model SIMPLE_RADIAL \
                                    --SiftExtraction.max_image_size 5000  \
                                    --SiftExtraction.estimate_affine_shape 0 \
                                    --SiftExtraction.domain_size_pooling 1 \
                                    --SiftExtraction.use_gpu 1 \
                                    --SiftExtraction.max_num_features 16384 \
                                    --SiftExtraction.gpu_index {}'.format(db_file, img_dir, gpu_index)
    bash_run(cmd)

    # feature matching
    cmd = ' exhaustive_matcher --database_path {} \
                                     --SiftMatching.guided_matching 1 \
                                     --SiftMatching.use_gpu 1 \
                                     --SiftMatching.max_num_matches 65536 \
                                     --SiftMatching.max_error 3 \
                                     --SiftMatching.gpu_index {}'.format(db_file, gpu_index)

    bash_run(cmd)


def run_sfm(img_dir, db_file, out_dir):
    print('Running SfM...')

    cmd = ' mapper \
            --database_path {} \
            --image_path {} \
            --output_path {} \
            --Mapper.tri_min_angle 3.0 \
            --Mapper.filter_min_tri_angle 3.0'.format(db_file, img_dir, out_dir)
 
    bash_run(cmd)


def prepare_mvs(img_dir, sparse_dir, mvs_dir):
    print('Preparing for MVS...')

    cmd = ' image_undistorter \
            --image_path {} \
            --input_path {} \
            --output_path {} \
            --output_type COLMAP \
            --max_image_size 2000'.format(img_dir, sparse_dir, mvs_dir)

    bash_run(cmd)


def run_photometric_mvs(mvs_dir, window_radius):
    print('Running photometric MVS...')

    cmd = ' patch_match_stereo --workspace_path {} \
                    --PatchMatchStereo.window_radius {} \
                    --PatchMatchStereo.min_triangulation_angle 3.0 \
                    --PatchMatchStereo.filter 1 \
                    --PatchMatchStereo.geom_consistency 1 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 15 \
                    --PatchMatchStereo.num_iterations 12'.format(mvs_dir,
                                                                 window_radius, gpu_index)

    bash_run(cmd)


def run_fuse(mvs_dir, out_ply):
    print('Running depth fusion...')

    cmd = ' stereo_fusion --workspace_path {} \
                         --output_path {} \
                         --input_type geometric'.format(mvs_dir, out_ply)
    
    bash_run(cmd)


def run_possion_mesher(in_ply, out_ply, trim):
    print('Running possion mesher...')

    cmd = ' poisson_mesher \
            --input_path {} \
            --output_path {} \
            --PoissonMeshing.trim {}'.format(in_ply, out_ply, trim)

    bash_run(cmd)


def main(img_dir, out_dir, run_mvs=False):
    
    #### run sfm
    sfm_dir = '/gel/usr/mokad6/Desktop/NeRF++/meeting_room_perspective_more_frames/dense/1' # os.path.join(out_dir, 'sfm')
    # os.makedirs(sfm_dir, exist_ok=True)

    sparse_dir = '/gel/usr/mokad6/Desktop/NeRF++/meeting_room_perspective_more_frames/dense/1/sparse' # os.path.join(sfm_dir, 'sparse')
    # extract camera parameters and undistorted images
    os.makedirs(os.path.join(sfm_dir, 'posed_images'), exist_ok=True)
    extract_all_to_dir(sparse_dir, os.path.join(sfm_dir, 'posed_images'))
    normalize_cam_dict(os.path.join(sfm_dir, 'posed_images/kai_cameras.json'),
                       os.path.join(sfm_dir, 'posed_images/kai_cameras_normalized.json'))

if __name__ == '__main__':
    ### note: this script is intended for the case where all images are taken by the same camera, i.e., intrinisics are shared.
    
    img_dir = '/gel/usr/mokad6/Desktop/NeRF++/nerfplusplus/data/meeting_room/images'
    out_dir = '/gel/usr/mokad6/Desktop/NeRF++/nerfplusplus/data/meeting_room'
    run_mvs = False
    main(img_dir, out_dir, run_mvs=run_mvs)

