# PanoHDR-NeRF
Code built on top of [NeRF++](https://github.com/Kai-46/nerfplusplus)
* Work with 360 capture of large-scale unbounded scenes.
* Support multi-gpu training and inference with PyTorch DistributedDataParallel (DDP).
* Work with HDR Panoramas

## Create environment
```bash
conda env create --file environment.yml
conda activate nerfplusplus
```

## Calibration scripts

These scripts are used to extract frames from a video and linearize these frames. The README file will tell you everything on how to use the scripts.

## OpenSfM

The documentation is very useful for installing and using OpenSfM (https://www.opensfm.org/docs/). Here are the steps to run OpenSfM on panoramic images :
+ Create a folder for your dataset in the ```data``` folder. Let's call it the ```dataset``` folder.
+ Create an ```images``` folder inside ```dataset``` and put all the images of your dataset in it.
+ In ```dataset```, create a ```camera_models_overrides.json``` file that follows this format :

    ```json
    {
        "all": { 
            "width": 960,
            "height": 480,
            "projection_type": "equirectangular"
        }
    }
    ```
    ```equirectangular``` or ```spherical``` types should be use for panos
+ OpenSfM will use the default parameters from ```opensfm/config.py```. To change those parameters, you can add a ```config.yaml``` file in the ```dataset``` folder. For example, it could look like this :

    ```yaml
    processes: 16                  # Number of threads to use
    ```
+ You can start the reconstruction with the command 

    ```
    bin/opensfm_run_all data/dataset
    ```
+ Once it is done, you will need the ```reconstruction.json``` file
+ You can also visualize the reconstruction results with the ```undistorted/depthmaps/merged.ply``` file by opening it in MeshLab, for example

## Prepare your dataset

For this step you will need the scripts from the Pano_NeRF repo, your dataset and the ```reconstruction.json``` file from OpenSfM.
+ Run 

    ```
    python colmap_runner/deal_with_openSFM.py --input_model /path/to/reconstruction.json --width 960 --height 480
    ```
    This will a output a `kai_cameras.json` and a `kai_cameras_normalized.json` file. Only `kai_cameras_normalized.json` will be useful.
+ Run

    ```
    python misc/prepare_my_dataset.py --input_img_dir /path/to/your/images/ --input_json_file /path/to/kai_cameras_normalized.json --output_dir data/dataset_name --hasRGB
    ```
    Use ```--hasLinear``` for processing of HDR images.

You should now have a dataset split in `train`, `test` and `validation` folders. You should also have the corresponding pose and intrinsics for each image.

## MaskRCNN

### Installation of detectron 2
+ `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

To generate masks, run:
```
python misc/mask.py --input_dir data/dataset/train/ --output_dir data/dataset/train/
```

## PanoHDR_NeRF

At this point, you should have a dataset split in three (train, test and validation) with the corresponding poses, intrinsics and masks for each image. You will also need a config file for your experiment. You can find examples of config files in `configs/`. To see the available options or to see their description, you can use :

```
python ddp_train_nerf.py --help
```
A quick overview of the options will also be at the end of this section. Here is how to train a model and render panos :
+ To train a model, run
    ```
    python ddp_train_nerf.py --config path/to/config_file.txt
    ```
    By default, NeRF++ trains the model for 500 000 iterations, but you can get a decent model after a few hours depending on your needs.
+ To test you model, run
    ```
    python ddp_test_nerf.py --config path/to/config_file.txt --render_splits test
    ```
    This will render full images at the positions in the dataset's ``test`` folder and compute error metrics if the ground truth is available.

### A few options

For everythong related to the Mip-NeRF paper, see https://jonbarron.info/mipnerf/ for more information

+ `--saveEXR` saves output images in EXR format.
+ `--is_HDR` to train and test on HDR images.
+ `--ipe` to use integrated positionnal encoding as in the Mip-NeRF paper.
+ `--radius_multiplier` to multiply the radius of the cones casted for the IPE by a constant. Normally, you want it set to 1 (default).
+ `--single_mlp` to use the same MLP for coarse and fine sampling, as in Mip-NeRF.
+ `--coarse_loss_mult` how much to downweight the coarse loss(es) when using a single MLP. Normally, you want it set to 0.1 (default).
+ `--alt_sampling` to use the sampling method used in Mip-NeRF.
+ `--sphere_sample` to sample rays uniformly in 3D Cartesian coordinates when training on panos. See https://www.rojtberg.net/1985/how-to-generate-random-points-on-a-sphere/ for more information.
