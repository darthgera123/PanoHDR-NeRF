# LANet
Pytorch port of Modified [Luminance Attention Network](https://github.com/LWT3437/LANet)

## Training
`python train.py  --train_ldr <ldr_data> --train_hdr <hdr_data> --val_ldr <ldr_data> --val_hdr <hdr_data>  --exp_name augs --epochs <epochs_number> --isBottom`

## Inference
`python test.py --ckpt <ckpt_dir> --test_dir <input_dir> --output_dir <output_dir>`

## Metrics
We compare the results on Render-MSE, PU-PSNR and HDR-VDP3. The scripts for comparing them are in `metrics`. 