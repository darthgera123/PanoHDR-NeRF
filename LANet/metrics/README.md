# Run metrics
To run the metrics, make sure you have matlab. 
`module load matlab/R2020b`
Make sure all files are in `*.hdr` format
## HDR-VDP 3
+ `cd hdrvdp-3.0.6`
+ `matlab -batch "metric <gt_dir> <pred_dir>`

## PU-PSNR/SSIM
+ Download https://github.com/gfxdisp/pu21.git
+ `mv pupsnr pu21/matlab/examples`
+ `cd pu21/matlab/examples`
+ `matlab -batch "pupsnr <gt_dir> <pred_dir> `
