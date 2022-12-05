function[] = pupsnr(arg1,arg2)
    if ~exist( 'pu21_encoder', 'class' )
        addpath( fullfile( pwd, '..') );
    end
    gtDir = arg1; %gets directory
    predDir = arg2;
    imgs = dir(fullfile(predDir,'*.hdr')); %gets all wav files in struct
    psnr = [];
    ssim = [];
    for k = 1:length(imgs)
      baseFileName = imgs(k).name;
      gtImgName = fullfile(gtDir, baseFileName);
      predImgName = fullfile(predDir, baseFileName);
      %fprintf(1, 'Now reading %s\n', gtImgName);
      I_ref = hdrread(gtImgName);
      I_pred = hdrread(predImgName);
      PSNR = pu21_metric( I_pred, I_ref, 'PSNR' );
      SSIM = pu21_metric( I_pred, I_ref, 'SSIM' );
      %fprintf( 1, 'PSNR = %g dB, SSIM = %g\n', PSNR, SSIM );
      psnr = [psnr,PSNR];
      ssim = [ssim,SSIM];
    end
    fprintf( 1, 'PU PSNR = %g dB, PU SSIM = %g\n', mean(psnr), mean(ssim));
end
