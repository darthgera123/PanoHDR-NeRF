function[] = metric(arg1,arg2)
    if ~exist( 'hdrvdp3', 'file' )
        addpath( fullfile( pwd, '..') );
    end
    gtDir = arg1; %gets directory
    predDir = arg2;
    imgs = dir(fullfile(predDir,'*.hdr')); %gets all wav files in struct
    hdrvdp=[];
    ppd = hdrvdp_pix_per_deg( 24, [1920 1080], 1.0 );
%     ssim = [];
    for k = 1:length(imgs)
      baseFileName = imgs(k).name;
      gtImgName = fullfile(gtDir, baseFileName);
      predImgName = fullfile(predDir, baseFileName);
      %fprintf(1, 'Now reading %s\n', gtImgName);
      I_ref = hdrread(gtImgName);
      I_pred = hdrread(predImgName);
      HDR_VDP = hdrvdp3( 'quality', I_pred, I_ref, 'rgb-bt.709', ppd );
      hdrvdp = [hdrvdp,HDR_VDP.Q];
    end
    fprintf( 1, 'HDR_VDP = %g\n', median(hdrvdp));
end