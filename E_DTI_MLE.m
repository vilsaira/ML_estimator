% This function is written to work with ExploreDTI dataformat

% Way to find which functions call edti stuff 
% [fList,pList] = matlab.codetools.requiredFilesAndProducts('F:\M_Code_ExploreDTI_v4.8.6\Source\Stuff\E_DTI_Get_DT_KT_from_DWI_b_mask.m')

[ fileName, filePath ] = uigetfile( '*.mat', 'ExploreDTI mat file' );
%%
scalingFactor = 1;
filePath = 'f:\biling\biling_tms_19\';
fileName = 'data_DKI.mat';
% filePath = 'c:\viljami\Kallosauma_DTI';
% fileName = '02_dwi_GR_TV_FOV_c_MD_C_trafo.mat';
fin = strcat(filePath, filesep, fileName);
% fout = strcat(filePath, filesep, fileName(1:end-4), '_MLE.mat');
fout = 'test_1.mat';
EDTI = load(fin);

brainMask = ~isnan(EDTI.FA);
nVoxels = sum(brainMask(:));
nDWIs = length(EDTI.DWI);

model = 'SH';
lmax = 6;
switch model
    case 'DT'
        Z = E_DTI_MLE_get_Z(EDTI.b, EDTI.g, length(EDTI.DWI), EDTI.NrB0, scalingFactor);
        Z = Z(:,1:7); % DT model
        DWI = EDTI.DWI;
    case 'KT'
        Z = E_DTI_MLE_get_Z(EDTI.b, EDTI.g, length(EDTI.DWI), EDTI.NrB0, scalingFactor);
        DWI = EDTI.DWI;
    case 'SH'
        % select single shell without b0
        scaling = 100;
        bval=round(sum(EDTI.b(:,[1 4 6]),2)/100)*100;
        b0inds = (bval == 0);
        bval(b0inds) = [];
        bvec = EDTI.g;                     
%         bvec = [zeros(EDTI.NrB0, 3); bvec];
        Z =  sphericalHnew_vs(bvec, bval, lmax,scaling);
        DWI = EDTI.DWI(EDTI.NrB0+1:end);
end

%% initialization
MLEopts = E_DTI_MLE_set_opts; 
if strcmp(model, 'SH')    
    brainMask = brainMask & (EDTI.FA/sqrt(3) > 0.05);
end

[Y, sumYSQ, iTheta, iSigmaSQ] =...
    E_DTI_MLE_get_Y_sumYSQ_iTheta_iSigmaSQ(brainMask, DWI, Z, EDTI.NrB0, model);
%% CPU approach
mlThetaCPU = iTheta;
mlSigmaSQCPU = iSigmaSQ;
nParams = size(Z,2);
for nVoxel = 1:length(Y)
    %%
    nVoxel = 3500;
    theta = iTheta(:,nVoxel);
    sigmasq = iSigmaSQ(nVoxel);
    expZtheta = exp(Z*theta);
    a = sumYSQ(nVoxel) + expZtheta'*expZtheta;
    b = Y(:, nVoxel) .* expZtheta;
    [sigmasq, iter_sigmasq] = CPU_testing_iter_sigmasq(...
        sigmasq, a, b, 0, MLEopts.iter_limit_sigmasq,...
        MLEopts.tolerance_sigmasq, nDWIs);
    %%
    [theta, sigmasq, ~, ~, ~, ~] =...
        CPU_testing_iter_voxel(theta, sigmasq, Z, Y(:, nVoxel), nParams,...
                               sumYSQ(nVoxel), nDWIs,...
                               0, MLEopts.iter_limit_sigmasq, MLEopts.tolerance_sigmasq,...
                               0, MLEopts.iter_limit_voxel, MLEopts.tolerance_theta,...
                               0, MLEopts.iter_limit_theta, MLEopts.tolerance_theta,...
                               0, MLEopts.iter_limit_S0, MLEopts.tolerance_S0,...
                               0, MLEopts.iter_limit,...
                               MLEopts.tolerance_loglikelihood, nVoxel);
    mlThetaCPU(:, nVoxel) = theta;
    mlSigmaSQCPU(:, nVoxel) = sigmasq;
end

%% GPU ----------------------------------------------------------------------------- GPU
if ispc % test if script is running on Windows PC
    % windows registry TdrDelay is 2 seconds by default, this has to be
    % increased for GPU computing  
    try
        TdrDelayReg = winqueryreg('HKEY_LOCAL_MACHINE', 'System\CurrentControlSet\Control\GraphicsDrivers', 'TdrDelay');
    catch
        % by default this key is not used and thus not found
        TdrDelayReg = 2;
    end
    if TdrDelayReg < 30*60 % 30 minutes
        regFilePath = [pwd, '\MLE_reg.reg'];
        regKeyPath = '[HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers]';
        fp = fopen(regFilePath, 'wt');
        fprintf(fp, 'Windows Registry Editor Version 5.00\n');
        fprintf(fp, '%s\n', regKeyPath);
        fprintf(fp, '%s\n', '"TdrDelay"=dword:00000708');
        fclose(fp);
        [status, cmdout] = system(regFilePath, '-echo');
        if status~=0
            error(['Could not edit Windows registry automatically', char(10),...
                   'Run ', regFilePath, ' or manually edit registry', char(10),...
                   'HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers', char(10),...
                   'add dword named "TdrDelay" and set proper value']);
        else
            delete(regFilePath);
        end
    end
end
%%
[mlTheta, mlSigmaSQ] = MLEfun_EDTI(Y, Z, iTheta, iSigmaSQ, sumYSQ, MLEopts);
% [mlTheta, mlSigmaSQ] = MLEfun_EDTI(Y(:,1), Z, iTheta(:,1), iSigmaSQ(1), sumYSQ(1), MLEopts);

%%
% for i = 1:5
% [mlTheta(2:7,i)-iTheta(2:7,i)./scalingFactor]
% pause
% end  
%
%save results to new mat file
MLE_SigmaSQ = EDTI.FA;
MLE_SigmaSQ(brainMask) = mlSigmaSQ;
DT = EDTI.DT;
for i = 1:6
    DT{i}(brainMask) = mlTheta(i+1, :);
end
if isfield(EDTI, 'KT')
    KT = EDTI.KT;
    for i = 1:15
        KT{i}(brainMask) = mlTheta(i+7, :);
    end
end
DWIB0 = EDTI.DWIB0;
DWIB0(brainMask) = mlTheta(1,:);
[FEFA, FA, FE, SE, eigval] = E_DTI_eigensystem_analytic(DT);


DWI = EDTI.DWI;
b = EDTI.b;
VDims = EDTI.VDims;
MDims = EDTI.MDims;
bval = EDTI.bval;
g = EDTI.g;
info = EDTI.info;
NrB0 = EDTI.NrB0;
chi_sq = [];
chi_sq_iqr = [];

save(fout,'DT');
if isfield(EDTI, 'KT');
    save(fout,'KT','-append');
end

save(fout,'MLE_SigmaSQ', '-append');
save(fout,'DWI','-append');
save(fout,'b','-append');
save(fout,'FE','-append');
save(fout,'SE','-append');
save(fout,'FA','-append');
save(fout,'FEFA','-append');
save(fout,'VDims','-append');
save(fout,'MDims','-append');
save(fout,'eigval','-append');
save(fout,'bval','-append');
save(fout,'g','-append');
save(fout,'info','-append');
save(fout,'NrB0','-append');
save(fout,'chi_sq','-append');
save(fout,'DWIB0','-append');

if isfield(EDTI, 'par');
    par = EDTI.par;
    save(fout, 'par', '-append');
end
if isfield(EDTI, 'DM_info')
    DM_info = EDTI.DM_info;
    save(fout, 'DM_info', '-append');
end

% save GPU info
G = gpuDevice;
GPUinfo.tolerance.sigmasq = MLEopts.tolerance_sigmasq;
GPUinfo.tolerance.theta = MLEopts.tolerance_theta;
GPUinfo.tolerance.S0 = MLEopts.tolerance_S0;
GPUinfo.tolerance.loglikelihood = MLEopts.tolerance_loglikelihood;
GPUinfo.iter_limit.sigmasq = MLEopts.iter_limit_sigmasq;
GPUinfo.iter_limit.S0 = MLEopts.iter_limit_S0;
GPUinfo.iter_limit.voxel = MLEopts.iter_limit_voxel;
GPUinfo.iter_limit.theta = MLEopts.iter_limit_theta;
GPUinfo.iter_limit.regulator = MLEopts.iter_limit;
GPUinfo.rescaling = MLEopts.rescaling;
GPUinfo.lambda0 = MLEopts.lambda0;
GPUinfo.Name = G.Name;
GPUinfo.SupportsDouble = G.SupportsDouble;
GPUinfo.ComputeCapability = G.ComputeCapability;
GPUinfo.DriverVersion = G.DriverVersion;
GPUinfo.ToolkitVersion = G.ToolkitVersion;
GPUinfo.Z = Z;

save(fout, 'GPUinfo', '-append');
%% save SigmaSQ as mnifti
nii = load_untouch_nii('F:\biling\biling_tms_19\data_DKI_MK.nii');
niifout = [fout(1:end-4), '_SigmaSQ.nii'];

I = MLE_SigmaSQ;
I = flipdim(I,1);
I = flipdim(I,2);
I = permute(I,[2 1 3]);


nii.img = I;
nii.fileprefix = [fout(1:end-4), '_SigmaSQ'];

save_untouch_nii(nii, niifout)

%% save SH as nifti
nii = load_untouch_nii('F:\biling\biling_tms_19\data_DTI_HARDI_Lmax_8_CSD_RF_NSim_0.8_8.nii');
niifout = [fout(1:end-4), '_SH_' ,num2str(lmax), '_iMLE.nii'];

nParams = size(iTheta,1);
SH = zeros([size(EDTI.FA), nParams]);

for i = 1:nParams
    tmp = EDTI.FA; % Get NaN mask
%     tmp(brainMask) = mlTheta(i,:);
    tmp(brainMask) = iTheta(i,:);
    SH(:,:,:,i) = tmp;
end

I = SH;
I = flipdim(I,1);
I = flipdim(I,2);
I = permute(I,[2 1 3 4]);


nii.img = I;
nii.hdr.dime.dim(5) = nParams;
nii.fileprefix = [fout(1:end-4), '_SH_' ,num2str(lmax), '_MLE'];

save_untouch_nii(nii, niifout)