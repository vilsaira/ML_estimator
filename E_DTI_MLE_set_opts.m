function MLEopts = E_DTI_MLE_set_opts

MLEopts.tolerance_sigmasq = 1e-4;
MLEopts.tolerance_theta = 1e-4;
MLEopts.tolerance_S0 = 1e-4;
MLEopts.tolerance_loglikelihood = 1e-5;
% MLEopts.tolerance_rice = 1e-27
MLEopts.iter_limit_sigmasq = 100;
MLEopts.iter_limit_S0 = 100;
MLEopts.iter_limit_voxel = 10;
MLEopts.iter_limit_theta = 10;
MLEopts.iter_limit = 50; % This can't be large due regulator lambda (1.0e-5*5^(iter))
MLEopts.lambda0 = 1e-5;
MLEopts.rescaling = 5.0; 

end