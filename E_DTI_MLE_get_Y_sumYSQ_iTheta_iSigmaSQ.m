function [Y, sumYSQ, iTheta, iSigmaSQ] =...
    E_DTI_MLE_get_Y_sumYSQ_iTheta_iSigmaSQ(brainMask, ...
                                           DWI, ...
                                           Z, NrB0, model)                                           

Y = cellfun(@(x) double(x(brainMask)), DWI, 'UniformOutput', false);
Y = cell2mat(Y)';

% remove physically implausible signals with ExploreDTI PIS approach i.e.
% find maximum high b intensity per voxel and set each b0 intensity to that
    
maxHB = max(Y(NrB0+1:end,:));
for i = 1:NrB0
    inds = Y(i,:) < maxHB;
    Y(i,inds) = maxHB(inds)+eps;
end

sumYSQ = sum( Y.^2, 1);
% Initial theta guess from LLS:

% switch model
%     case 'SH'
%         iTheta = Z \ Y;
%     otherwise       
        iTheta = Z \ log(Y);
% end

% switch model
%     case 'SH'
%         iSigmaSQ = sum((Y - Z*iTheta).^2);
%     otherwise       
        iSigmaSQ = sum((Y - exp(Z*iTheta)).^2);
% end


end