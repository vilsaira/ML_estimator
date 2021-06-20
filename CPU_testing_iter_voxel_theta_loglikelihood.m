function [theta, new_theta, DeltaTheta, loglikelihood, loglikelihood0, expo, exp_scaling, iter_loglikelihood] =...
    CPU_testing_iter_voxel_theta_loglikelihood(theta, I, score, nParams, EN,...
                              Ztheta, ZZ, c, expo,...
                              exp_scaling, loglikelihood, loglikelihood0,...
                              iter_loglikelihood, iter_limit_loglikelihood,...
                              tolerance_loglikelihood)

go_loglikelihood = true;
iter_loglikelihood = 0;

% regularizer params
lambda0 = 1.0e-5;
lambda = lambda0;
rescaling = 5.0;
new_theta = theta(2:end);
while go_loglikelihood
%     loglikelihood0 = loglikelihood;
    iter_loglikelihood=iter_loglikelihood+1;
%   warning on verbose
%   Change warning message for nearlySingularMatrix into error to use try
%   catch to regulate solution
    s = warning('error', 'MATLAB:nearlySingularMatrix');
    warning('error', 'MATLAB:nearlySingularMatrix');
    if (lambda>0.0)
        % Regularized linear solution
       try
          % Regular processing part
%                 I = I + diag(repmat(lambda,1,nParams-1));
%                 Iaug= I;
%                 Iaug(:,end+1) = score;
%                 rank(I)
%                 rank(Iaug)
%                 xls=R\(Q’*y);
%                 To be sure you’ve really computed the least-squares approximate solution, we encourage
%                 you to check that the residual is orthogonal to the columns of A, for example with the
%                 commands
%                 r=A*x-y; % compute residual
%                 A’*r % compute inner product of columns of A and r
%               if (rank(I) == rank(Iaug)) && (rank(I) < length(score))
%                   % infinite number of solutions
%               else
%                   % unique solution
%               end                  
% It = (I + diag(repmat(lambda,1,nParams-1)));
          DeltaTheta = (I + diag(repmat(lambda,1,nParams-1))) \ score;
       catch
          % Exception-handling part         
          DeltaTheta = pinv(I + diag(repmat(lambda,1,nParams-1)))*score;
          % fprintf('Matrix is close to singular or badly scaled. Results may be inaccurate. RCOND = %s)\n', lasterr);
       end
        %DeltaTheta = (I + diag(repmat(lambda,1,nParams-1))) \ score;
        lambda=lambda*rescaling;        % rescaling > 1
    else
        DeltaTheta = I \ score; 
    end;    
    % Restore the warnings back to their previous (non-error) state1
    warning(s);

    new_theta=theta(2:end)+DeltaTheta;
    
    Ztheta=(ZZ*new_theta+c);
    scaling=max(Ztheta);  
    expo=exp(Ztheta-scaling);
    exp_scaling=exp(scaling);    
    loglikelihood= EN'*Ztheta-sum(expo)*exp_scaling;                       

%     go_loglikelihood = ...
%         ((loglikelihood-loglikelihood0)/abs(loglikelihood0) > tolerance_loglikelihood) ...
%         && ...
%         ( iter_loglikelihood < iter_limit_loglikelihood);

% This loop continues until new loglikelihood is larger than the initial
% one i.e. this does not maximize the likelihood w.r.t. theta(1) and
% sigmasq
    go_loglikelihood = ...
        ((loglikelihood<loglikelihood0) && ...
        (iter_loglikelihood < iter_limit_loglikelihood));
end

% theta(2:end) = new_theta;

end