function [S0, iter_S0, twotau] = ...
    CPU_testing_iter_voxel_S0(S0, sigmasq, a, b, twotau, iter_S0, iter_limit_S0, tolerance_S0)

go_S0 = true;
iter_S0 = 0;
while go_S0                
    iter_S0 = iter_S0 + 1;
    S00 = S0;
    S0 = log( sum( b .* cpu_besseli( 1, twotau ) ./ ...
        cpu_besseli( 0, twotau ) ) ) - a;
    twotau = b * exp( S0 ) / sigmasq;
    
    go_S0 = ( abs( (S0-S00 ) / S00) > tolerance_S0 ) ...
            && ...
            ( iter_S0 < iter_limit_S0 );            
end       

end