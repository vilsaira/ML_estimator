function [sigmasq, iter_sigmasq] = CPU_testing_iter_sigmasq(sigmasq,...
                                       a, b, iter_sigmasq, ...
                                       iter_limit_sigmasq,...
                                       tolerance_sigmasq,...
                                       nDWIs)

go_sigmasq = true;
iter_sigmasq = 0;

while go_sigmasq
    iter_sigmasq = iter_sigmasq + 1;
    sigmasq0 = sigmasq;                
    twotau = b / sigmasq;
    sigmasq = 0.5 * a ./ (nDWIs + sum( twotau .* ...
        cpu_besseli( 1, twotau ) ./ ...
        cpu_besseli( 0, twotau ) ) );

    go_sigmasq = ( abs(sigmasq-sigmasq0) > tolerance_sigmasq ) ...
                 && ...
                 ( iter_sigmasq < iter_limit_sigmasq );
    
end

end