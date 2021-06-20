function [ output ] = cpu_besseli( nu, z )
% This function is based on approximations given in the numerical recipes
% in C ( http://apps.nrbook.com/c/index.html pp 237 ). Chapter 6.6 Modified
% Bessel Functions of Integer Order

% Since we have to deal only with nu = 0 and nu = 1 this just might work!

% nu must be 0 or 1, z can be scalar or a vector
az = abs(z);
indsLess = az < 3.75;
indsMore = ~indsLess;
indsLessZero = (z < 0.0);
output = zeros(size(z));
if (nu==0)        
    if any(indsLess)
        y = z(indsLess) ./ 3.75;
        y = y.*y;
        output(indsLess) = 1.0+y.*(3.5156229 + y.*(3.0899424 + y.*(1.2067492 + ...
                y.*(0.2659732+y.*(0.360768e-1+y.*0.45813e-2)))));
        output(indsLess) = output(indsLess).*exp(-az(indsLess));
    end
    if any(indsMore)
        y = 3.75 ./ az(indsMore);
        output(indsMore) = (1.0 ./ sqrt(az(indsMore))) .* ...
            (0.39894228 + y.*(0.1328592e-1 ...
            + y.*(0.225319e-2 + y.*(-0.157565e-2 + y.*(0.916281e-2 ...
            + y .*(-0.2057706e-1 + y.*(0.2635537e-1 + y.*(-0.1647633e-1 ...
            + y.*0.392377e-2))))))));
    end
elseif (nu==1)  
    if any(indsLess)
        y = z(indsLess) ./ 3.75;
        y = y.*y;
        output(indsLess) = az(indsLess).* ...
            (0.5 + y.*(0.87890594 + y.*(0.51498869 + y.*(0.15084934 ...
            +y.*(0.2658733e-1 +y.*(0.301532e-2 +y.*0.32411e-3))))));
        output(indsLess) = output(indsLess).*exp(-az(indsLess));
    end
    if any(indsMore)
        y = 3.75 ./ az(indsMore);
        output(indsMore) = 0.2282967e-1 + y.*(-0.2895312e-1 + y.*(0.1787654e-1 ...
            -y.*0.420059e-2));
        output(indsMore) = 0.39894228+y.*(-0.3988024e-1+y.*(-0.362018e-2 ...
            +y.*(0.163801e-2 + y.*(-0.1031555e-1 + y.*output(indsMore)))));
        output(indsMore) = output(indsMore) ./sqrt(az(indsMore));
%         output(indsMore) = output(indsMore) .* exp(az(indsMore))./sqrt(az(indsMore));        
    end      
else
    output = NaN(size(z));
end

    output(indsLessZero) = -output(indsLessZero);
end
