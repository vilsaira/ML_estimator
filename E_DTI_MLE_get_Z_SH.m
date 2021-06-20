function Z = E_DTI_MLE_get_Z(b, g, nDWIs, NrB0, scalingFactor, L)

A = [g(:,1).^2,...
     2.*g(:,1).*g(:,2),...
     2.*g(:,1).*g(:,3),...
     g(:,2).^2,...
     2.*g(:,2).*g(:,3),...
     g(:,3).^2];
     
bvals = round(b(NrB0+1:end,:) ./ A);
bvals = unique(bvals);

bval = zeros(NrB0,1);
% bval = [];
for i = 1:length(bvals)
    bval = [bval; bvals(i).*ones(nDWIs-NrB0,1)];
end
     
bval = bval ./ scalingFactor;

g(NrB0+1:end+NrB0, :) = g;
g(1:NrB0,:) =0;
% phi = zeros(size(bval));
phi = atan(g(:,2)./g(:,1));
phi(isnan(phi)) = 0;

Z = [];
for l = 0:2:L
    Pl = legendre(l, g(:,3));
    Z = [Z, (Pl(1,:)'.*(bval))];
    for m = 0:1:l
        Nlm = sqrt( factorial(l-m)*(2*l+1)/(factorial(l+m)*2*pi));
        Z_pn = Pl(m+1,:)'*Nlm.*(bval.^l);
        Z = [Z, Z_pn.*cos(m*phi), Z_pn.*sin(m*phi)]
    end
end