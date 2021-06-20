function Z = E_DTI_MLE_get_Z(b, g, nDWIs, NrB0, scalingFactor)

if (size(g,1) ~= nDWIs)
    g = [zeros(NrB0,3);g];
end

b_KT = [g(:,1).^4, ...                          % W1111
        g(:,2).^4, ...                          % W2222
        g(:,3).^4, ...                          % W3333
        4*(g(:,1).^3).*g(:,2), ...              % W1112
        4*(g(:,1).^3).*g(:,3), ...              % W1113
        4*(g(:,2).^3).*g(:,1), ...              % W2221
        4*(g(:,2).^3).*g(:,3), ...              % W2223
        4*(g(:,3).^3).*g(:,1), ...              % W3331
        4*(g(:,3).^3).*g(:,2), ...              % W3332
        6*(g(:,1).^2).*(g(:,2).^2), ...         % W1122
        6*(g(:,1).^2).*(g(:,3).^2), ...         % W1133
        6*(g(:,2).^2).*(g(:,3).^2), ...         % W2233
        12*g(:,2).*g(:,3).*(g(:,1).^2), ...     % W1123
        12*g(:,1).*g(:,3).*(g(:,2).^2), ...     % W2213
        12*g(:,1).*g(:,2).*(g(:,3).^2)];        % W3312
    
b_KT = repmat((1/54)*(b(:,1)+b(:,4)+b(:,6)).^2,[1 15]).*b_KT;

Z = [ones(nDWIs,1), -b, b_KT];

Z(:, 2:7) = Z(:, 2:7)./scalingFactor;
Z(:, 8:end) = Z(:, 8:end)./(scalingFactor^2);

end