function labels = gs_match(y, GS, sig2, mu, truncation)

% Inputs:
%   y           : matrix containing image patches
%   GS          : GMM model
%   sig2        : noise variance
%   mu          : user parameter for denoising
%   truncation  : true for prior model learned by the sketching, false for 
%                  prior model learned by EM
    
%
% Outputs:
%   labels      : index of Gaussian components each patch belongs to


numMix = length(GS.S);

energy = zeros(numMix, size(y,2));
for k = 1:numMix
    iSPlusSig2   = 1 ./ (GS.S{k} + sig2);
    if ~truncation
        uy = GS.U{k}' * y;
        energy(k,:) = gmm_distance(uy, iSPlusSig2, ...
                                      -2*log(GS.wts(k)));
    else
        t = 20;

        uy           = GS.U{k}(:,1:t)' * y;
        energy(k, :)   = gmm_distance(uy, iSPlusSig2(1:t), ...
                                      -2 * log(GS.wts(k))) ;
        energy(k, :) = energy(k,:)  - 1./ (sig2 + mu)* sum(uy.^2);
      
    end
end

[~, labels] = min(energy, [],1);