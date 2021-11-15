function zhat = fepll_per_patch(ztilde, sigma2, prior_model, varargin)
% Inputs:
%   ztilde      : patches extracted from noisy image
%   sigma2      : noise variance
%   prior_model : model generated using get_prior_model.m
%   varargin    : refer to retrieve arguments for a list
%
% Outputs:
%   zhat        : restored image patches
% ________________________________________

options      = makeoptions(varargin{:});

mu = getoptions(options, 'mu', 0);

P2           = size(ztilde, 1);

switch prior_model.name
    case 'Sketching'
        truncation = 1;
        
    case 'EM'
        truncation = 0;
end
% Remove DC component
zdc          = mean(ztilde);
ztilde       = bsxfun(@minus, ztilde, zdc);

% Gaussian selection

labels = gs_match(ztilde, prior_model.GS, sigma2, mu, truncation);


% Patch estimation
U          = prior_model.GS.U;
S          = prior_model.GS.S;
lab_list   = unique(labels(:))';
zhat       = zeros(size(ztilde));


for k = lab_list
    inds = labels == k;
    if truncation
        t = 20;
        gammaj        = S{k}(1:t) ./ (S{k}(1:t) + sigma2);
        gammaP        = mu  ./ (mu + sigma2);
        zt            = ztilde(:, inds);
        ctilde        = U{k}(:, 1:t)' * zt;
        chat          = bsxfun(@times, gammaj - gammaP,  ctilde);
        zhat(:, inds) = U{k}(:, 1:t) * chat + gammaP * zt;
    else
        gammaj        = S{k} ./ (S{k} + sigma2);
        ctilde        = U{k}' * ztilde(:, inds);
        chat          = bsxfun(@times, gammaj,  ctilde);
        zhat(:, inds) = U{k} * chat;
    end

end

% Add back DC component
zhat = bsxfun(@plus, zhat, zdc);
