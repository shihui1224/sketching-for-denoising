% A script to learn a GMM prior given the empirical sketch (sk_file.mat)
% Copyright (C) 2021 Shi Hui
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Affero General Public License for more details.

% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

op = 1;
r = 20; k = 20;
skw_file = 'sk_file.mat';
disp('loading sketch file');
load(skw_file,'skw');
estmix = gmm_estimator_n(skw, r, op, 'LBFGS_smallNbIt', 500, 'LBFGS_bigNbIt', 3000);
disp('Estimate mixture model...');
tstart = tic;
[params, wts, nres] = estmix.estim(k);
dim = estmix.d;
nmodels = k;
X = reshape(params, estmix.d, estmix.r, k);
for i = 1:k
    [u, s] = svd(X(:,:,i));
    U(i) = {u};
    S(i) = {cat(1, diag(s.^2), zeros(dim - r, 1))};
    nu(i) = {2*ones(dim, 1)};
end
GS  = struct('U', {U}, 'S', {S}, 'dim', dim, 'nmodels', nmodels, 'wts', wts, 'nu', {nu});
t = toc(tstart);
disp(['...done. Time: ' num2str(t)]);
save([num2str(op) 'sketch_prior.mat'],'GS', 'nres','t')


