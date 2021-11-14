op = 1;
r = 20; k = 20;
skw_file = 'sk_file.mat';
disp('loading sketch file');
load(skw_file,'skw');
estmix = gmm_estimator_n(skw, r, op, 'LBFGS_smallNbIt', 5000, 'LBFGS_bigNbIt', 300);
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


