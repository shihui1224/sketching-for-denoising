% function [val,grad] = gmmcostfun_n(estimator,z,thetas,alpha)
%
% Global cost function and its gradient efficient for GMMs with zero-means and non-diagonal covariance matrices, ie a fast implementation of "genericcostfun" specialized in the case of GMMs.
%
% Copyright (C) 2021 Hui Shi. 
% This file is can be used with the SketchMLbox (Sketching for Mixture Learning toolbox). 
%
% The SketchMLbox is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version. Copyright (C) 2016 Nicolas Keriven. 
%



function [val,grad] = gmmcostfun_n(estimator,sketch,params,weights)
    k = size(params,2);
    W = estimator.sk_wrapper.W;
    r = estimator.r;
    d = estimator.d;
    m = estimator.m;

    bigphi = zeros(2*m, k);
    wx = W'*reshape(params, d, r*k);
    a = reshape(wx', r, m*k);
    bigphi(1:m, :) = exp(-.5* reshape(dot(a, a), k, m))';

    res = bigphi*weights - sketch;
    val = res'*res;
    b = reshape(wx, m*r, k) .* repmat(bigphi(1:m, :) .* res(1:m), r, 1);
    grad = W * reshape(b, m, r*k);
    grad = -2 * weights'.*reshape(grad, d*r,k);
    grad = grad(:);
    grad = ([grad; 2*(bigphi' * res)]);

end
