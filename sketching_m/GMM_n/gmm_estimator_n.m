%% gmm (zero-means, covariance non-diagonal) estimator %%.
% Copyright (C) 2021 Hui Shi. 
% This file is can be used with the SketchMLbox (Sketching for Mixture Learning toolbox). 
%
% The SketchMLbox is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version. Copyright (C) 2016 Nicolas Keriven. 


classdef gmm_estimator_n<mixture_estimator

properties 
           r; % the disired rank
           op;
end
methods
    %% build
    % sk_wrapper : see sketch_wrapper.m
    % varargin : options, can be passed as couple 'name','value' %   see load_estimator_options.m for available options and default
    %   values
    function self=gmm_estimator_n(sk_wrapper, r, op, varargin)
        self@mixture_estimator(sk_wrapper, varargin{:});
        self.r = r;
        self.p = self.d*self.r;
        self.lb = -inf*ones(self.p, 1);
        self.ub = inf*ones(self.p, 1);
        % optional, for speed-up
        self.costfun = @gmmcostfun_n;
        self.op = op;
    end
    %% estim
    
    %override self.estim to include the algorithm LR-OMP
    function [params, weights, nres] = estim(self,k)
        [params, weights, nres] = self.LR_OMP(k);
        mix = self.construct_mixture(params, weights);
        self.mixture = mix;

    end
    
    function [params,weights,nres] = LR_OMP(self,k)           
        % init
        res = self.sk_wrapper.sk_emp; % residual
        params = [];
        for i = 1:k
            % search one atom (original)
            paramsup = self.local_min(res);
            % expand support
            params = [params paramsup];
            % project to find weights
            weights = self.proj_weights(self.sk_wrapper.sk_emp, params);
            % adjust and update residual
            [params, weights, res] = self.adjust_update(self.op, params, weights, self.options.LBFGS_smallNbIt);
            nres(i)=norm(res);                   
        end
        % final adjustment
        [params, weights, res] = self.adjust_update(self.op, params, weights, self.options.LBFGS_bigNbIt);
        nres(end+1) = norm(res);
        weights = weights / sum(weights);
    end

%% step 1

    function params = local_min(self, res)         
        v = self.init_param();
        try % precaution
            opts = struct('x0', v, 'pgtol', 1e-20, 'maxIts', 500);
            opts.printEvery = Inf;
            [~, params] = evalc('lbfgsb(@(p)self.scal_prod(p,res),self.lb,self.ub,opts);');
        catch ME
            disp(ME.message);
            params = v;
        end
    end
    
    function [val, grad] = scal_prod(self, param, res)
        [phi_re, jphi] = self.sketch_distrib(param);
        nphi = norm(phi_re);
        phi_re = phi_re/nphi;
        val = phi_re'*res(1:self.m);
        grad = jphi(res(1:self.m) - val * phi_re) / nphi;
        grad = grad(:);
        val = -val;
        grad = -grad;
     end
    
 
%% step 2 

    function weights = proj_weights(self, sketch, params)
        %v = self.sketch_distrib_k(params);
        %weights = inv(v' * v) * v' * sketch(1:self.m);
        K = size(params, 2);
        bigphi = zeros(2*self.m, K);
        bigphi(1:self.m, :) = self.sketch_distrib_k(params);
        opts = struct('x0', ones(K, 1)/K);
        [~, weights] = evalc('lbfgsb(@(w)self.costweights(w, sketch, bigphi), zeros(K, 1), ones(K, 1), opts);');    
    end
    
    function [val, grad] = costweights(self, weights, sketch, bigphi)
        res = bigphi * weights - sketch;
        val = res' * res;
        grad = 2 * bigphi' * res;
    end
        
%% step 3

    function [params, weights, res] = adjust_update(self, op, params, weights, nb_it)
    	k = size(params, 2);
    	sketch = self.sk_wrapper.sk_emp;
    	fun = @(x)(self.costfun(self, sketch, reshape(x(1:end-k), self.p, k), x(end-k+1:end)));

        if op == 0

    	    opts = struct('maxIts', nb_it, 'maxTotalIts', 1000*nb_it, 'x0', [params(:); weights]);
    	    opts.printEvery = Inf;
    	    l = [repmat(self.lb, k, 1); zeros(k, 1)];
    	    u = [repmat(self.ub, k, 1); ones(k, 1)];
    	    xk = lbfgsb(fun, l, u, opts);
    	else 
            [f, xk] = self.b_fista(fun, [params(:); weights], nb_it);
        end
        params = reshape(xk(1:end-k), self.p, k);
    	weights = xk(end-k+1:end);
    	res = sketch - cat(1, self.sketch_distrib_k(params) * weights, zeros(self.m, 1));
    end
  
    

    %% Required methods
%         given parameters of a mixture Θ ∈ R^ (p+p*r)×K
%         that contains parameters θk ∈ R^(p + p*r) and weights α ∈ R ^K ,
%         this function returns the corresponding mixture.
    function mix = construct_mixture(self, params, weights)
        k = size(params, 2);
        mu = zeros(self.d, k);
        Sigma = zeros(self.d, self.d,k);
        X = reshape(params, [self.d, self.r, k]);
        for l = 1:k
            x = X(:, :, l);
            Sigma(:,:,l) = x*x';               
        end
        mix=mixture_gmm(mu,Sigma,weights);
    end

%         Given a mixture, returns the corresponding parameters Θ, α
    function [params, weights] = toparams(self, mix)
        weights = mix.weights;
        params = zeros(self.p, length(weights));
        for l = 1:length(weights)
            XXt = mix.Sigma(:,:,l);
            [u, s] = svds(XXt, self.r);               
            X = u*sqrt(s);               
            params(:, l) = X(:);
        end
    end

    function v1 = init_param(self)  
    	v1 = self.sk_wrapper.mean_var * (0.5 + randn(self.d, self.r));
    	v1 = v1(:);       
   end

    function phi_re = sketch_distrib_k(self, params)
        W = self.sk_wrapper.W;
        [~, k] = size(params);
        wx = W' * reshape(params, self.d, self.r*k);
        a = reshape(wx', self.r, self.m * k);
        phi_re = exp(-.5 * reshape(dot(a,a), k, self.m))';
    end

    function [phi_re, jphi] = sketch_distrib(self, param)
        W = self.sk_wrapper.W; % d*m
        m = self.m;
        XW = reshape(param,[self.d, self.r])'*W;     
        phi_re = exp(-.5 * dot(XW, XW)');
    	jphi = @(x)(-W*(XW'.*phi_re.*x));       
    end

    function [f,xopt] = b_fista(self, fun, x0, nb_it)
        l0 = 10^(15); eta = 2; y = x0; t = 1; x = x0;
        l = l0;
        
        for k = 1: nb_it
            [f, g] = fun(y);
            %i_k = 0;
            lbar = l;
            while true
                ply = y-g/lbar;
                r = ply-y;
                Q = f+dot(r,g)+norm(r, 'fro').^2*lbar*0.5;
                [F, ~] = fun(ply);
                if F <= Q
                    break;
                end
                lbar = lbar*eta;
                l = lbar;
            end
            x_new = y-g/l;
            t_new = 0.5*(1+sqrt(1+4*t.^2));
            y_new = x_new +t/t_new*(x_new - x);
            x = x_new;
            t = t_new;
            y = y_new;
        end
        xopt = x;

        f = fun(xopt);
    end
end
end


