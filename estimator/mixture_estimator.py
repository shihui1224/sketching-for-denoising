import abc
import numpy as np
from scipy.optimize import minimize


class MixtureEstimator:

    def __init__(self, sk_wrapper, options):
        self.sk_wrapper = sk_wrapper
        self.d = sk_wrapper.d
        self.m = sk_wrapper.m
        self.p = 0
        self.mixture = None
        self.options = options

    def scal_prod(self, param, res):
        phi, jphi = self.sketch_distrib(param)
        nphi = np.linalg.norm(phi) + 1e-10
        phi = phi / nphi
        val = phi @ res
        return -val

    def grad_scal_prod(self, param, res):
        phi, jphi = self.sketch_distrib(param)
        nphi = np.linalg.norm(phi) + 1e-10
        phi = phi / nphi
        val = phi @ res
        grad = jphi(res - val * phi) / nphi
        return -grad

    def local_min(self, res):
        v = self.init_param()
        # opt = minimize(self.scal_prod, x0=v, args=(res), method="Newton-CG", jac=self.grad_scal_prod)

        opt = minimize(self.scal_prod, x0=v, args=res, method="L-BFGS-B", jac=self.grad_scal_prod,
                       options={'maxcor': 5, 'gtol': 1e-20, 'maxiter': 100, 'maxls': 50})
        params = opt.x
        return params

    def ideal_sketch_multi(self, params):
        k = len(params)
        val = np.zeros([2 * self.m, k])
        for i in range(k):
            val[:, i] = self.sketch_distrib(params[i])[0]
        return val

    def costweights(self, weights, sketch, bigphi):
        k = weights.shape[0]
        bigphi = np.reshape(bigphi, [k, 2 * self.m])
        r = weights @ bigphi - sketch
        val = np.transpose(r) @ r
        return val

    def grad_costweights(self, weights, sketch, bigphi):
        k = weights.shape[0]
        bigphi = np.reshape(bigphi, [k, 2 * self.m])
        r = weights @ bigphi - sketch
        # r = bigphi @ weights - sketch
        grad = bigphi @ r
        return grad

    def proj_cone(self, sketch, params):
        bigphi = self.ideal_sketch_multi(params)  # 2m, k
        bigphi = bigphi.flatten()
        k = len(params)
        x0 = np.ones(k) / k
        opt = minimize(self.costweights, x0, args=(sketch, bigphi), method="L-BFGS-B", jac=self.grad_costweights,
                       bounds=[(self.options.min_weight, 1)] * k, options={'maxcor': 5, 'maxiter': 100, 'maxls': 50})
        weights = opt.x
        return weights

    def costfun(self, x, k, sketch):
        """"""
        params = np.reshape(x[:-k], [k, self.p])
        weights = x[-k:]
        bigphi = np.zeros([2 * self.m, k])
        for i in range(k):
            bigphi[:, i], _ = self.sketch_distrib(params[i, :])
        res = bigphi @ weights - sketch
        val = np.transpose(res) @ res
        return val

    def grad_costfun(self, x, k, sketch):
        params = np.reshape(x[:-k], [k, self.p])
        weights = x[-k:]
        bigphi = np.zeros([2 * self.m, k])
        bigjphi = []
        for i in range(k):
            bigphi[:, i], g = self.sketch_distrib(params[i, :])
            bigjphi.append(g)
        res = bigphi @ weights - sketch
        grad = np.zeros_like(params)
        for i in range(k):
            grad[i, :] = 2 * weights[i] * bigjphi[i](res)
        grad = grad.flatten()
        grad = np.hstack((grad, 2 * np.transpose(bigphi) @ res))
        return grad

    def adjust_update(self, params, weights, niter):
        k = len(params)
        sketch = self.sk_wrapper.sk_emp
        weights[weights < self.options.min_weight] = self.options.min_weight
        params = np.array(params)
        params = np.squeeze(params.flatten())
        x0 = np.hstack((params, weights))
        opt_result = minimize(fun=self.costfun, x0=x0, method="L-BFGS-B", args=(k, sketch), jac=self.grad_costfun,
                              bounds=[(-np.inf, np.inf)] * self.p * k + [(self.options.min_weight, 1)] * k,
                              options={'maxcor': 5, 'maxiter': niter})

        xk = opt_result.x
        # print(opt_result.success)
        # print(opt_result.message)
        params = np.reshape(xk[:-k], [k, self.p])
        weights = xk[-k:]
        """final projection"""
        weights[weights < self.options.min_weight] = self.options.min_weight

        """update"""
        params_list = params.tolist()
        sk = self.ideal_sketch_multi(params_list)  # 2m, k
        sk = sk @ weights
        res = sketch - sk
        return params_list, weights, res

    def LROMP(self, k):
        """initialisation"""
        res = self.sk_wrapper.sk_emp
        params = []
        nres = []
        for i in range(k):
            """search 1 atom"""
            paramsup = self.local_min(res)
            """expand support"""
            params.append(paramsup)
            """back projection """
            weights = self.proj_cone(self.sk_wrapper.sk_emp, params)
            """adjust and update residual"""
            params, weights, res = self.adjust_update(params, weights, self.options.sm_niter)
            nres.append(np.linalg.norm(res))

        """final adjustment"""
        params, weights, res = self.adjust_update(params, weights, self.options.big_niter)
        weights /= np.sum(weights)
        nres.append(np.linalg.norm(res))
        return params, weights, nres

    def estimate(self, k):
        params, weights, nres = self.LROMP(k)
        mixture = self.construct_mixture(params, weights)
        self.mixture = mixture
        return nres

    @abc.abstractmethod
    def construct_mixture(self, params, weights):
        """construct mixture from parameters and weights"""
        pass

    @abc.abstractmethod
    def sketch_distrib(self, param):
        """return v, grad_v"""
        pass

    @abc.abstractmethod
    def init_param(self):
        pass



