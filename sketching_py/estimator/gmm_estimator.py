import numpy as np
from estimator.mixture_estimator import MixtureEstimator
from estimator.mixture_gmm import MixtureGmm
import argparse as ap

class GmmEstimator(MixtureEstimator):

    def __init__(self, sk_wrapper, options, r):
        super().__init__(sk_wrapper, options)
        if r > self.d:
            raise Exception("The rank should be lower than the dimension")
        self.r = r
        self.p = self.d * self.r

    def construct_mixture(self, params, weights):
        k = len(params)
        mu = np.zeros([k, self.d])
        Sigma = np.zeros([k, self.d, self.d])
        for i in range(k):
            x = np.reshape(params[i], [self.r, self.d])
            Sigma[i, :, :] = np.transpose(x)@x
        mix = MixtureGmm(mu, Sigma, weights)
        return mix

    def init_param(self):
        v = self.sk_wrapper.mean_var*(0.5 + np.random.randn(self.p))
        return v

    def sketch_distrib(self, param):
        w = np.transpose(self.sk_wrapper.W)
        # w : d*m
        x_r = np.reshape(param, [self.r, self.d])
        xw = x_r@w
        w2 = np.zeros([self.p, self.m])
        for i in range(self.r):
            w2[i*self.d: (i+1)*self.d, :] = xw[i, :]*w
        phi1 = np.exp(-0.5 * np.diag(np.transpose(xw) @ xw))

        phi = np.hstack((phi1, np.zeros(self.m)))
        jphi = lambda x: -w2@(phi1*x[:self.m])
        return phi, jphi


def load_estimator_options(estimator_options=None):
    p = ap.ArgumentParser()
    p.add_argument('-sm_niter', default=300, type=int)
    p.add_argument('-big_niter', default=3000, type=int)
    p.add_argument('-min_weight', default=0, type=int)
    args = p.parse_args(estimator_options)
    return args