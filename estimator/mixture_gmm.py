import numpy as np


class MixtureGmm:
    def __init__(self, mu, sigma, weights):
        self.mu = mu
        self.Sigma = sigma
        self.weights = weights
        self.k, self.d = mu.shape

    def shift(self, s):
        self.mu = self.mu + s

    def normalize(self, m):
        self.mu = m*self.mu
        if self.k == 1:
            self.Sigma = np.diag(m)@self.Sigma@np.diag(m)
        else:
            for l in range(self.k):
                self.Sigma[l, :, :] = np.diag(m)@self.Sigma[l, :, :]@np.diag(m)

    # FOR ISOTROPIC GMM
    def mean_var(self):
        mv = 0
        if self.k == 1:
            s = np.linalg.eigvals(self.Sigma)
            mv = mv + np.sum(s)
        else:
            for l in range(self.k):
                s = np.linalg.eigvals(self.Sigma[l, :, :])
                mv = mv + np.sum(s)
        mv = mv/(self.d*self.k)
        return mv
