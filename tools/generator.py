import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from estimator import mixture_gmm

def draw_gmm(truegmm, n):
    mu = truegmm.mu
    sigma = truegmm.Sigma
    wts = truegmm.weights
    k, d = mu.shape
    X = np.zeros([d, n])
    label = random.choices(range(k), weights=wts, k=n)
    for i in range(n):
        j = label[i]
        x = np.random.multivariate_normal(mu[j], sigma[j])
        X[:,i] = x
    return X


def generate_gmm(d, k):
    weights = np.random.rand(k)
    weights = weights/sum(weights)
    Sigma = np.zeros([k, d, d])
    mu = np.zeros([k, d])
    for i in range(k):
        y = np.random.randn(1,d)
        y = y/np.linalg.norm(y)
        sig = np.transpose(y)@y + 0.02*np.eye(d)
        Sigma[i, :, :] = sig
    GMM = mixture_gmm.MixtureGmm(mu, Sigma, weights)
    return GMM


def plot_results(X, means, covariances, cmap, title):
    splot = plt.subplot()
    plt.scatter(X[0, :], X[1, :], .2)
    means = means[:, :2]
    covariances = covariances[:, :2, :2]
    for i, (mean, covar) in enumerate(zip(
            means, covariances)):
        v, vecs = np.linalg.eigh(covar)
        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        width, height = 2*2*np.sqrt(v)
        ell = mpl.patches.Ellipse(xy = mean, width = width, height= height, angle=angle, color=cmap(i))
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)
    plt.xlim(-3., 3.)
    plt.ylim(-3., 3.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
